from __future__ import annotations

import hashlib
import http.client
import json
import os
import re
import time
import urllib.error
import urllib.request
from typing import Any, Optional, Sequence

from metrics_rebuild.share.paths import CONFIG_PATH, LLM_CACHE_PATH

DEFAULT_LLM_TIMEOUT_SECONDS = 90
DEFAULT_LLM_MAX_ATTEMPTS = 3
DEFAULT_LLM_MAX_TOKENS = 1024

_LLM_CONFIG_CACHE: Optional[tuple[tuple[Optional[int], str, str, str, str], dict]] = None
_LLM_MEMORY_CACHE: dict[str, dict] = {}


def strip_yaml_value(value: str) -> str:
    return value.split("#", 1)[0].strip().strip("\"'")


def load_llm_config() -> dict:
    global _LLM_CONFIG_CACHE
    env_api_key = os.environ.get("VERUSEVAL_LLM_API_KEY", "")
    env_base_url = os.environ.get("VERUSEVAL_LLM_BASE_URL", "")
    env_model_name = os.environ.get("VERUSEVAL_LLM_MODEL_NAME", "")
    env_enable_thinking = os.environ.get("VERUSEVAL_LLM_ENABLE_THINKING", "")
    try:
        mtime = CONFIG_PATH.stat().st_mtime_ns
    except OSError:
        mtime = None

    cache_key = (mtime, env_api_key, env_base_url, env_model_name, env_enable_thinking)
    if _LLM_CONFIG_CACHE is not None and _LLM_CONFIG_CACHE[0] == cache_key:
        return dict(_LLM_CONFIG_CACHE[1])

    values: dict[str, str] = {}
    in_llm = False
    if mtime is not None:
        try:
            lines = CONFIG_PATH.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            return {"status": "not_available", "reason": f"config_read_failed:{exc}"}
    else:
        lines = []

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        if not in_llm:
            if stripped.split("#", 1)[0].strip() == "llm:":
                in_llm = True
            continue
        if indent == 0:
            break
        if ":" not in stripped:
            continue
        key, raw_value = stripped.split(":", 1)
        values[key.strip()] = strip_yaml_value(raw_value)

    api_key = env_api_key or values.get("api_key", "")
    base_url = (env_base_url or values.get("base_url", "")).rstrip("/")
    model_name = env_model_name or values.get("model_name", "")
    enable_thinking_text = env_enable_thinking or values.get("enable_thinking", "")
    enable_thinking = None
    if enable_thinking_text.lower() in {"1", "true", "yes", "on"}:
        enable_thinking = True
    elif enable_thinking_text.lower() in {"0", "false", "no", "off"}:
        enable_thinking = False
    missing = [
        name
        for name, value in (
            ("api_key", api_key),
            ("base_url", base_url),
            ("model_name", model_name),
        )
        if not value
    ]
    if missing:
        config = {
            "status": "not_available",
            "reason": "missing_llm_config:" + ",".join(missing),
            "has_api_key": bool(api_key),
            "base_url": base_url,
            "model_name": model_name,
        }
    else:
        config = {
            "status": "ok",
            "api_key": api_key,
            "base_url": base_url,
            "model_name": model_name,
            "enable_thinking": enable_thinking,
            "has_api_key": True,
        }
    _LLM_CONFIG_CACHE = (cache_key, dict(config))
    return config


def public_llm_metadata(config: dict, *, status: Optional[str] = None, reason: Optional[str] = None) -> dict:
    metadata = {
        "status": status or config.get("status"),
        "model_name": config.get("model_name"),
        "enable_thinking": config.get("enable_thinking"),
        "base_url_configured": bool(config.get("base_url")),
        "api_key_configured": bool(config.get("has_api_key")),
    }
    if reason:
        metadata["reason"] = reason
    elif config.get("reason"):
        metadata["reason"] = config.get("reason")
    return metadata


def llm_fingerprint() -> str:
    config = load_llm_config()
    payload = {
        "status": config.get("status"),
        "base_url": config.get("base_url"),
        "model_name": config.get("model_name"),
        "enable_thinking": config.get("enable_thinking"),
        "has_api_key": bool(config.get("has_api_key")),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:12]


def load_llm_disk_cache() -> dict:
    if _LLM_MEMORY_CACHE:
        return _LLM_MEMORY_CACHE
    try:
        if LLM_CACHE_PATH.exists():
            data = json.loads(LLM_CACHE_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                _LLM_MEMORY_CACHE.update(data)
    except (OSError, json.JSONDecodeError):
        pass
    return _LLM_MEMORY_CACHE


def save_llm_disk_cache() -> None:
    try:
        LLM_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        LLM_CACHE_PATH.write_text(
            json.dumps(_LLM_MEMORY_CACHE, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    except OSError:
        pass


def balanced_json_slice(text: str, open_ch: str, close_ch: str) -> Optional[str]:
    start = text.find(open_ch)
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        ch = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def extract_json_from_llm_text(text: str) -> Optional[Any]:
    cleaned = str(text or "").strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        cleaned = fence.group(1).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    openings = [
        (index, open_ch, close_ch)
        for open_ch, close_ch in (("{", "}"), ("[", "]"))
        if (index := cleaned.find(open_ch)) >= 0
    ]
    if openings:
        _, open_ch, close_ch = min(openings)
        candidate = balanced_json_slice(cleaned, open_ch, close_ch)
        if candidate:
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass
    return None


def llm_exception_summary(exc: BaseException) -> dict:
    return {
        "type": type(exc).__name__,
        "message": str(exc)[:300],
    }


def llm_content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content")
            else:
                text = getattr(item, "text", None) or getattr(item, "content", None)
            if text is not None:
                parts.append(str(text))
        if parts:
            return "\n".join(parts)
    return str(content)


def call_llm_chat_completion_with_openai_sdk(
    *,
    config: dict,
    messages: Sequence[dict],
    temperature: float,
    timeout_seconds: int,
    max_tokens: int,
    enable_thinking: Optional[bool],
) -> str:
    from openai import OpenAI  # type: ignore

    client = OpenAI(
        api_key=config["api_key"],
        base_url=config["base_url"],
        timeout=float(timeout_seconds),
    )
    extra_body = (
        {"enable_thinking": enable_thinking}
        if enable_thinking is not None
        else None
    )
    response = client.chat.completions.create(
        model=config["model_name"],
        messages=[dict(message) for message in messages],
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
        extra_body=extra_body,
        stream=False,
    )
    choices = getattr(response, "choices", None)
    if not choices:
        raise ValueError("openai_response_missing_choices")
    message = getattr(choices[0], "message", None)
    if message is None:
        raise ValueError("openai_response_missing_message")
    content = getattr(message, "content", None)
    if content is None and isinstance(message, dict):
        content = message.get("content")
    return llm_content_to_text(content)


def llm_metadata_with_transport(
    config: dict,
    *,
    status: str,
    client: str,
    sdk_error: Optional[dict] = None,
) -> dict:
    metadata = public_llm_metadata(config, status=status)
    metadata["client"] = client
    if sdk_error is not None:
        metadata["openai_sdk_error"] = sdk_error
    return metadata


def call_llm_json(
    *,
    task: str,
    messages: Sequence[dict],
    temperature: float = 0.0,
    timeout_seconds: int = DEFAULT_LLM_TIMEOUT_SECONDS,
    max_attempts: int = DEFAULT_LLM_MAX_ATTEMPTS,
    max_tokens: int = DEFAULT_LLM_MAX_TOKENS,
    use_disk_cache: bool = True,
) -> dict:
    config = load_llm_config()
    if config.get("status") != "ok":
        return {
            "status": "not_available",
            "reason": config.get("reason", "llm_config_unavailable"),
            "llm": public_llm_metadata(config),
        }

    cache_payload = {
        "task": task,
        "model": config.get("model_name"),
        "enable_thinking": config.get("enable_thinking"),
        "messages": list(messages),
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": "json_object",
        "parser_version": 7,
    }
    cache_key = hashlib.sha256(
        json.dumps(cache_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    cache = load_llm_disk_cache() if use_disk_cache else {}
    if cache_key in cache:
        cached = dict(cache[cache_key])
        cached["cached"] = True
        return cached

    request_payload = {
            "model": config["model_name"],
            "messages": list(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
            "stream": False,
        }
    if config.get("enable_thinking") is not None:
        request_payload["enable_thinking"] = config["enable_thinking"]
    request_body = json.dumps(
        request_payload,
        ensure_ascii=False,
    ).encode("utf-8")

    def make_request() -> urllib.request.Request:
        return urllib.request.Request(
            f"{config['base_url']}/chat/completions",
            data=request_body,
            method="POST",
            headers={
                "Authorization": f"Bearer {config['api_key']}",
                "Content-Type": "application/json",
            },
        )

    attempts = max(1, int(max_attempts or 1))
    last_error: Optional[dict] = None
    for attempt in range(1, attempts + 1):
        sdk_error: Optional[dict] = None
        transport_client = "openai_sdk"
        try:
            try:
                content = call_llm_chat_completion_with_openai_sdk(
                    config=config,
                    messages=messages,
                    temperature=temperature,
                    timeout_seconds=timeout_seconds,
                    max_tokens=max_tokens,
                    enable_thinking=config.get("enable_thinking"),
                )
            except Exception as exc:
                sdk_error = llm_exception_summary(exc)
                transport_client = "urllib"
                with urllib.request.urlopen(make_request(), timeout=timeout_seconds) as response:
                    response_text = response.read().decode("utf-8", errors="replace")
                try:
                    payload = json.loads(response_text)
                    content = payload["choices"][0]["message"]["content"]
                except (KeyError, IndexError, TypeError, json.JSONDecodeError) as parse_exc:
                    last_error = {
                        "status": "not_available",
                        "reason": f"llm_response_parse_failed:{type(parse_exc).__name__}",
                        "raw": response_text[:1200],
                        "llm": llm_metadata_with_transport(
                            config,
                            status="error",
                            client=transport_client,
                            sdk_error=sdk_error,
                        ),
                        "attempts": attempt,
                    }
                    if attempt < attempts:
                        time.sleep(min(0.5 * attempt, 2.0))
                        continue
                    return last_error

            parsed = extract_json_from_llm_text(content)
            if parsed is None:
                last_error = {
                    "status": "not_available",
                    "reason": "llm_json_parse_failed",
                    "raw": str(content)[:1200],
                    "llm": llm_metadata_with_transport(
                        config,
                        status="error",
                        client=transport_client,
                        sdk_error=sdk_error,
                    ),
                    "attempts": attempt,
                }
                if attempt < attempts:
                    time.sleep(min(0.5 * attempt, 2.0))
                    continue
                return last_error

            result = {
                "status": "ok",
                "json": parsed,
                "raw": str(content)[:1200],
                "llm": llm_metadata_with_transport(
                    config,
                    status="ok",
                    client=transport_client,
                    sdk_error=sdk_error,
                ),
                "cached": False,
                "attempts": attempt,
            }
            if use_disk_cache:
                cache[cache_key] = result
                save_llm_disk_cache()
            return result
        except urllib.error.HTTPError as exc:
            try:
                body = exc.read().decode("utf-8", errors="replace")[:800]
            except Exception:
                body = ""
            last_error = {
                "status": "not_available",
                "reason": f"llm_http_error:{exc.code}",
                "http_body": body,
                "llm": llm_metadata_with_transport(
                    config,
                    status="error",
                    client=transport_client,
                    sdk_error=sdk_error,
                ),
                "attempts": attempt,
            }
            break
        except (
            urllib.error.URLError,
            TimeoutError,
            OSError,
            http.client.HTTPException,
            http.client.IncompleteRead,
            http.client.RemoteDisconnected,
        ) as exc:
            last_error = {
                "status": "not_available",
                "reason": f"llm_call_failed:{type(exc).__name__}",
                "error": str(exc)[:300],
                "llm": llm_metadata_with_transport(
                    config,
                    status="error",
                    client=transport_client,
                    sdk_error=sdk_error,
                ),
                "attempts": attempt,
            }
            if attempt < attempts:
                time.sleep(min(0.5 * attempt, 2.0))
    return last_error or {
        "status": "not_available",
        "reason": "llm_call_failed:unknown",
        "llm": public_llm_metadata(config, status="error"),
        "attempts": attempts,
    }


__all__ = [
    "DEFAULT_LLM_MAX_ATTEMPTS",
    "DEFAULT_LLM_MAX_TOKENS",
    "DEFAULT_LLM_TIMEOUT_SECONDS",
    "call_llm_json",
    "extract_json_from_llm_text",
    "llm_fingerprint",
    "load_llm_config",
    "public_llm_metadata",
]
