from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from utils.client_util import call_llm
from utils.file_util import load_content
from utils.inference_util import (
    build_messages,
    build_messages_with_examples,
    extract_code_block,
    parse_spec_check_response,
)
from utils.repair_util import (
    VerusErrorType,
    group_errors_by_type,
    parse_verus_errors,
    sort_grouped_errors_by_priority,
)
from utils.verus_util import run_code

from .schemas import PlannerDecision, RepairBlueprint, VerificationResult


WORKFLOW_DIR = Path(__file__).resolve().parents[1]
PROMPT_DIR = WORKFLOW_DIR / "prompt" / "pipeline"
REPAIR_PROMPT_DIR = WORKFLOW_DIR / "prompt" / "repair"


def _prompt(name: str) -> str:
    return (PROMPT_DIR / name).read_text(encoding="utf-8")


def _messages(user_content: str) -> List[Dict[str, str]]:
    system_prompt = (WORKFLOW_DIR / "prompt" / "system_prompt.txt").read_text(
        encoding="utf-8"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


def format_errors(errors_by_type: Dict[str, List[Dict[str, Any]]]) -> str:
    sections = []
    for error_type, errors in errors_by_type.items():
        sections.append(f"[{error_type}]")
        for error in errors:
            line = error.get("line", "?")
            sections.append(f"Line {line}: {error.get('message', '')}")
    return "\n".join(sections) if sections else "No parsed Verus errors."


class InitialGenerator:
    def __init__(self, model_name: str, temperature: float):
        self.model_name = model_name
        self.temperature = temperature

    def generate(
        self,
        source_path: Path,
        num_candidates: int,
        examples: Optional[Sequence[Sequence[str]]] = None,
    ) -> List[str]:
        code = load_content(str(source_path))
        if examples is None:
            messages = build_messages(code)
        else:
            messages = build_messages_with_examples(code, examples)
        responses = call_llm(
            messages,
            self.model_name,
            num_candidates,
            self.temperature,
        )
        return [extract_code_block(response) for response in responses]


class Verifier:
    def verify(self, path: Path) -> VerificationResult:
        stdout, stderr = run_code(str(path))
        verified_matches = re.findall(r"(\d+)\s+verified", stdout, re.IGNORECASE)
        error_matches = re.findall(r"(\d+)\s+errors?", stdout, re.IGNORECASE)
        verified = sum(map(int, verified_matches)) if verified_matches else -1
        errors = sum(map(int, error_matches)) if error_matches else -1

        parsed_errors = parse_verus_errors(stderr)
        errors_by_type = sort_grouped_errors_by_priority(
            group_errors_by_type(parsed_errors)
        )
        if not errors_by_type and not (verified > 0 and errors == 0):
            errors_by_type = {
                "Other": [
                    {
                        "error_type": "Other",
                        "message": stderr.strip() or stdout.strip() or "Unknown Verus failure",
                        "line": 0,
                        "column": 0,
                        "code": "",
                    }
                ]
            }
        return VerificationResult(
            path=path,
            verified_obligations=verified,
            errors=errors,
            stdout=stdout,
            stderr=stderr,
            errors_by_type=errors_by_type,
        )


class BestCandidateSelector:
    def select(self, candidates: Iterable[VerificationResult]) -> VerificationResult:
        candidate_list = list(candidates)
        if not candidate_list:
            raise ValueError("Cannot select from an empty candidate set")
        return min(candidate_list, key=lambda item: item.score_key)


class Judge:
    def __init__(self, model_name: str, temperature: float):
        self.model_name = model_name
        self.temperature = temperature

    def evaluate(self, code: str) -> Tuple[bool, str]:
        user_prompt = _prompt("judge.txt").replace("$CODE", code)
        response = call_llm(
            _messages(user_prompt), self.model_name, 1, self.temperature
        )[0]
        return parse_spec_check_response(response)


class Aligner:
    def __init__(self, model_name: str, temperature: float):
        self.model_name = model_name
        self.temperature = temperature

    def align(self, code: str, diagnosis: str) -> str:
        user_prompt = (
            _prompt("aligner.txt")
            .replace("$DIAGNOSIS", diagnosis)
            .replace("$CODE", code)
        )
        response = call_llm(
            _messages(user_prompt), self.model_name, 1, self.temperature
        )[0]
        return extract_code_block(response)


class Planner:
    def __init__(self, model_name: str, temperature: float):
        self.model_name = model_name
        self.temperature = temperature
        self.error_types = list(VerusErrorType.__members__)
        self.repairers = {name: f"{name}Repairer" for name in self.error_types}

    @staticmethod
    def _extract_json(text: str) -> Dict[str, Any]:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1:
            raise ValueError("Planner response does not contain a JSON object")
        value = json.loads(cleaned[start : end + 1])
        if not isinstance(value, dict):
            raise ValueError("Planner response must be a JSON object")
        return value

    def _fallback(
        self,
        verification: VerificationResult,
        reason: str,
    ) -> PlannerDecision:
        error_type = next(iter(verification.errors_by_type), "Other")
        if error_type not in self.repairers:
            error_type = "Other"
        return PlannerDecision(
            root_cause_error_type=error_type,
            selected_repairer=self.repairers[error_type],
            fix_order=[error_type],
            summary=f"Planner fallback: {reason}",
            root_cause="planner-fallback",
            evidence=format_errors(verification.errors_by_type),
            repair_direction="Apply the specialized error guidance conservatively.",
            risk="The Planner output could not be validated.",
            used_fallback=True,
        )

    def plan(self, code: str, verification: VerificationResult) -> PlannerDecision:
        registry = "\n".join(
            f"- {error_type}: {repairer}"
            for error_type, repairer in self.repairers.items()
        )
        score = (
            f"verified={verification.verified_obligations}, "
            f"errors={verification.errors}"
        )
        prompt = (
            _prompt("planner.txt")
            .replace("$REPAIRER_REGISTRY", registry)
            .replace("$SCORE", score)
            .replace("$ERRORS", format_errors(verification.errors_by_type))
            .replace("$CODE", code)
        )
        try:
            response = call_llm(
                _messages(prompt), self.model_name, 1, self.temperature
            )[0]
            value = self._extract_json(response)
            error_type = str(value["root_cause_error_type"]).strip()
            repairer = str(value["selected_repairer"]).strip()
            if error_type not in self.repairers:
                raise ValueError(f"Unknown root cause error type: {error_type}")
            if repairer != self.repairers[error_type]:
                raise ValueError(
                    f"Repairer {repairer} does not match error type {error_type}"
                )
            fix_order = value.get("fix_order", [error_type])
            if not isinstance(fix_order, list) or not fix_order:
                fix_order = [error_type]
            fix_order = [
                item for item in map(str, fix_order) if item in self.repairers
            ] or [error_type]
            return PlannerDecision(
                root_cause_error_type=error_type,
                selected_repairer=repairer,
                fix_order=fix_order,
                summary=str(value.get("summary", "")),
                root_cause=str(value.get("root_cause", "")),
                evidence=str(value.get("evidence", "")),
                repair_direction=str(value.get("repair_direction", "")),
                risk=str(value.get("risk", "")),
            )
        except Exception as exc:
            return self._fallback(verification, str(exc))


class Repairer:
    def __init__(self, model_name: str, temperature: float):
        self.model_name = model_name
        self.temperature = temperature

    @staticmethod
    def _knowledge(error_type: str) -> str:
        path = REPAIR_PROMPT_DIR / error_type / "repair_prompt.txt"
        if not path.is_file():
            path = REPAIR_PROMPT_DIR / "Other" / "repair_prompt.txt"
        return path.read_text(encoding="utf-8")

    def create_blueprint(
        self,
        code: str,
        verification: VerificationResult,
        decision: PlannerDecision,
        history: str,
    ) -> RepairBlueprint:
        decision_json = json.dumps(decision.__dict__, ensure_ascii=False, indent=2)
        prompt = (
            _prompt("repairer.txt")
            .replace("$REPAIRER_NAME", decision.selected_repairer)
            .replace("$PLANNER_DECISION", decision_json)
            .replace("$KNOWLEDGE", self._knowledge(decision.root_cause_error_type))
            .replace("$ERRORS", format_errors(verification.errors_by_type))
            .replace("$HISTORY", history)
            .replace("$CODE", code)
        )
        response = call_llm(
            _messages(prompt), self.model_name, 1, self.temperature
        )[0]
        return RepairBlueprint(decision.selected_repairer, response.strip())


class Actor:
    def __init__(self, model_name: str, temperature: float):
        self.model_name = model_name
        self.temperature = temperature

    def act(
        self,
        code: str,
        blueprint: RepairBlueprint,
        num_candidates: int,
    ) -> List[str]:
        prompt = (
            _prompt("actor.txt")
            .replace("$REPAIRER_NAME", blueprint.repairer_name)
            .replace("$BLUEPRINT", blueprint.content)
            .replace("$CODE", code)
        )
        responses = call_llm(
            _messages(prompt),
            self.model_name,
            num_candidates,
            self.temperature,
        )
        return [extract_code_block(response) for response in responses]


class Rewriter:
    def __init__(self, model_name: str, temperature: float):
        self.model_name = model_name
        self.temperature = temperature

    def rewrite(self, code: str, verification: VerificationResult) -> str:
        prompt = (
            _prompt("rewriter.txt")
            .replace("$ERRORS", format_errors(verification.errors_by_type))
            .replace("$CODE", code)
        )
        response = call_llm(
            _messages(prompt), self.model_name, 1, self.temperature
        )[0]
        return extract_code_block(response)
