import os
import sys
from typing import Dict, List

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.file_util import load_content


def extract_code_block(text: str) -> str:
    start = text.find("```")
    if start == -1:
        code = text.strip()
    else:
        after = text[start + 3 :]
        newline = after.find("\n")
        after = after[newline + 1 :] if newline != -1 else ""
        end = after.find("```")
        code = (after[:end] if end != -1 else after).strip()

    if "main()" not in code:
        code += "\n\nfn main() {}"
    return code


def build_messages(x_code: str) -> List[Dict[str, str]]:
    system_prompt = load_content("./prompt/system_prompt.txt")
    user_prompt = load_content("./prompt/user_prompt.txt") + f"{x_code}\n"
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def build_messages_with_examples(
    x_code: str, examples: List[List[str]]
) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = []
    system_prompt = load_content("./prompt/system_prompt.txt")
    user_prompt_template = load_content("./prompt/user_prompt.txt")
    messages.append({"role": "system", "content": system_prompt})

    for example in examples:
        messages.extend(
            [
                {"role": "user", "content": user_prompt_template + example[0]},
                {"role": "assistant", "content": example[1]},
            ]
        )

    messages.append(
        {"role": "user", "content": user_prompt_template + x_code}
    )
    return messages


def parse_spec_check_response(response_text: str) -> tuple[bool, str]:
    if not response_text:
        return False, "Error: Empty response from LLM."

    import re

    clean_text = response_text.replace("```xml", "").replace("```", "").strip()
    result_match = re.search(
        r"<result>(.*?)</result>", clean_text, re.IGNORECASE | re.DOTALL
    )
    diagnosis_match = re.search(
        r"<DIAGNOSIS>(.*?)</DIAGNOSIS>",
        clean_text,
        re.IGNORECASE | re.DOTALL,
    )

    if not result_match:
        lower_text = clean_text.lower()
        if "true" in lower_text[:20]:
            return True, ""
        if "false" in lower_text[:20]:
            return False, clean_text
        return False, f"Parse Error: Missing <result> tags. Raw output:\n{response_text}"

    verdict_text = result_match.group(1).strip().lower()
    if "true" in verdict_text:
        return True, ""
    if "false" not in verdict_text:
        return False, f"Parse Error: Invalid content inside <result>: '{verdict_text}'"

    diagnosis = diagnosis_match.group(1).strip() if diagnosis_match else ""
    if not diagnosis or diagnosis.upper() == "N/A":
        diagnosis = f"Verdict is False but diagnosis is empty/N/A. Raw: {clean_text}"
    return False, diagnosis
