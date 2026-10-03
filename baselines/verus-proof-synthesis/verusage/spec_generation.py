"""Spec-generating front end shared with the VeruSAGE repair pipeline."""

import re

from global_config import GlobalConfig


SPEC_SYSTEM_PROMPT = (
    "You are an expert in Verus, a verification-aware programming language. "
    "Write formally verified Rust code with meaningful specifications. Preserve "
    "the executable behavior and never use assume, admit, external, or external_body."
)

SPEC_USER_TEMPLATE = """Consider the following Verus code which lacks formal specifications:
```rust
{program}
```

Add meaningful preconditions (`requires`), postconditions (`ensures`), specification
functions when needed, loop invariants, decreases clauses, assertions, and proof blocks so
that Verus can verify the program. Preserve the original executable statements and function
behavior. Output the complete Verus program and no explanation. Do not use `assume`,
`admit`, `#[verifier::external]`, or `#[verifier::external_body]`.
"""


def extract_rust_code(text: str) -> str:
    matches = re.findall(r"```(?:rust|verus)?\s*\n?(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if matches:
        return max(matches, key=len).strip()
    return text.strip()


def generate_spec_program(code: str, exemplars: list[dict], temp: float) -> str:
    """Generate a complete specified program before VeruSAGE repair starts."""
    config = GlobalConfig.get_config()
    llm = GlobalConfig.get_llm()
    examples = [
        {
            "query": SPEC_USER_TEMPLATE.format(program=item["input"]),
            "answer": item["output"],
        }
        for item in exemplars
    ]
    responses = llm.infer_llm(
        config.aoai_generation_model,
        None,
        examples,
        SPEC_USER_TEMPLATE.format(program=code),
        SPEC_SYSTEM_PROMPT,
        answer_num=1,
        max_tokens=config.max_token,
        temp=temp,
    )
    if not responses:
        raise RuntimeError("spec generation returned no candidate")
    generated = extract_rust_code(responses[0])
    if not generated:
        raise RuntimeError("spec generation returned an empty program")
    compact = re.sub(r"\s+", "", generated).lower()
    forbidden = ("assume(", "admit(", "#[verifier::external]", "#[verifier::external_body]")
    if any(token in compact for token in forbidden):
        raise RuntimeError("spec generation returned a forbidden verification bypass")
    return generated
