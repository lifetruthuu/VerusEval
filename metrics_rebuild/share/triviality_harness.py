"""Build isolated hosts for semantic-triviality probes.

Only trigger metadata and irrelevant declarations are removed. Keep original
logical expressions, signatures, and reachable spec definitions; proof helpers
must never be promoted to axioms.
"""
from __future__ import annotations

import re

from . import functions
from .clauses import FN_RE, extract_clauses_from_text, find_matching_brace, find_matching_paren
from .text import strip_comments, token_spans


def erase_trigger_annotations(text: str) -> str:
    spans = token_spans(text)
    edits = []
    for i, token in enumerate(spans):
        if token.text != "#":
            continue
        j = i + 1
        if j < len(spans) and spans[j].text == "!":
            j += 1
        if j + 1 >= len(spans) or spans[j].text != "[":
            continue
        name = j + 1
        if name + 2 < len(spans) and spans[name].text == "verifier" and spans[name + 1].text == "::":
            name += 2
        if spans[name].text not in {"trigger", "auto", "all_triggers"}:
            continue
        depth = 0
        for end in range(j, len(spans)):
            depth += spans[end].text == "["
            depth -= spans[end].text == "]"
            if depth == 0:
                edits.append((token.start, spans[end].end))
                break
    for start, end in reversed(edits):
        text = text[:start] + " " + text[end:]
    return text


def explicit_triggers(text: str, *, extended: bool = False) -> tuple[str, list[dict]]:
    """Use existing array-access or remainder terms covering bound variables.

    Never invent a predicate/identity wrapper to manufacture a trigger. Nested
    bound variables and local let bindings cannot occur in an outer trigger.
    Unsupported cases remain undecided and are recorded by Verus as usual.
    """
    pattern = re.compile(r"\b(?:forall|exists|choose)\s*\|([^|]+)\|")
    edits = []
    audit = []
    for quantifier in pattern.finditer(text):
        variables = set()
        for declaration in quantifier.group(1).split(","):
            name = declaration.split(":", 1)[0].strip()
            if not re.fullmatch(r"[A-Za-z_]\w*", name):
                break
            variables.add(name)
        else:
            body = text[quantifier.end():]
            depths = {"(": 0, "[": 0, "{": 0}
            closes = {")": "(", "]": "[", "}": "{"}
            for token in token_spans(body):
                if token.text in closes:
                    opener = closes[token.text]
                    if depths[opener] == 0:
                        body = body[:token.start]
                        break
                    depths[opener] -= 1
                elif token.text in depths:
                    depths[token.text] += 1
                elif token.text == "," and not any(depths.values()):
                    body = body[:token.start]
                    break
            if extended:
                # A nested binder that shadows an outer variable ends the
                # usable prefix; terms before it still refer to the outer one.
                for nested in pattern.finditer(body):
                    names = {d.split(":", 1)[0].strip() for d in nested.group(1).split(",")}
                    if names & variables:
                        body = body[:nested.start()]
                        break
            local = set(re.findall(r"\blet\s+(?:mut\s+)?([A-Za-z_]\w*)", body))
            for nested in pattern.finditer(body):
                local.update(d.split(":", 1)[0].strip() for d in nested.group(1).split(","))
            candidates = []
            terms = [m.group() for m in re.finditer(r"\b[A-Za-z_]\w*@?\[[^\[\]\n]+\]|\b[A-Za-z_]\w*\s*%\s*[A-Za-z_]\w*\b", body)]
            if extended:
                # Include existing calls (including method calls) and simple
                # arithmetic subterms. These annotations do not add predicates.
                calls = re.finditer(r"\b[A-Za-z_]\w*(?:(?:::|@?\s*\.)\s*[A-Za-z_]\w*)*\s*\(", body)
                for call in calls:
                    called_name = re.findall(r"[A-Za-z_]\w*", call.group())[-1]
                    if called_name[0].isupper():
                        continue  # Enum constructors are not legal Verus triggers.
                    end = find_matching_paren(body, call.end() - 1)
                    if end is not None:
                        term = body[call.start():end + 1]
                        if not re.search(r"\b(?:let|forall|exists|lambda|choose|if|match)\b", term):
                            terms.append(term)
                for term in re.finditer(r"\b[A-Za-z_]\w*\s*[-+*/%]\s*(?:[A-Za-z_]\w*|[0-9]+)\b(?!\w|\s*[@.\[(])", body):
                    terms.append(term.group())
                for term in re.finditer(r"\b[A-Za-z_]\w*\s*[-+*/%]\s*[A-Za-z_]\w*@?\s*\.\s*len\s*\(\s*\)(?:\s+as\s+(?:int|nat))?", body):
                    terms.append(term.group())
                arithmetic = (
                    r"\b[A-Za-z_]\w*\s*[-+*/%]\s*[A-Za-z_]\w*@?\[[^\[\]\n]+\]",
                    r"\b[A-Za-z_]\w*@?\s*\.\s*len\s*\(\s*\)\s*%\s*[A-Za-z_]\w*\b",
                    r"\b[A-Za-z_]\w*\s*%\s*\(\s*[A-Za-z_]\w*\s+as\s+[A-Za-z_]\w*\s*\)",
                )
                for expression in arithmetic:
                    terms.extend(m.group() for m in re.finditer(expression, body))
            for term in terms:
                identifiers = set(re.findall(r"\b[A-Za-z_]\w*\b", term))
                covered = identifiers & variables
                if covered and not identifiers & local:
                    candidates.append((term, covered))
            remaining = set(variables)
            selected = []
            while remaining:
                viable = [(term, covered) for term, covered in candidates if covered <= remaining]
                if not viable:
                    break
                term, covered = max(viable, key=lambda pair: (len(pair[1]), -len(pair[0])))
                selected.append(term)
                remaining -= covered
            if variables and not remaining:
                edits.append((quantifier.end(), " #![trigger " + ", ".join(selected) + "] "))
                audit.append({"variables": sorted(variables), "terms": selected})
    for offset, insertion in reversed(edits):
        text = text[:offset] + insertion + text[offset:]
    return text, audit


def attribute_start(text: str, start: int) -> int:
    """Include function attributes when removing/replacing its declaration."""
    tokens = token_spans(text[:start])
    i = len(tokens) - 1
    while i >= 0 and tokens[i].text == "]":
        depth = 1
        j = i - 1
        while j >= 0 and depth:
            depth += tokens[j].text == "]"
            depth -= tokens[j].text == "["
            j -= 1
        if j >= 0 and tokens[j].text == "!":
            j -= 1
        if j < 0 or tokens[j].text != "#":
            break
        start = tokens[j].start
        i = j - 1
    return start


def isolated_host(text: str, target: str, part: str, clause_index: int | None = None, *, preserve_triggers: bool = False) -> tuple[str, dict]:
    clean = strip_comments(text)
    blocks = []
    for match in FN_RE.finditer(clean):
        found = functions._find_top_level_signature_end(clean, match.end())
        if found is None or not found[1]:
            continue
        body_open = found[0]
        body_close = find_matching_brace(clean, body_open)
        assert body_close is not None
        mode = functions.function_mode(match.group("prefix") or "")
        if functions._leading_verifier_spec_attribute(clean, match.start()) is not None:
            mode = "spec"
        blocks.append({"name": match.group("name"), "mode": mode,
                       "start": attribute_start(clean, match.start()), "end": body_close + 1,
                       "header": clean[match.start():body_open]})
    chosen = [b for b in blocks if b["name"] == target and b["mode"] == "exec"]
    assert len(chosen) == 1, (target, chosen)
    kind = "requires" if part == "precondition_falsity" else "ensures"
    wanted = {kind} if kind == "requires" else {"ensures", "default_ensures"}
    clauses = [c.text for c in extract_clauses_from_text(chosen[0]["header"]) if c.kind in wanted]
    original_clauses = list(clauses)
    if clause_index is not None:
        assert part == "postcondition_truth"
        clauses = [clauses[clause_index]]
    target_header = functions.function_declaration_prefix(chosen[0]["header"])
    if clauses:
        target_header += "\n    " + kind + "\n        " + ",\n        ".join(clauses) + ",\n"

    # Retain every type/import/non-function item, and spec definitions reachable
    # from them or the selected declaration. Proof helpers are not made axioms.
    outside = clean
    for block in reversed(blocks):
        outside = outside[:block["start"]] + outside[block["end"]:]
    roots = outside + "\n" + target_header
    kept = set()
    while True:
        additions = [b for b in blocks if b["mode"] == "spec" and b["name"] not in kept
                     and re.search(r"\b" + re.escape(b["name"]) + r"\b", roots)]
        if not additions:
            break
        for block in additions:
            kept.add(block["name"])
            roots += "\n" + clean[block["start"]:block["end"]]
    out = clean
    for block in reversed(blocks):
        if block is chosen[0]:
            replacement = "#[verifier::external_body]\n" + target_header + "\n{ unimplemented!() }\n"
        elif block["mode"] == "spec" and block["name"] in kept:
            replacement = clean[block["start"]:block["end"]]
        else:
            replacement = "\n"
        out = out[:block["start"]] + replacement + out[block["end"]:]
    return (out if preserve_triggers else erase_trigger_annotations(out)), {
        "target": target, "part": part, "clauses": clauses,
        "original_clauses": original_clauses, "clause_index": clause_index,
        "kept_spec_functions": sorted(kept),
        "removed_functions": [b["name"] for b in blocks if b is not chosen[0] and b["name"] not in kept],
    }
