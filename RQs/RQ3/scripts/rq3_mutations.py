"""Eight contract edits; candidate construction never inspects I/O outcomes."""
from __future__ import annotations

import importlib.util
import random
import re
import sys
from functools import lru_cache
from pathlib import Path

from rq3_io_common import ROOT, contract_mask, digest
from metrics_rebuild.share.text import strip_comments, token_spans
from metrics_rebuild.share.functions import function_parameters_from_header, strength_contexts_for_path

OPERATORS = {
    "PRE_ADD": "pre_strengthening",
    "PRE_REFINE": "pre_strengthening",
    "PRE_REMOVE": "pre_weakening",
    "PRE_RELAX": "pre_weakening",
    "POST_ADD": "post_strengthening",
    "POST_REFINE": "post_strengthening",
    "POST_REMOVE": "post_weakening",
    "POST_RELAX": "post_weakening",
}


def parser_module():
    from RQs.shared import contract_edits
    return contract_edits


def relation_edits(text, strengthen):
    """Conservative positive positions. Final whole-predicate proof is mandatory."""
    if any(s in text for s in ("==>", "<==>", "forall", "exists", "match", "if ", "&&", "||")):
        return
    tokens = token_spans(text)
    if any(t.text in ("!", "!=") for t in tokens):
        return
    replacements = {"<=": ("<", "=="), ">=": (">", "==")} if strengthen else {
        "<": ("<=",), ">": (">=",), "==": ("<=", ">=")}
    for token in tokens:
        for replacement in replacements.get(token.text, ()):
            yield text[:token.start] + replacement + text[token.end:]
    if strengthen:
        # Strict integer boundaries, only literal RHS with no following arithmetic.
        for match in re.finditer(r"(?<![<>=!])([<>])\s*(-?\d+)(?![\w.])\s*$", text.strip()):
            value = int(match.group(2)) + (1 if match.group(1) == ">" else -1)
            yield text[:match.start(2)] + str(value) + text[match.end(2):]


def post_proposals(path, target, historical):
    module = parser_module()
    clean = strip_comments(path.read_text())
    layout = module.locate_target(clean, target)
    ctx = next(c for c in strength_contexts_for_path(str(path)) if c["function"] == target)
    candidates = [(r["replacement_text"], "historical_proposal") for r in historical
                  if r["operator"] == "POST_ADD_UNPROMISED" and r.get("replacement_text")]
    body = clean[layout.body_open + 1:layout.body_close].strip()
    params = function_parameters_from_header(clean[layout.fn_start:layout.body_open])
    integer_params = [p["name"] for p in params if re.fullmatch(r"(?:u|i)(?:8|16|32|64|128)|usize|isize|int|nat", p["type"].strip())]
    for ret in ctx.get("returns", []):
        name, typ = ret["name"], ret["type"].strip()
        if re.fullmatch(r"(?:u|i)(?:8|16|32|64|128)|usize|isize|int|nat", typ):
            if re.fullmatch(r"-?\d+(?:[iu](?:8|16|32|64|128))?", body):
                candidates.append((f"{name} == {body}", "constant_return"))
            candidates.extend((f"{name} {op} {rhs}", "typed_template")
                              for rhs in [*integer_params, "0", "1"] for op in ("==", ">=", "<="))
        elif typ == "bool":
            candidates.extend([(name, "typed_template"), (f"!{name}", "typed_template")])
        elif "Vec<" in typ or "Seq<" in typ:
            candidates.extend([(f"{name}.len() > 0", "typed_template"), (f"{name}.len() == 0", "typed_template")])
    for param in params:
        if "&mut" in param["type"] and "Vec<" in param["type"]:
            name = param["name"]
            candidates.append((f"{name}.len() == old({name}).len()", "typed_template"))
    return candidates


def generate(path, target, sample_id, historical=(), seed=20261001, budget=3):
    module = parser_module()
    clean = strip_comments(path.read_text())
    layout = module.locate_target(clean, target)
    if layout is None:
        raise ValueError("target_layout_missing")
    rng = random.Random(str(seed) + sample_id)
    result = {}
    for operator in OPERATORS:
        kind = "requires" if operator.startswith("PRE_") else "ensures"
        action = operator.split("_", 1)[1]
        parts = layout.parts_of(kind)
        edits = []
        if action == "ADD":
            proposals = ([(c, "typed_template") for _, c in module.spurious_precondition_candidates(clean[layout.fn_start:layout.body_open])]
                         if kind == "requires" else post_proposals(path, target, historical))
            existing = {" ".join(t.text for t in token_spans(g.parts[i][0])) for g, i in parts}
            for clause, source in proposals:
                if " ".join(t.text for t in token_spans(clause)) not in existing:
                    edits.append((module.insert_clause(clean, layout, kind, clause), "", clause, source))
        elif action == "REMOVE" and len(parts) >= 2:
            for group, index in parts:
                start, end = module.removal_span_for_part(group, index, clean)
                text = module.splice(clean, [(start, end, "")])
                try:
                    same_body = contract_mask(clean, target)[0] == contract_mask(text, target)[0]
                except ValueError:
                    same_body = False
                if not same_body:
                    # A trailing quantified clause needs a separator before the body.
                    text = module.splice(clean, [(start, end, "\n")])
                edits.append((text, group.parts[index][0], "", "clause_removal"))
            rng.shuffle(edits)
        elif action in ("REFINE", "RELAX"):
            for group, index in parts:
                before, start, end = group.parts[index]
                for after in relation_edits(before, action == "REFINE"):
                    edits.append((module.splice(clean, [(start, end, after)]), before, after, "relation_replacement"))
            rng.shuffle(edits)
        unique, seen = [], set()
        for text, before, after, source in edits:
            key = digest(text)
            if key in seen:
                continue
            seen.add(key)
            unique.append({"operator": operator, "direction": OPERATORS[operator], "text": text,
                           "changed_kind": kind, "changed_text": before, "replacement_text": after,
                           "proposal_source": source, "candidate_rank": len(unique) + 1})
        result[operator] = {"available_candidates": len(unique), "candidates": unique[:budget]}
    return clean, result
