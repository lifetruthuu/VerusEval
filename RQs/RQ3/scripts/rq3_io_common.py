"""Data and concrete witness helpers for the RQ3 paired I/O experiment."""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from metrics_rebuild.share import contract_eval as ce
from metrics_rebuild.share import io_cases as io
from metrics_rebuild.share.clauses import CLAUSE_BOUNDARY_KINDS, _token_depth_update, _top_level
from metrics_rebuild.share.functions import (
    FN_RE, _find_top_level_signature_end, strength_contexts_for_path,
)
from metrics_rebuild.share.text import strip_comments, token_spans

OUTPUT = ROOT / "data/evidence/contract_variants"
RESULTS = ROOT / "RQs/RQ3/results"
SEED = 20261001
CATEGORIES = ("positive", "negative", "invalid")
METRICS = dict(zip(CATEGORIES, (
    "correct_io_pass_rate", "wrong_io_reject_rate", "invalid_test_filtering_rate",
)))


def digest(value) -> str:
    text = value if isinstance(value, str) else json.dumps(value, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(text.encode()).hexdigest()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text())


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n")
    temporary.replace(path)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict], fields=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def stem(mutant_id: str) -> str:
    return mutant_id.replace("::", "__").replace("/", "_")


def context(path: Path, target: str, observable=False) -> dict:
    ctx = next(item for item in strength_contexts_for_path(str(path)) if item["function"] == target)
    if observable:
        descriptor = io._target_descriptor_for_generated(str(path), target, ctx)
        ctx = io._observable_io_context(ctx, descriptor)
    return ctx


def contract_mask(text: str, target: str) -> tuple[str, dict[str, list[str]]]:
    """Remove only requires/ensures of the named exec function, preserving all other tokens."""
    clean = strip_comments(text)
    matches = [m for m in FN_RE.finditer(clean) if m.group("name") == target]
    if len(matches) != 1:
        raise ValueError("ambiguous_or_missing_target")
    match = matches[0]
    end = _find_top_level_signature_end(clean, match.end())
    if end is None or not end[1]:
        raise ValueError("target_body_missing")
    header_end = end[0]
    spans = token_spans(clean[match.start():header_end])
    depths = {"paren": 0, "bracket": 0, "brace": 0}
    starts = []
    for index, span in enumerate(spans):
        if (_top_level(depths) and span.text in CLAUSE_BOUNDARY_KINDS
                and not (index and spans[index - 1].text in {".", "::"})):
            starts.append((span.text, match.start() + span.start))
        _token_depth_update(span.text, depths)
    ranges, groups = [], {"requires": [], "ensures": []}
    for index, (kind, start) in enumerate(starts):
        stop = starts[index + 1][1] if index + 1 < len(starts) else header_end
        if kind in groups:
            groups[kind].append(" ".join(t.text for t in token_spans(clean[start:stop])))
            ranges.append((start, stop))
    for start, stop in reversed(ranges):
        clean = clean[:start] + " " + clean[stop:]
    return " ".join(t.text for t in token_spans(clean)), groups


def edit_valid(base: Path, mutant: Path, target: str, direction: str) -> bool:
    a, ac = contract_mask(base.read_text(), target)
    b, bc = contract_mask(mutant.read_text(), target)
    untouched = "ensures" if direction.startswith("pre_") else "requires"
    return a == b and ac[untouched] == bc[untouched]


def mapped_cases(ctx: dict, suite: dict) -> list[dict]:
    """Express frozen cases in the base/variant's argument and return names."""
    target = suite.get("target") or {}
    cases = []
    for item in suite.get("cases", []):
        inputs = io._remap_payload_by_position(
            item.get("inputs", {}), target.get("parameters", []), ctx.get("parameters", []))
        output = None
        if item.get("kind") in {"positive", "negative"}:
            output = io._remap_payload_by_position(
                item.get("mutated_output", {}) if item["kind"] == "negative" else item.get("output", {}),
                io._observable_output_fields(target), ctx.get("returns", []))
        if inputs is not None:
            cases.append({"inputs": inputs, "output": output, "source": "frozen:" + item["id"]})
    return cases


def value_variants(value, constants: list[int], depth=0):
    if isinstance(value, bool):
        yield not value
    elif isinstance(value, int):
        yield from dict.fromkeys([value - 1, value + 1, 0, 1, -1, *constants])
    elif isinstance(value, float):
        yield from (0.0, 1.0, -1.0, value - 1.0, value + 1.0)
    elif isinstance(value, str):
        yield from ("", "a", "b", value[:-1], value + "a")
    elif isinstance(value, list):
        yield []
        yield value[:1]
        yield value[:-1]
        yield list(reversed(value))
        if len(value) < 12:
            yield value + (value[:1] or [0])
        if depth < 2:
            for index in range(min(len(value), 4)):
                for changed in list(value_variants(value[index], constants[:6], depth + 1))[:8]:
                    updated = copy.deepcopy(value)
                    updated[index] = changed
                    yield updated
    elif isinstance(value, dict) and depth < 2:
        for name, item in value.items():
            for changed in list(value_variants(item, constants[:6], depth + 1))[:8]:
                yield {**value, name: changed}
        if set(value) == {"Some"}:
            yield {"None": None}
        elif set(value) == {"None"}:
            yield {"Some": 0}


def z3_candidates(base: dict, mutant: dict, direction: str, limit=8, purpose="difference") -> list[dict]:
    """Use the existing partial parser only to propose values; NEVER treat SAT as evidence."""
    from metrics_rebuild.share import precondition_satisfiability as parser
    if parser.z3 is None or any("&mut" in p["type"] for p in base.get("parameters", [])):
        return []
    params = list(base.get("parameters", []))
    post = direction.startswith("post_")
    if post:
        params += list(base.get("returns", []))
    # Keep model serialization exact for the simple types already supported here.
    if any(not (ce.is_int_like_type(p["type"]) or ce.is_bool_type(p["type"])) for p in params):
        return []
    z3 = parser.z3
    try:
        variables, constraints, _ = parser._z3_variables(params)
        def conjunction(ctx, kind):
            return z3.And(*[parser._z3_expr(c["text"], variables) for c in ctx.get(kind, [])])
        p0, pm = conjunction(base, "requires"), conjunction(mutant, "requires")
        conditions = {"pre_strengthening": z3.And(p0, z3.Not(pm)),
                      "pre_weakening": z3.And(pm, z3.Not(p0)),
                      "pre_omission": z3.And(pm, z3.Not(p0))}
        if post:
            q0, qm = conjunction(base, "ensures"), conjunction(mutant, "ensures")
            conditions.update(post_strengthening=z3.And(p0, pm, q0, z3.Not(qm)),
                              post_weakening=z3.And(p0, pm, qm, z3.Not(q0)),
                              post_omission=z3.And(p0, pm, qm, z3.Not(q0)))
        condition = conditions[direction]
        if purpose == "nondegenerate":
            strengthening = direction.endswith("strengthening")
            condition = (z3.And(pm, qm if strengthening else z3.Not(qm)) if post
                         else pm if strengthening else z3.Not(pm))
        solver = z3.Solver()
        solver.set(timeout=1000)
        solver.add(*constraints, condition)
        result = []
        while len(result) < limit and solver.check() == z3.sat:
            model = solver.model()
            values = {p["name"]: model.eval(variables[p["name"]], model_completion=True) for p in params}
            raw = {name: z3.is_true(v) if z3.is_bool(v) else v.as_long() for name, v in values.items()}
            result.append({"inputs": {p["name"]: raw[p["name"]] for p in base.get("parameters", [])},
                           "output": {p["name"]: raw[p["name"]] for p in base.get("returns", [])} if post else None,
                           "source": "z3_proposal"})
            solver.add(z3.Or(*[variables[name] != value for name, value in values.items()]))
        return result
    except (parser.UnsupportedExpression, ValueError, TypeError, KeyError, z3.Z3Exception):
        return []


def candidate_cases(base: dict, mutant: dict, suite: dict, direction: str,
                    changed_text: str, limit=256) -> list[dict]:
    post = direction.startswith("post_")
    seeds = mapped_cases(base, suite)
    constants = sorted({int(n) + offset for n in re.findall(r"(?<![\w])-?\d+(?![\w])", changed_text)
                        for offset in (-1, 0, 1) if abs(int(n)) < 2**64})[:32]
    defaults = {p["name"]: io._default_value_for_param(p, base.get("type_registry"))
                for p in base.get("parameters", [])}
    out_defaults = {p["name"]: io._default_value_for_param(p, base.get("type_registry"))
                    for p in base.get("returns", [])}
    seeds.append({"inputs": defaults, "output": out_defaults, "source": "typed_default"})
    proposals = z3_candidates(base, mutant, direction)
    proposals.extend(z3_candidates(base, mutant, direction, purpose="nondegenerate"))
    proposals.extend(seeds)
    # Output alternatives at the same legal input are essential for post strengthening.
    for seed in seeds:
        if post and isinstance(seed.get("output"), dict):
            for name, value in seed["output"].items():
                for changed in value_variants(value, constants):
                    proposals.append({**seed, "output": {**seed["output"], name: changed}, "source": "output_boundary"})
    for seed in seeds:
        for name, value in seed["inputs"].items():
            for changed in value_variants(value, constants):
                proposals.append({**seed, "inputs": {**seed["inputs"], name: changed}, "source": "input_boundary"})
    if post:
        for seed in seeds:
            for other in seeds:
                if isinstance(other.get("output"), dict):
                    proposals.append({**seed, "output": other["output"], "source": "cross_output"})
    result, seen = [], set()
    for proposal in proposals:
        inputs = ce.typed_input_payload(base, proposal["inputs"])
        output = ce.typed_output_payload(base, proposal.get("output")) if post else None
        if inputs is None or (post and output is None):
            continue
        key = digest([inputs, output])
        if key in seen:
            continue
        seen.add(key)
        result.append({"inputs": inputs, "output": output, "source": proposal["source"],
                       "key": f"case_{len(result):03d}"})
        if len(result) >= limit:
            break
    return result


def contract_decisions(ctx: dict, cases: list[dict]) -> dict:
    # The general I/O evaluator skips empty postconditions. In this experiment
    # an absent ensures denotes true, but the concrete precondition still matters.
    if not ctx.get("ensures"):
        return ce.batch_verus_requires_decide(ctx, cases)
    return ce.batch_verus_contract_decide(ctx, cases)


def witness_matches(direction: str, base_pre, mutant_pre, base_contract=None, mutant_contract=None) -> bool:
    if direction == "pre_strengthening":
        return base_pre is True and mutant_pre is False
    if direction in {"pre_omission", "pre_weakening"}:
        return base_pre is False and mutant_pre is True
    if base_pre is not True or mutant_pre is not True:
        return False
    if direction == "post_strengthening":
        return base_contract is True and mutant_contract is False
    return base_contract is False and mutant_contract is True


def suite_state(metric: dict) -> str:
    if int(metric.get("failed") or 0):
        return "detected"
    if metric.get("status") == "ok" and int(metric.get("total") or 0) > 0 and not metric.get("unknown"):
        return "clear"
    if metric.get("status") == "partial" and int(metric.get("total") or 0) > 0:
        return "unresolved"
    return "unavailable"


def pair_state(base_metric: dict, mutant_metric: dict) -> str:
    if suite_state(base_metric) != "clear":
        return "base_not_clear"
    return suite_state(mutant_metric)
