from __future__ import annotations

import re
from typing import Any, Mapping, Optional, Sequence

from ._shared import clause_kind, clause_normalized, clause_text, get_field

try:
    import z3
    Z3_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - handled at runtime for optional installs.
    z3 = None  # type: ignore[assignment]
    Z3_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"


TOKEN_RE = re.compile(
    r"""
    <==>|==>|=~=|!=|==|<=|>=|&&&|\|\|\||&&|\|\||::|->|\.\.|
    "(?:\\.|[^"\\])*"|
    '(?:\\.|[^'\\])'|
    0x[0-9A-Fa-f_]+(?:[A-Za-z_][A-Za-z0-9_]*)?|
    \d[\d_]*(?:\.\d[\d_]*)?(?:[eE][+-]?\d+)?(?:[A-Za-z_][A-Za-z0-9_]*)?|
    [A-Za-z_][A-Za-z0-9_]*|
    [()[\]{}+\-*/%!&<>,.:|;=^]
    """,
    re.VERBOSE,
)

UNSIGNED_BOUNDS = {
    "u8": (0, 2**8 - 1),
    "u16": (0, 2**16 - 1),
    "u32": (0, 2**32 - 1),
    "u64": (0, 2**64 - 1),
    "u128": (0, 2**128 - 1),
    "usize": (0, 2**64 - 1),
    "nat": (0, None),
}

SIGNED_BOUNDS = {
    "i8": (-(2**7), 2**7 - 1),
    "i16": (-(2**15), 2**15 - 1),
    "i32": (-(2**31), 2**31 - 1),
    "i64": (-(2**63), 2**63 - 1),
    "i128": (-(2**127), 2**127 - 1),
    "isize": (-(2**63), 2**63 - 1),
}

TYPE_CONSTANTS = {
    **{(name, "MAX"): upper for name, (_lower, upper) in UNSIGNED_BOUNDS.items() if upper is not None},
    **{(name, "MIN"): lower for name, (lower, _upper) in UNSIGNED_BOUNDS.items()},
    **{(name, "MAX"): upper for name, (_lower, upper) in SIGNED_BOUNDS.items()},
    **{(name, "MIN"): lower for name, (lower, _upper) in SIGNED_BOUNDS.items()},
}

INT_TYPES = set(UNSIGNED_BOUNDS) | set(SIGNED_BOUNDS) | {"int"}
REAL_TYPES = {"f32", "f64"}
BOOL_TYPES = {"bool"}
CHAR_TYPE = "char"


class UnsupportedExpression(ValueError):
    pass


def _preprocess_expr(expr: str) -> str:
    text = str(expr)
    text = re.sub(r"#\s*!\s*\[[^\n]*?\]\]", " ", text)
    text = re.sub(r"#\s*!\s*\[[^\]]*\]", " ", text)
    text = re.sub(r"#\s*\[\s*trigger\s*\]", " ", text)
    text = re.sub(r"#\s*\[[^\]]*\]", " ", text)
    text = re.sub(r",\s*$", "", text.strip())
    text = text.replace("&&&", "&&").replace("|||", "||")
    text = text.replace("=~=", "==")
    text = re.sub(r"(\d+\.\d+)(?:f32|f64)\b", r"\1", text)
    text = re.sub(
        r"(\d+)(?:u8|u16|u32|u64|u128|usize|i8|i16|i32|i64|i128|isize|nat|int)\b",
        r"\1",
        text,
    )
    text = text.replace("@", "")
    return text


def _tokens(expr: str) -> list[str]:
    return [match.group(0) for match in TOKEN_RE.finditer(_preprocess_expr(expr))]


def _base_type(type_text: str) -> str:
    cleaned = re.sub(r"\b(?:tracked|ghost|mut)\b", " ", str(type_text))
    cleaned = cleaned.replace("&", " ").replace("'", " ")
    match = re.search(
        r"\b(u8|u16|u32|u64|u128|usize|i8|i16|i32|i64|i128|isize|int|nat|bool|char|f32|f64)\b",
        cleaned,
    )
    return match.group(1) if match else "int"


def _collection_element_type(type_text: str) -> Optional[str]:
    cleaned = re.sub(r"\b(?:tracked|ghost|mut)\b", " ", str(type_text))
    cleaned = cleaned.replace("&", " ")
    match = re.search(r"\b(?:Vec|Seq)\s*<(?P<inner>.+)>", cleaned)
    if match:
        return _base_type(match.group("inner"))
    slice_match = re.search(r"\[\s*(?P<inner>[^;\]]+)", cleaned)
    if slice_match:
        return _base_type(slice_match.group("inner"))
    return None


def _collection_inner_type(type_text: str) -> Optional[str]:
    cleaned = re.sub(r"\b(?:tracked|ghost|mut)\b", " ", str(type_text))
    cleaned = cleaned.replace("&", " ")
    match = re.search(r"\b(?:Vec|Seq)\s*<(?P<inner>.+)>", cleaned)
    if match:
        return match.group("inner").strip()
    slice_match = re.search(r"\[\s*(?P<inner>[^;\]]+)", cleaned)
    if slice_match:
        return slice_match.group("inner").strip()
    return None


def _split_top_level_type_commas(text: str) -> list[str]:
    parts: list[str] = []
    start = 0
    angle = paren = bracket = 0
    for idx, ch in enumerate(text):
        if ch == "<":
            angle += 1
        elif ch == ">" and angle:
            angle -= 1
        elif ch == "(":
            paren += 1
        elif ch == ")" and paren:
            paren -= 1
        elif ch == "[":
            bracket += 1
        elif ch == "]" and bracket:
            bracket -= 1
        elif ch == "," and angle == 0 and paren == 0 and bracket == 0:
            parts.append(text[start:idx].strip())
            start = idx + 1
    tail = text[start:].strip()
    if tail:
        parts.append(tail)
    return parts


def _tuple_field_types(type_text: str) -> list[str]:
    cleaned = re.sub(r"\b(?:tracked|ghost|mut)\b", " ", str(type_text)).strip()
    cleaned = cleaned.replace("&", " ")
    if not (cleaned.startswith("(") and cleaned.endswith(")")):
        return []
    inner = cleaned[1:-1].strip()
    if ":" in inner:
        return []
    return _split_top_level_type_commas(inner)


def _is_collection_type(type_text: str) -> bool:
    return _collection_element_type(type_text) is not None


def _display_type(type_text: str) -> str:
    element = _collection_element_type(type_text)
    if element is None:
        return _base_type(type_text)
    if re.search(r"\bVec\s*<", str(type_text)):
        return f"Vec<{element}>"
    if re.search(r"\bSeq\s*<", str(type_text)):
        return f"Seq<{element}>"
    return f"Seq<{element}>"


def _parameter_type_map(parameters: Sequence[Mapping[str, str]]) -> dict[str, str]:
    type_map: dict[str, str] = {}
    for parameter in parameters:
        name = parameter.get("name", "")
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name):
            type_map[name] = _display_type(parameter.get("type", "int"))
    return type_map


def _numeric_literal_value(token: str) -> Optional[Any]:
    if z3 is None:
        raise UnsupportedExpression("z3_unavailable")
    compact = re.sub(
        r"(?:u8|u16|u32|u64|u128|usize|i8|i16|i32|i64|i128|isize|nat|int|f32|f64)\b",
        "",
        token.replace("_", ""),
    )
    if re.fullmatch(r"0x[0-9A-Fa-f]+", compact):
        return z3.IntVal(int(compact, 16))
    if re.fullmatch(r"\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", compact):
        if "." in compact or "e" in compact.lower():
            return z3.RealVal(compact)
        return z3.IntVal(int(compact))
    return None


def _as_bool(expr: Any) -> Any:
    if z3 is None:
        raise UnsupportedExpression("z3_unavailable")
    if z3.is_bool(expr):
        return expr
    raise UnsupportedExpression("expected_boolean_expression")


def _as_int(expr: Any) -> Any:
    if z3 is None:
        raise UnsupportedExpression("z3_unavailable")
    if z3.is_int_value(expr) or (hasattr(expr, "sort") and expr.sort() == z3.IntSort()):
        return expr
    raise UnsupportedExpression("expected_integer_expression")


def _as_arith(expr: Any) -> Any:
    if z3 is None:
        raise UnsupportedExpression("z3_unavailable")
    if z3.is_int_value(expr) or z3.is_rational_value(expr):
        return expr
    if hasattr(expr, "sort") and expr.sort() in {z3.IntSort(), z3.RealSort()}:
        return expr
    raise UnsupportedExpression("expected_numeric_expression")


def _sort_for_base_type(base_type: str) -> Any:
    if z3 is None:
        raise UnsupportedExpression("z3_unavailable")
    if base_type in BOOL_TYPES:
        return z3.BoolSort()
    if base_type in REAL_TYPES:
        return z3.RealSort()
    return z3.IntSort()


def _sort_for_type_text(type_text: str) -> Any:
    inner = _collection_inner_type(type_text)
    if inner is not None:
        return z3.ArraySort(z3.IntSort(), _sort_for_type_text(inner))
    return _sort_for_base_type(_base_type(type_text))


def _is_array_expr(expr: Any) -> bool:
    if z3 is None or not hasattr(expr, "sort"):
        return False
    try:
        return expr.sort().kind() == z3.Z3_ARRAY_SORT
    except Exception:
        return False


_EXPR_KEY_META = "__expr_keys__"
_OPAQUE_EXPR_META = "__opaque_expr_ids__"


def _sanitize_z3_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_") or "value"


def _sort_name(sort: Any) -> str:
    return _sanitize_z3_name(str(sort))


def _expr_meta(variables: dict[str, Any], name: str, default: Any) -> Any:
    if name not in variables:
        variables[name] = default
    return variables[name]


def _register_expr_key(variables: dict[str, Any], expr: Any, key: str, *, opaque: bool = False) -> Any:
    if hasattr(expr, "get_id"):
        _expr_meta(variables, _EXPR_KEY_META, {})[expr.get_id()] = key
        if opaque:
            _expr_meta(variables, _OPAQUE_EXPR_META, set()).add(expr.get_id())
    return expr


def _expr_key(variables: Mapping[str, Any], expr: Any) -> Optional[str]:
    if not hasattr(expr, "get_id"):
        return None
    meta = variables.get(_EXPR_KEY_META)
    if isinstance(meta, dict):
        return meta.get(expr.get_id())
    return None


def _is_opaque_expr(variables: Mapping[str, Any], expr: Any) -> bool:
    if not hasattr(expr, "get_id"):
        return False
    meta = variables.get(_OPAQUE_EXPR_META)
    return isinstance(meta, set) and expr.get_id() in meta


def _symbol_for_sort(
    variables: dict[str, Any],
    key: str,
    sort: Any,
    *,
    z3_name: Optional[str] = None,
    opaque: bool = True,
) -> Any:
    storage_key = f"__symbol__:{_sort_name(sort)}:{key}"
    if storage_key not in variables:
        name = _sanitize_z3_name(z3_name or key)
        if sort == z3.BoolSort():
            variables[storage_key] = z3.Bool(name)
        elif sort == z3.IntSort():
            variables[storage_key] = z3.Int(name)
        elif sort == z3.RealSort():
            variables[storage_key] = z3.Real(name)
        else:
            variables[storage_key] = z3.Const(name, sort)
    return _register_expr_key(variables, variables[storage_key], key, opaque=opaque)


def _array_range_sort(expr: Any) -> Any:
    if z3 is None or not _is_array_expr(expr):
        return z3.IntSort()
    return expr.sort().range()


def _array_sort_like(expr: Any) -> Any:
    if z3 is None:
        raise UnsupportedExpression("z3_unavailable")
    return z3.ArraySort(z3.IntSort(), _array_range_sort(expr))


def _expr_repr(variables: Mapping[str, Any], expr: Any) -> str:
    key = _expr_key(variables, expr)
    if key:
        return key
    if hasattr(expr, "sexpr"):
        return expr.sexpr()
    return str(expr)


def _char_literal_value(token: str) -> Optional[int]:
    if not re.fullmatch(r"'(?:\\.|[^'\\])'", token):
        return None
    inner = token[1:-1]
    escapes = {
        r"\n": "\n",
        r"\r": "\r",
        r"\t": "\t",
        r"\\": "\\",
        r"\'": "'",
        r"\"": '"',
        r"\0": "\0",
    }
    return ord(escapes.get(inner, inner[-1] if inner.startswith("\\") else inner))


class _Z3ExprParser:
    def __init__(self, tokens: Sequence[str], variables: dict[str, Any]):
        self.tokens = list(tokens)
        self.variables = variables
        self.pos = 0

    def parse(self) -> Any:
        expr = self._parse_equivalence()
        if self._peek() is not None:
            raise UnsupportedExpression(f"unexpected_token:{self._peek()}")
        return expr

    def _peek(self) -> Optional[str]:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def _take(self, token: Optional[str] = None) -> str:
        current = self._peek()
        if current is None:
            raise UnsupportedExpression("unexpected_end")
        if token is not None and current != token:
            raise UnsupportedExpression(f"expected:{token}")
        self.pos += 1
        return current

    def _as_bool(self, expr: Any) -> Any:
        if z3.is_bool(expr):
            return expr
        return self._coerce_sort(expr, z3.BoolSort(), "expected_boolean_expression")

    def _as_int(self, expr: Any) -> Any:
        if z3.is_int_value(expr) or (hasattr(expr, "sort") and expr.sort() == z3.IntSort()):
            return expr
        return self._coerce_sort(expr, z3.IntSort(), "expected_integer_expression")

    def _as_arith(self, expr: Any) -> Any:
        if z3.is_int_value(expr) or z3.is_rational_value(expr):
            return expr
        if hasattr(expr, "sort") and expr.sort() in {z3.IntSort(), z3.RealSort()}:
            return expr
        return self._coerce_sort(expr, z3.IntSort(), "expected_numeric_expression")

    def _coerce_sort(self, expr: Any, target_sort: Any, error: str) -> Any:
        if hasattr(expr, "sort") and expr.sort() == target_sort:
            return expr
        key = _expr_key(self.variables, expr)
        if key and _is_opaque_expr(self.variables, expr):
            return _symbol_for_sort(self.variables, key, target_sort, z3_name=key, opaque=True)
        raise UnsupportedExpression(error)

    def _align_for_equality(self, left: Any, right: Any) -> tuple[Any, Any]:
        if not hasattr(left, "sort") or not hasattr(right, "sort"):
            return left, right
        if left.sort() == right.sort():
            return left, right
        if _is_opaque_expr(self.variables, left):
            return self._coerce_sort(left, right.sort(), "sort mismatch"), right
        if _is_opaque_expr(self.variables, right):
            return left, self._coerce_sort(right, left.sort(), "sort mismatch")
        raise UnsupportedExpression("sort mismatch")

    def _align_if_branches(self, then_expr: Any, else_expr: Any) -> tuple[Any, Any]:
        if hasattr(then_expr, "sort") and hasattr(else_expr, "sort") and then_expr.sort() == else_expr.sort():
            return then_expr, else_expr
        if hasattr(then_expr, "sort") and _is_opaque_expr(self.variables, else_expr):
            return then_expr, self._coerce_sort(else_expr, then_expr.sort(), "sort mismatch")
        if hasattr(else_expr, "sort") and _is_opaque_expr(self.variables, then_expr):
            return self._coerce_sort(then_expr, else_expr.sort(), "sort mismatch"), else_expr
        raise UnsupportedExpression("sort mismatch")

    def _make_bound_variable(self, name: str, type_text: str) -> tuple[Any, list[Any]]:
        inner_type = _collection_inner_type(type_text)
        if inner_type is not None:
            variable = z3.Array(name, z3.IntSort(), _sort_for_type_text(inner_type))
            length = z3.Int(f"{name}_len")
            self.variables[f"{name}.len()"] = length
            constraints = [length >= 0]
            usize_upper = UNSIGNED_BOUNDS["usize"][1]
            if usize_upper is not None:
                constraints.append(length <= usize_upper)
            return variable, constraints

        base_type = _base_type(type_text)
        if base_type in BOOL_TYPES:
            return z3.Bool(name), []
        if base_type in REAL_TYPES:
            return z3.Real(name), []

        variable = z3.Int(name)
        constraints: list[Any] = []
        if base_type in UNSIGNED_BOUNDS:
            lower, upper = UNSIGNED_BOUNDS[base_type]
            constraints.append(variable >= lower)
            if upper is not None:
                constraints.append(variable <= upper)
        elif base_type in SIGNED_BOUNDS:
            lower, upper = SIGNED_BOUNDS[base_type]
            constraints.append(variable >= lower)
            constraints.append(variable <= upper)
        elif base_type == CHAR_TYPE:
            constraints.append(variable >= 0)
            constraints.append(variable <= 0x10FFFF)
        return variable, constraints

    def _parse_quantifier(self, quantifier: str) -> Any:
        self._take("|")
        bound_variables: list[Any] = []
        type_constraints: list[Any] = []
        saved: dict[str, Any] = {}

        while True:
            name = self._take()
            if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name):
                raise UnsupportedExpression(f"unsupported_quantifier_binder:{name}")

            type_text = "int"
            if self._peek() == ":":
                self._take(":")
                type_parts: list[str] = []
                while self._peek() not in {None, ",", "|"}:
                    type_parts.append(self._take())
                type_text = " ".join(type_parts) or "int"

            scoped_keys = [name, f"{name}.len()"]
            for scoped_key in scoped_keys:
                if scoped_key in self.variables:
                    saved[scoped_key] = self.variables[scoped_key]
            variable, constraints = self._make_bound_variable(name, type_text)
            self.variables[name] = variable
            bound_variables.append(variable)
            type_constraints.extend(constraints)

            if self._peek() == ",":
                self._take(",")
                continue
            break

        self._take("|")
        body = self._as_bool(self._parse_equivalence())
        scoped_names = set()
        for variable in bound_variables:
            for key, current in list(self.variables.items()):
                if current is variable:
                    scoped_names.add(key)
                    scoped_names.add(f"{key}.len()")
        for key in scoped_names:
            if key in saved:
                self.variables[key] = saved[key]
            else:
                self.variables.pop(key, None)
        for name, value in saved.items():
            if name not in scoped_names:
                self.variables[name] = value

        if quantifier == "forall":
            if type_constraints:
                return z3.ForAll(bound_variables, z3.Implies(_as_bool(_conjoin_z3(type_constraints)), body))
            return z3.ForAll(bound_variables, body)
        if type_constraints:
            return z3.Exists(bound_variables, z3.And(*type_constraints, body))
        return z3.Exists(bound_variables, body)

    def _parse_equivalence(self) -> Any:
        expr = self._parse_implication()
        while self._peek() == "<==>":
            self._take("<==>")
            right = self._parse_implication()
            expr = self._as_bool(expr) == self._as_bool(right)
        return expr

    def _parse_implication(self) -> Any:
        left = self._parse_or()
        if self._peek() == "==>":
            self._take("==>")
            right = self._parse_implication()
            return z3.Implies(self._as_bool(left), self._as_bool(right))
        return left

    def _parse_or(self) -> Any:
        expr = self._parse_and()
        while self._peek() == "||":
            self._take("||")
            expr = z3.Or(self._as_bool(expr), self._as_bool(self._parse_and()))
        return expr

    def _parse_and(self) -> Any:
        expr = self._parse_compare()
        while self._peek() == "&&":
            self._take("&&")
            expr = z3.And(self._as_bool(expr), self._as_bool(self._parse_compare()))
        return expr

    def _parse_compare(self) -> Any:
        left = self._parse_add()
        op = self._peek()
        if op not in {"==", "!=", "<=", "<", ">=", ">"}:
            return left
        result = None
        while op in {"==", "!=", "<=", "<", ">=", ">"}:
            self._take()
            right = self._parse_add()
            if op == "==":
                left, right = self._align_for_equality(left, right)
                comparison = left == right
            elif op == "!=":
                left, right = self._align_for_equality(left, right)
                comparison = left != right
            else:
                left_int = self._as_arith(left)
                right_int = self._as_arith(right)
                if op == "<=":
                    comparison = left_int <= right_int
                elif op == "<":
                    comparison = left_int < right_int
                elif op == ">=":
                    comparison = left_int >= right_int
                else:
                    comparison = left_int > right_int
            result = comparison if result is None else z3.And(self._as_bool(result), self._as_bool(comparison))
            left = right
            op = self._peek()
        return result

    def _parse_add(self) -> Any:
        expr = self._parse_mul()
        while self._peek() in {"+", "-", "^", "&", "|"}:
            op = self._take()
            right = self._parse_mul()
            if op == "+":
                if _is_array_expr(expr) or _is_array_expr(right):
                    array_expr = expr if _is_array_expr(expr) else right
                    key = f"({_expr_repr(self.variables, expr)} + {_expr_repr(self.variables, right)})"
                    expr = _symbol_for_sort(self.variables, key, _array_sort_like(array_expr), z3_name=key, opaque=True)
                else:
                    expr = self._as_arith(expr) + self._as_arith(right)
            elif op == "-":
                expr = self._as_arith(expr) - self._as_arith(right)
            else:
                key = f"({_expr_repr(self.variables, expr)} {op} {_expr_repr(self.variables, right)})"
                expr = _symbol_for_sort(self.variables, key, z3.IntSort(), z3_name=key, opaque=True)
        return expr

    def _parse_mul(self) -> Any:
        expr = self._parse_unary()
        while self._peek() in {"*", "/", "%"}:
            op = self._take()
            right = self._parse_unary()
            if op == "*":
                expr = self._as_arith(expr) * self._as_arith(right)
            elif op == "/":
                expr = self._as_arith(expr) / self._as_arith(right)
            else:
                expr = self._as_int(expr) % self._as_int(right)
        return expr

    def _parse_unary(self) -> Any:
        tok = self._peek()
        if tok == "!":
            self._take("!")
            return z3.Not(self._as_bool(self._parse_unary()))
        if tok == "-":
            self._take("-")
            return -self._as_arith(self._parse_unary())
        if tok == "*":
            self._take("*")
            return self._parse_unary()
        if tok == "&":
            self._take("&")
            if self._peek() == "mut":
                self._take("mut")
            return self._parse_unary()
        expr = self._parse_primary()
        while self._peek() == "as":
            self._take("as")
            self._consume_cast_type()
        return expr

    def _consume_cast_type(self) -> None:
        while self._peek() in {"&", "*"}:
            self._take()
            if self._peek() == "mut":
                self._take()
        tok = self._take()
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", tok):
            raise UnsupportedExpression(f"unsupported_cast_type:{tok}")
        if self._peek() == "<":
            depth = 1
            self._take("<")
            while depth > 0 and self._peek() is not None:
                t = self._take()
                if t == "<":
                    depth += 1
                elif t == ">":
                    depth -= 1

    def _parse_primary(self) -> Any:
        tok = self._take()
        if tok in {"forall", "exists"}:
            return self._parse_quantifier(tok)
        if tok == "if":
            return self._parse_if_expression()
        if tok == "match":
            return self._parse_match_expression()
        if tok == "{":
            return self._parse_braced_expression(opening_consumed=True)
        if tok == "(":
            if self._peek() == ")":
                self._take(")")
                return _symbol_for_sort(self.variables, "()", z3.IntSort(), z3_name="unit", opaque=True)
            expr = self._parse_equivalence()
            if self._peek() == ",":
                tuple_tokens = ["(", _expr_repr(self.variables, expr)]
                depth = 0
                while self._peek() is not None:
                    if self._peek() == ")" and depth == 0:
                        break
                    tok_in_tuple = self._take()
                    tuple_tokens.append(tok_in_tuple)
                    if tok_in_tuple in {"(", "[", "{"}:
                        depth += 1
                    elif tok_in_tuple in {")", "]", "}"} and depth > 0:
                        depth -= 1
                self._take(")")
                key = " ".join(tuple_tokens + [")"])
                return self._parse_postfix_existing(
                    _symbol_for_sort(self.variables, key, z3.IntSort(), z3_name=key, opaque=True),
                    key,
                )
            self._take(")")
            return self._parse_postfix_existing(expr, f"({_expr_repr(self.variables, expr)})")
        if tok == "[":
            literal_key = self._collect_bracket_literal_key(opening_consumed=True)
            array = _symbol_for_sort(
                self.variables,
                literal_key,
                z3.ArraySort(z3.IntSort(), z3.IntSort()),
                z3_name=literal_key,
                opaque=True,
            )
            return self._parse_postfix_existing(array, literal_key)
        if tok in {"true", "false"}:
            return z3.BoolVal(tok == "true")
        char_value = _char_literal_value(tok)
        if char_value is not None:
            return z3.IntVal(char_value)
        if re.fullmatch(r'"(?:\\.|[^"\\])*"', tok):
            return _symbol_for_sort(self.variables, tok, z3.IntSort(), z3_name=tok, opaque=True)
        numeric_value = _numeric_literal_value(tok)
        if numeric_value is not None:
            return numeric_value
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", tok):
            if tok == "old" and self._peek() == "(":
                self._take("(")
                inner_expr = self._parse_equivalence()
                self._take(")")
                inner_key = _expr_repr(self.variables, inner_expr)
                key = f"old({inner_key})"
                sort = inner_expr.sort() if hasattr(inner_expr, "sort") else z3.IntSort()
                expr = _symbol_for_sort(
                    self.variables, key, sort,
                    z3_name=f"old_{_sanitize_z3_name(inner_key)}",
                    opaque=True,
                )
                return self._parse_postfix_existing(expr, key)
            if self._peek() == "::":
                parts = [tok]
                while self._peek() == "::":
                    self._take("::")
                    if self._peek() == "<":
                        generic_tokens = [self._take("<")]
                        depth = 1
                        while depth > 0:
                            generic_tok = self._take()
                            generic_tokens.append(generic_tok)
                            if generic_tok == "<":
                                depth += 1
                            elif generic_tok == ">":
                                depth -= 1
                        parts.append("".join(generic_tokens))
                        continue
                    parts.append(self._take())
                path_key = "::".join(parts)
                if len(parts) == 2 and self._peek() != "(":
                    value = TYPE_CONSTANTS.get((parts[0], parts[1]))
                    if value is not None:
                        return z3.IntVal(value)
                return self._parse_postfix_value(path_key, path_key)
            return self._parse_postfix_value(tok, tok)
        raise UnsupportedExpression(f"unsupported_token:{tok}")

    def _parse_if_expression(self) -> Any:
        if self._peek() == "let":
            return self._parse_if_let_expression()
        condition = self._as_bool(self._parse_equivalence())
        then_expr = self._parse_braced_expression()
        self._take("else")
        if self._peek() == "if":
            self._take("if")
            else_expr = self._parse_if_expression()
        else:
            else_expr = self._parse_braced_expression()
        then_expr, else_expr = self._align_if_branches(then_expr, else_expr)
        return z3.If(condition, then_expr, else_expr)

    def _parse_if_let_expression(self) -> Any:
        self._take("let")
        pattern_tokens: list[str] = []
        binders: list[str] = []
        while self._peek() not in {None, "="}:
            tok = self._take()
            pattern_tokens.append(tok)
            if (
                re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", tok)
                and tok not in {"Some", "None", "Option"}
                and self._peek() in {")", ","}
            ):
                binders.append(tok)
        self._take("=")
        scrutinee = self._parse_equivalence()
        condition_key = f"if let {' '.join(pattern_tokens)} = {_expr_repr(self.variables, scrutinee)}"
        condition = _symbol_for_sort(self.variables, condition_key, z3.BoolSort(), z3_name=condition_key, opaque=True)
        saved = {name: self.variables.get(name) for name in binders}
        missing = {name for name in binders if name not in self.variables}
        for name in binders:
            self.variables[name] = z3.Int(name)
        then_expr = self._parse_braced_expression()
        for name, value in saved.items():
            if name in missing:
                self.variables.pop(name, None)
            else:
                self.variables[name] = value
        self._take("else")
        if self._peek() == "if":
            self._take("if")
            else_expr = self._parse_if_expression()
        else:
            else_expr = self._parse_braced_expression()
        then_expr, else_expr = self._align_if_branches(then_expr, else_expr)
        return z3.If(condition, then_expr, else_expr)

    def _parse_match_expression(self) -> Any:
        tokens = ["match"]
        seen_brace = False
        depth = 0
        while self._peek() is not None:
            tok = self._take()
            tokens.append(tok)
            if tok == "{":
                seen_brace = True
                depth += 1
            elif tok == "}":
                depth -= 1
                if seen_brace and depth == 0:
                    break
        if not seen_brace:
            raise UnsupportedExpression("unsupported_match_expression")
        key = " ".join(tokens)
        return _symbol_for_sort(self.variables, key, z3.BoolSort(), z3_name=key, opaque=True)

    def _parse_braced_expression(self, *, opening_consumed: bool = False) -> Any:
        if not opening_consumed:
            self._take("{")
        local_binders: list[str] = []
        while self._peek() in {"&&", "||", ";"}:
            self._take()
        while self._peek() == "let":
            local_binders.extend(self._skip_let_statement())
        expr = self._parse_equivalence()
        while self._peek() == ";":
            self._take(";")
            if self._peek() == "}":
                break
            if self._peek() == "let":
                local_binders.extend(self._skip_let_statement())
                continue
            expr = self._parse_equivalence()
        self._take("}")
        for name in local_binders:
            self.variables.pop(name, None)
        return expr

    def _skip_let_statement(self) -> list[str]:
        self._take("let")
        binders: list[str] = []
        while self._peek() not in {None, ";"}:
            tok = self._take()
            if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", tok) and tok not in {"let", "mut"}:
                binders.append(tok)
        if self._peek() == ";":
            self._take(";")
        added: list[str] = []
        for name in binders:
            if name not in self.variables:
                self.variables[name] = z3.Int(name)
                added.append(name)
        return added

    def _collect_until_balanced(self, opening: str, closing: str) -> list[str]:
        self._take(opening)
        tokens = [opening]
        depth = 1
        while depth > 0:
            tok = self._take()
            tokens.append(tok)
            if tok == opening:
                depth += 1
            elif tok == closing:
                depth -= 1
        return tokens

    def _collect_call_text(self) -> str:
        return " ".join(self._collect_until_balanced("(", ")"))

    def _collect_bracket_literal_key(self, *, opening_consumed: bool = False) -> str:
        tokens = ["["]
        if not opening_consumed:
            self._take("[")
        depth = 1
        while depth > 0:
            tok = self._take()
            tokens.append(tok)
            if tok == "[":
                depth += 1
            elif tok == "]":
                depth -= 1
        return " ".join(tokens)

    def _parse_argument_list(self) -> list[Any]:
        self._take("(")
        args: list[Any] = []
        if self._peek() == ")":
            self._take(")")
            return args
        while True:
            args.append(self._parse_equivalence())
            if self._peek() == ",":
                self._take(",")
                if self._peek() == ")":
                    break
                continue
            break
        self._take(")")
        return args

    def _parse_call_key_only(self, key: str) -> str:
        call_text = self._collect_call_text()
        return f"{key}{call_text}"

    def _known_function_call(self, key: str) -> Optional[Any]:
        saved_pos = self.pos
        try:
            args = self._parse_argument_list()
        except UnsupportedExpression:
            self.pos = saved_pos
            return None
        name = key.split("::")[-1].split(".")[-1]
        if name == "abs" and len(args) == 1:
            arg = self._as_arith(args[0])
            return z3.If(arg >= 0, arg, -arg)
        if name == "min" and len(args) == 2:
            left, right = self._as_arith(args[0]), self._as_arith(args[1])
            return z3.If(left <= right, left, right)
        if name == "max" and len(args) == 2:
            left, right = self._as_arith(args[0]), self._as_arith(args[1])
            return z3.If(left >= right, left, right)
        self.pos = saved_pos
        return None

    def _parse_postfix_value(self, key: str, z3_name: str) -> Any:
        if self._peek() == "!" and self.pos + 1 < len(self.tokens) and self.tokens[self.pos + 1] in {"(", "["}:
            self._take("!")
            if self._peek() == "(":
                macro_key = f"{key}!{self._collect_call_text()}"
                return _symbol_for_sort(self.variables, macro_key, z3.BoolSort(), z3_name=macro_key, opaque=True)
            macro_key = f"{key}!{self._collect_bracket_literal_key()}"
            return self._parse_postfix_existing(
                _symbol_for_sort(
                    self.variables,
                    macro_key,
                    z3.ArraySort(z3.IntSort(), z3.IntSort()),
                    z3_name=macro_key,
                    opaque=True,
                ),
                macro_key,
            )

        if self._peek() == "(":
            known = self._known_function_call(key)
            if known is not None:
                return self._parse_postfix_existing(known, key)
            call_key = self._parse_call_key_only(key)
            return self._parse_postfix_existing(
                _symbol_for_sort(self.variables, call_key, z3.BoolSort(), z3_name=call_key, opaque=True),
                call_key,
            )

        if key in self.variables:
            expr = self.variables[key]
            _register_expr_key(self.variables, expr, key, opaque=False)
        else:
            expr = _symbol_for_sort(self.variables, key, z3.IntSort(), z3_name=z3_name, opaque=True)
        return self._parse_postfix_existing(expr, key)

    def _parse_postfix_existing(self, expr: Any, key: str) -> Any:
        current = expr
        current_key = key
        while True:
            if self._peek() == ".":
                self._take(".")
                member = self._take()
                if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", member) and not member.isdigit():
                    raise UnsupportedExpression(f"unsupported_member:{current_key}.{member}")

                if self._peek() == "(":
                    method_key = f"{current_key}.{member}"
                    if member == "len":
                        self._take("(")
                        self._take(")")
                        len_key = f"{current_key}.len()"
                        current = _symbol_for_sort(self.variables, len_key, z3.IntSort(), z3_name=f"{_sanitize_z3_name(current_key)}_len", opaque=False)
                        current_key = len_key
                        continue
                    if member == "index":
                        saved_pos = self.pos
                        try:
                            args = self._parse_argument_list()
                        except UnsupportedExpression:
                            self.pos = saved_pos
                            call_key = self._parse_call_key_only(method_key)
                            current = _symbol_for_sort(self.variables, call_key, _array_range_sort(current), z3_name=call_key, opaque=True)
                            current_key = call_key
                            continue
                        if len(args) != 1:
                            call_key = f"{method_key}({', '.join(_expr_repr(self.variables, arg) for arg in args)})"
                            current = _symbol_for_sort(
                                self.variables,
                                call_key,
                                _array_range_sort(current),
                                z3_name=call_key,
                                opaque=True,
                            )
                            current_key = call_key
                            continue
                        index = self._as_int(args[0])
                        created_array = not _is_array_expr(current)
                        array = current if not created_array else _symbol_for_sort(
                            self.variables,
                            current_key,
                            z3.ArraySort(z3.IntSort(), z3.IntSort()),
                            z3_name=current_key,
                            opaque=True,
                        )
                        current = z3.Select(array, index)
                        current_key = f"{current_key}.index({_expr_repr(self.variables, index)})"
                        _register_expr_key(
                            self.variables,
                            current,
                            current_key,
                            opaque=created_array or _is_opaque_expr(self.variables, array),
                        )
                        continue

                    if member == "is_empty":
                        self._take("(")
                        self._take(")")
                        len_key = f"{current_key}.len()"
                        len_var = _symbol_for_sort(
                            self.variables, len_key, z3.IntSort(),
                            z3_name=f"{_sanitize_z3_name(current_key)}_len",
                            opaque=False,
                        )
                        current = len_var == z3.IntVal(0)
                        current_key = f"{current_key}.is_empty()"
                        _register_expr_key(self.variables, current, current_key)
                        continue

                    call_key = self._parse_call_key_only(method_key)
                    if member in {"contains", "is_prefix_of", "is_suffix_of", "is_some", "is_none"}:
                        current = _symbol_for_sort(self.variables, call_key, z3.BoolSort(), z3_name=call_key, opaque=True)
                    elif member in {"filter", "map", "take", "drop", "skip", "subrange", "push", "insert", "remove", "reverse", "add", "to_seq"}:
                        current = _symbol_for_sort(self.variables, call_key, _array_sort_like(current), z3_name=call_key, opaque=True)
                    elif member in {"unwrap", "first", "last"}:
                        current = _symbol_for_sort(self.variables, call_key, _array_range_sort(current), z3_name=call_key, opaque=True)
                    else:
                        current = _symbol_for_sort(self.variables, call_key, z3.BoolSort(), z3_name=call_key, opaque=True)
                    current_key = call_key
                    continue

                current_key = f"{current_key}.{member}"
                if current_key in self.variables:
                    current = self.variables[current_key]
                    _register_expr_key(self.variables, current, current_key, opaque=False)
                else:
                    current = _symbol_for_sort(self.variables, current_key, z3.IntSort(), z3_name=current_key, opaque=True)
                continue

            if self._peek() == "[":
                self._take("[")
                index = self._as_int(self._parse_equivalence())
                self._take("]")
                created_array = not _is_array_expr(current)
                array = current if not created_array else _symbol_for_sort(
                    self.variables,
                    current_key,
                    z3.ArraySort(z3.IntSort(), z3.IntSort()),
                    z3_name=current_key,
                    opaque=True,
                )
                current = z3.Select(array, index)
                current_key = f"{current_key}[{_expr_repr(self.variables, index)}]"
                _register_expr_key(
                    self.variables,
                    current,
                    current_key,
                    opaque=created_array or _is_opaque_expr(self.variables, array),
                )
                continue

            return current


def _conjoin_z3(exprs: Sequence[Any]) -> Any:
    if z3 is None:
        raise UnsupportedExpression("z3_unavailable")
    if not exprs:
        return z3.BoolVal(True)
    if len(exprs) == 1:
        return exprs[0]
    return z3.And(*exprs)


def _parameter_records(parameters: Any) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    if not isinstance(parameters, Sequence) or isinstance(parameters, (str, bytes)):
        return records
    for parameter in parameters:
        name = get_field(parameter, "name")
        type_text = get_field(parameter, "type")
        if name:
            records.append({"name": str(name), "type": str(type_text or "int")})
    return records


def _requires_records(context: Any) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    requires = get_field(context, "requires")
    if isinstance(requires, Sequence) and not isinstance(requires, (str, bytes)):
        for item in requires:
            text = clause_text(item)
            records.append(
                {
                    "kind": clause_kind(item) or "requires",
                    "text": text,
                    "normalized": clause_normalized(item),
                }
            )
    return records


def _contexts_from_input(items: Sequence[Any]) -> list[dict[str, Any]]:
    if not items:
        return []

    if items and get_field(items[0], "requires") is not None:
        contexts = []
        for item in items:
            contexts.append(
                {
                    "function": str(get_field(item, "function", "<unknown>")),
                    "mode": str(get_field(item, "mode", "exec")),
                    "parameters": _parameter_records(get_field(item, "parameters", [])),
                    "requires": _requires_records(item),
                }
            )
        return contexts

    requires = [
        {"text": clause_text(clause), "normalized": clause_normalized(clause)}
        for clause in items
        if clause_kind(clause) == "requires"
    ]
    return [{"function": "<all_requires>", "parameters": [], "requires": requires}]


def _z3_variables(parameters: Sequence[Mapping[str, str]]) -> tuple[dict[str, Any], list[Any], dict[str, str]]:
    if z3 is None:
        raise UnsupportedExpression("z3_unavailable")

    variables: dict[str, Any] = {}
    constraints: list[Any] = []
    type_map = _parameter_type_map(parameters)
    for parameter in parameters:
        name = parameter.get("name", "")
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name):
            continue
        full_type = parameter.get("type", "int")
        tuple_fields = _tuple_field_types(full_type)
        if tuple_fields:
            variables[name] = z3.Int(name)
            for idx, field_type in enumerate(tuple_fields):
                field_name = f"{name}.{idx}"
                field_inner = _collection_inner_type(field_type)
                if field_inner is not None:
                    variables[field_name] = z3.Array(field_name.replace(".", "_"), z3.IntSort(), _sort_for_type_text(field_inner))
                    len_key = f"{field_name}.len()"
                    variables[len_key] = z3.Int(f"{name}_{idx}_len")
                    constraints.append(variables[len_key] >= 0)
                    usize_upper = UNSIGNED_BOUNDS["usize"][1]
                    if usize_upper is not None:
                        constraints.append(variables[len_key] <= usize_upper)
                else:
                    field_base = _base_type(field_type)
                    if field_base in BOOL_TYPES:
                        variables[field_name] = z3.Bool(field_name.replace(".", "_"))
                    elif field_base in REAL_TYPES:
                        variables[field_name] = z3.Real(field_name.replace(".", "_"))
                    else:
                        variables[field_name] = z3.Int(field_name.replace(".", "_"))
            continue
        inner_type = _collection_inner_type(full_type)
        if inner_type is not None:
            variables[name] = z3.Array(name, z3.IntSort(), _sort_for_type_text(inner_type))
            len_key = f"{name}.len()"
            variables[len_key] = z3.Int(f"{name}_len")
            constraints.append(variables[len_key] >= 0)
            usize_upper = UNSIGNED_BOUNDS["usize"][1]
            if usize_upper is not None:
                constraints.append(variables[len_key] <= usize_upper)
            continue

        base_type = _base_type(full_type)
        if base_type in BOOL_TYPES:
            variables[name] = z3.Bool(name)
            continue
        if base_type in REAL_TYPES:
            variables[name] = z3.Real(name)
            continue

        variables[name] = z3.Int(name)
        if base_type == CHAR_TYPE:
            constraints.append(variables[name] >= 0)
            constraints.append(variables[name] <= 0x10FFFF)
        elif base_type in UNSIGNED_BOUNDS:
            lower, upper = UNSIGNED_BOUNDS[base_type]
            constraints.append(variables[name] >= lower)
            if upper is not None:
                constraints.append(variables[name] <= upper)
        elif base_type in SIGNED_BOUNDS:
            lower, upper = SIGNED_BOUNDS[base_type]
            constraints.append(variables[name] >= lower)
            constraints.append(variables[name] <= upper)
    return variables, constraints, type_map


def _z3_expr(expr: str, variables: dict[str, Any]) -> Any:
    if z3 is None:
        raise UnsupportedExpression("z3_unavailable")
    tokens = _tokens(expr)
    if not tokens:
        raise UnsupportedExpression("empty_expression")
    if "=>" in tokens:
        raise UnsupportedExpression("unsupported_match_arrow")
    parser = _Z3ExprParser(tokens, variables)
    return parser._as_bool(parser.parse())


def _check_context(context: Mapping[str, Any], timeout_ms: int) -> dict[str, Any]:
    requires = list(context.get("requires", []))
    parameters = list(context.get("parameters", []))
    function = str(context.get("function", "<unknown>"))
    mode = str(context.get("mode", "exec"))
    kind_counts: dict[str, int] = {}
    for clause in requires:
        kind = str(clause.get("kind") or "requires")
        kind_counts[kind] = kind_counts.get(kind, 0) + 1

    if z3 is None:
        if not requires:
            status = "ok"
            satisfiable = True
            unsupported = []
        else:
            status = "unavailable"
            satisfiable = None
            unsupported = [
                {"clause": str(item.get("normalized") or item.get("text", "")), "reason": "z3_unavailable"}
                for item in requires
            ]
        return {
            "function": function,
            "mode": mode,
            "status": status,
            "satisfiable": satisfiable,
            "requires_total": len(requires),
            "preconditions_total": len(requires),
            "precondition_kind_counts": kind_counts,
            "supported_requires": 0,
            "unsupported_requires": len(unsupported),
            "parameters": parameters,
            "parameter_types": _parameter_type_map(parameters),
            "unsupported": unsupported,
            "unsat_reasons": [],
            "z3_import_error": Z3_IMPORT_ERROR or "z3 python package is not importable",
        }

    variables, type_constraints, type_map = _z3_variables(parameters)
    solver = z3.Solver()
    solver.set(timeout=timeout_ms)
    for constraint in type_constraints:
        solver.add(constraint)

    unsupported: list[dict[str, str]] = []
    supported = 0
    for clause in requires:
        text = str(clause.get("text", ""))
        normalized = str(clause.get("normalized") or text)
        try:
            solver.add(_z3_expr(text, variables))
            supported += 1
        except UnsupportedExpression as exc:
            unsupported.append({"clause": normalized, "reason": str(exc)})

    for name, variable in variables.items():
        if name.endswith(".len()"):
            solver.add(variable >= 0)

    check = solver.check()
    unsat = check == z3.unsat
    if unsat:
        status = "unsat"
        satisfiable = False
    elif check == z3.sat and not unsupported:
        status = "ok"
        satisfiable = True
    elif check == z3.sat:
        status = "unknown"
        satisfiable = None
    else:
        status = str(check)
        satisfiable = None
    return {
        "function": function,
        "mode": mode,
        "status": status,
        "satisfiable": satisfiable,
        "requires_total": len(requires),
        "preconditions_total": len(requires),
        "precondition_kind_counts": kind_counts,
        "supported_requires": supported,
        "unsupported_requires": len(unsupported),
        "parameters": parameters,
        "parameter_types": type_map,
        "unsupported": unsupported,
        "unsat_reasons": [],
        "z3_assertions": len(solver.assertions()),
        "z3_timeout_ms": timeout_ms,
    }


def precondition_satisfiability(items: Sequence[Any], timeout_ms: int = 1000) -> dict:
    contexts = _contexts_from_input(items)
    context_results = [_check_context(context, timeout_ms) for context in contexts]
    unsat_contexts = [context for context in context_results if context.get("status") == "unsat"]
    unsupported_total = sum(int(context.get("unsupported_requires", 0)) for context in context_results)
    requires_total = sum(int(context.get("requires_total", 0)) for context in context_results)
    unavailable = any(context.get("status") == "unavailable" for context in context_results)
    unknown = unavailable or any(context.get("satisfiable") is None for context in context_results)
    score = 0.0 if unsat_contexts else None if unknown else 1.0
    status = "unavailable" if unavailable else "unsat" if unsat_contexts else "unknown" if unknown else "ok"
    return {
        "score": score,
        "satisfiable_proxy": not unsat_contexts and not unknown,
        "status": status,
        "requires_total": requires_total,
        "contexts_total": len(context_results),
        "unsat_flags": [
            {
                "function": context.get("function"),
                "requires_total": context.get("requires_total"),
                "supported_requires": context.get("supported_requires"),
                "parameter_types": context.get("parameter_types", {}),
                "reasons": context.get("unsat_reasons") or ["z3_unsat"],
            }
            for context in unsat_contexts
        ],
        "unsupported_requires": unsupported_total,
        "contexts": context_results,
        "engine": "z3",
    }
