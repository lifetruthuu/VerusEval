"""Lightweight Pratt parser for Verus specification expressions.

This module turns a specification clause body (the text after ``requires`` /
``ensures`` / ``invariant`` / ...) into a shallow expression tree and derives
structural signals used by ``spec_size_complexity``:

* **Node count**    — number of non-transparent AST nodes.
* **Logical depth** — nesting depth of the boolean / quantifier skeleton
  (``&&``, ``||``, ``==>``, ``<==>``, quantifier bodies, ``if`` / ``match``),
  ignoring arithmetic and call-argument nesting.
* **Quantifiers**   — number of quantifier nodes.
* **Vocabulary**    — distinct operator kinds and distinct identifier names.

The parser is deliberately *total*: it never raises on malformed input and
always makes forward progress, so it can run over arbitrary LLM output. When it
cannot fully consume a clause it still returns the analysis of what it parsed
and flags ``parse_ok = False``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from metrics_rebuild.share.text import strip_comments

# --------------------------------------------------------------------------- #
# Operator classification (Verus-aware)
# --------------------------------------------------------------------------- #

LOGICAL_OPS = frozenset({"&&", "||", "&&&", "|||", "==>", "<==", "<==>"})
COMPARE_OPS = frozenset({"==", "!=", "<", "<=", ">", ">=", "=~=", "=~~=", "==="})
ARITH_OPS = frozenset({"+", "-", "*", "/", "%", "<<", ">>", "&", "^"})
RANGE_OPS = frozenset({"..", "..="})

# Keyword spellings that Verus accepts as synonyms of a symbolic operator. They
# are normalized at node construction so that `a implies b` and `a ==> b`
# contribute the same operator to the vocabulary and fold the same way in
# `_logic_depth`.
_OP_ALIASES: Dict[str, str] = {"implies": "==>"}

# Binding powers (higher binds tighter). Right-associative operators get a
# slightly lower right power so that `a ==> b ==> c` parses as `a ==> (b ==> c)`.
_INFIX_BP: Dict[str, float] = {
    # `decreases E when C` / `decreases E via f` clause modifiers bind loosest.
    "when": 0.5, "via": 0.5,
    "=>": 1,
    "<==>": 2,
    # `==>` is right-associative, its mirror `<==` is left-associative.
    "==>": 4, "implies": 4, "<==": 4,
    "||": 6, "|||": 6,
    "&&": 8, "&&&": 8,
    "==": 10, "!=": 10, "<": 10, "<=": 10, ">": 10, ">=": 10,
    "=~=": 10, "=~~=": 10, "===": 10,
    "matches": 10,
    "..": 11, "..=": 11,
    "|": 12, "^": 13, "&": 14,
    "<<": 15, ">>": 15,
    # `++` is deliberately absent: Verus concatenates sequences with `+` or
    # `.add()`. `++` is Dafny syntax, and accepting it here would silently bless
    # a real defect in generated specs instead of flagging it as unparsed.
    "+": 16, "-": 16,
    "*": 18, "/": 18, "%": 18,
    "as": 20,
}
_RIGHT_ASSOC = frozenset({"==>", "<==>", "implies"})
_PREFIX_OPS = frozenset({"!", "-", "*", "&"})
_PREFIX_BP = 22.0
# Verus also allows a leading connective in a bulleted conjunction/disjunction
# list (`&&& a &&& b`), which is equivalent to the infix form `a &&& b`.
_PREFIX_CONNECTIVES = frozenset({"&&&", "|||"})
# Infix operators spelled as identifiers rather than punctuation.
_KEYWORD_INFIX = frozenset({"as", "matches", "implies", "when", "via"})

_QUANTIFIERS = frozenset({"forall", "exists", "choose"})

# --------------------------------------------------------------------------- #
# Lexer
# --------------------------------------------------------------------------- #

_STR_RE = re.compile(r'"(?:\\.|[^"\\])*"', re.DOTALL)
_CHAR_RE = re.compile(r"'(?:\\.|[^'\\])'")
_LIFETIME_RE = re.compile(r"'[A-Za-z_][A-Za-z0-9_]*")
_NUM_RE = re.compile(
    r"(?:0x[0-9A-Fa-f_]+|0b[01_]+|0o[0-7_]+"
    r"|\d[\d_]*(?:\.\d[\d_]*)?(?:[eE][+-]?\d[\d_]*)?)"
    r"(?:[iu](?:8|16|32|64|128|size)|f(?:32|64)|nat|int)?"
)
_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
# Multi-char operators, longest match first (the alternation is tried in order,
# so longer spellings must precede their prefixes: `===` before `==`).
_MULTI_OPS = [
    "<==>", "=~~=", "=~=", "===", "&&&", "|||", "==>", "<==", "..=",
    "::", "->", "=>", "==", "!=", "<=", ">=", "<<", ">>", "&&", "||", "..",
]
_MULTI_OP_RE = re.compile("|".join(re.escape(op) for op in _MULTI_OPS))
_SINGLE_STRUCT = {
    "(": "LPAREN", ")": "RPAREN",
    "[": "LBRACK", "]": "RBRACK",
    "{": "LBRACE", "}": "RBRACE",
    ",": "COMMA", ";": "SEMI", "|": "PIPE", "#": "HASH",
}
_SINGLE_OPS = set("+-*/%!<>&^~=.@?:")


@dataclass(frozen=True)
class Token:
    kind: str
    value: str


def lex(text: str) -> List[Token]:
    """Tokenize a spec expression, correctly separating char literals from
    lifetimes."""
    toks: List[Token] = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch.isspace():
            i += 1
            continue
        if ch == '"':
            m = _STR_RE.match(text, i)
            if m:
                toks.append(Token("STR", m.group()))
                i = m.end()
                continue
            toks.append(Token("UNKNOWN", ch))
            i += 1
            continue
        if ch == "'":
            m = _CHAR_RE.match(text, i)
            if m:
                toks.append(Token("CHAR", m.group()))
                i = m.end()
                continue
            m = _LIFETIME_RE.match(text, i)
            if m:
                toks.append(Token("LIFETIME", m.group()))
                i = m.end()
                continue
            toks.append(Token("UNKNOWN", ch))
            i += 1
            continue
        if ch.isdigit():
            m = _NUM_RE.match(text, i)
            if m:
                toks.append(Token("NUM", m.group()))
                i = m.end()
                continue
        # Only ASCII Verus idents (`[A-Za-z_][A-Za-z0-9_]*`). Non-ASCII
        # letter-like characters (e.g. Cyrillic/Arabic in malformed asserts)
        # must not call `.group()` on a failed match; treat them as UNKNOWN.
        m = _IDENT_RE.match(text, i)
        if m:
            toks.append(Token("IDENT", m.group()))
            i = m.end()
            continue
        m = _MULTI_OP_RE.match(text, i)
        if m:
            toks.append(Token("OP", m.group()))
            i = m.end()
            continue
        struct = _SINGLE_STRUCT.get(ch)
        if struct is not None:
            toks.append(Token(struct, ch))
            i += 1
            continue
        if ch in _SINGLE_OPS:
            toks.append(Token("OP", ch))
            i += 1
            continue
        toks.append(Token("UNKNOWN", ch))
        i += 1  # unknown byte — retain a diagnostic token and make progress
    toks.append(Token("EOF", ""))
    return toks


# --------------------------------------------------------------------------- #
# AST
# --------------------------------------------------------------------------- #

# Node kinds that add a level to the *logical* skeleton.
_DEPTH_KINDS = frozenset({"LOGICAL", "QUANT", "IF", "MATCH"})
# Node kinds that are structurally transparent (not counted as their own node).
_TRANSPARENT = frozenset({"GROUP", "BLOCK", "ATTR", "EMPTY"})


@dataclass
class Node:
    kind: str
    op: str = ""
    name: str = ""
    children: List["Node"] = field(default_factory=list)
    qkind: str = ""  # for QUANT: forall / exists / choose
    binders: Tuple[str, ...] = ()


# --------------------------------------------------------------------------- #
# Pratt parser
# --------------------------------------------------------------------------- #

_MAX_TOKENS = 20000  # hard ceiling; guards against pathological inputs


class Parser:
    def __init__(self, tokens: Sequence[Token]) -> None:
        self.toks = tokens
        self.pos = 0
        self.triggers = 0
        self.parse_ok = True

    # -- token helpers ----------------------------------------------------- #
    def peek(self, k: int = 0) -> Token:
        idx = self.pos + k
        if idx < len(self.toks):
            return self.toks[idx]
        return self.toks[-1]

    def advance(self) -> Token:
        tok = self.peek()
        if self.pos < len(self.toks) - 1:
            self.pos += 1
        return tok

    def at_end(self) -> bool:
        return self.peek().kind == "EOF"

    def eat(self, kind: str) -> bool:
        if self.peek().kind == kind:
            self.advance()
            return True
        return False

    # -- entry ------------------------------------------------------------- #
    def parse(self) -> Node:
        node = self.parse_expr(0.0)
        if not self.at_end():
            # leftover tokens: partial parse.
            self.parse_ok = False
        return node

    # -- Pratt core -------------------------------------------------------- #
    def parse_expr(self, min_bp: float) -> Node:
        left = self.nud()
        guard = 0
        while True:
            guard += 1
            if guard > _MAX_TOKENS:
                self.parse_ok = False
                break
            left = self.parse_postfix(left)
            op = self._infix_op()
            if op is None:
                break
            base = _INFIX_BP[op]
            if base <= min_bp:
                break
            self.advance()  # consume the operator
            rbp = base - 0.5 if op in _RIGHT_ASSOC else base + 0.5
            right = self.parse_expr(rbp)
            symbol = _OP_ALIASES.get(op, op)
            kind = "LOGICAL" if symbol in LOGICAL_OPS else "BINARY"
            left = Node(kind, op=symbol, children=[left, right])
        return left

    def _infix_op(self) -> Optional[str]:
        tok = self.peek()
        if tok.kind == "OP" and tok.value in _INFIX_BP:
            return tok.value
        if tok.kind == "PIPE":  # bitor in infix position
            return "|"
        if tok.kind == "IDENT" and tok.value in _KEYWORD_INFIX:
            return tok.value
        return None

    def parse_postfix(self, left: Node) -> Node:
        guard = 0
        while True:
            guard += 1
            if guard > _MAX_TOKENS:
                self.parse_ok = False
                return left
            tok = self.peek()
            if (
                tok.kind == "OP"
                and tok.value == "!"
                and self.peek(1).kind in {"LPAREN", "LBRACK", "LBRACE"}
            ):
                self.advance()
                opener = self.advance().kind
                closer = {
                    "LPAREN": "RPAREN",
                    "LBRACK": "RBRACK",
                    "LBRACE": "RBRACE",
                }[opener]
                items = self.parse_comma_list(closer)
                left = Node("MACRO", name=self._callable_name(left), children=items)
            elif tok.kind == "LPAREN":
                self.advance()
                args = self.parse_comma_list("RPAREN")
                left = Node("CALL", children=[left, *args])
            elif tok.kind == "LBRACK":
                self.advance()
                idx = self.parse_comma_list("RBRACK")
                left = Node("INDEX", children=[left, *idx])
            elif tok.kind == "OP" and tok.value == ".":
                self.advance()
                nxt = self.peek()
                name = nxt.value
                self.advance()
                if self.peek().kind == "LPAREN":
                    self.advance()
                    args = self.parse_comma_list("RPAREN")
                    left = Node("METHOD", name=name, children=[left, *args])
                else:
                    left = Node("FIELD", name=name, children=[left])
            elif tok.kind == "OP" and tok.value == "@":
                self.advance()
                left = Node("VIEW", children=[left])
            elif tok.kind == "OP" and tok.value == "?":
                self.advance()
                left = Node("UNARY", op="?", children=[left])
            elif tok.kind == "OP" and tok.value == "::":
                self.advance()
                nxt = self.peek()
                if nxt.kind == "OP" and nxt.value == "<":
                    self.skip_generics()
                    # turbofish adds nothing structural on its own
                else:
                    name = nxt.value
                    self.advance()
                    left = Node("PATH", name=name, children=[left])
            else:
                break
        return left

    @staticmethod
    def _callable_name(node: Node) -> str:
        if node.kind in {"IDENT", "PATH"}:
            return node.name
        return "anonymous"

    def nud(self) -> Node:
        tok = self.peek()
        kind = tok.kind

        if kind in ("NUM", "STR", "CHAR", "LIFETIME"):
            self.advance()
            return Node("LITERAL", op=tok.value)

        if kind == "HASH":
            self.consume_attribute()
            if self.at_end() or self.peek().kind in ("RPAREN", "RBRACK", "RBRACE"):
                return Node("EMPTY")
            return self.nud()

        if kind == "OP" and tok.value in _PREFIX_CONNECTIVES:
            return self.parse_prefix_connective(tok.value)

        if kind == "OP" and tok.value in _PREFIX_OPS:
            self.advance()
            operand = self.parse_expr(_PREFIX_BP)
            return Node("UNARY", op=tok.value, children=[operand])

        if kind == "LPAREN":
            self.advance()
            items = self.parse_comma_list("RPAREN")
            if len(items) == 1:
                return Node("GROUP", children=items)
            return Node("TUPLE", children=items)

        if kind == "LBRACK":
            self.advance()
            items = self.parse_comma_list("RBRACK")
            return Node("ARRAY", children=items)

        if kind == "LBRACE":
            return self.parse_block()

        if kind == "PIPE":
            return self.parse_closure()

        if kind == "IDENT":
            val = tok.value
            if val in _QUANTIFIERS:
                self.advance()
                return self.parse_quant(val)
            if val == "if":
                self.advance()
                return self.parse_if()
            if val == "match":
                self.advance()
                return self.parse_match()
            if val == "let":
                self.advance()
                return self.parse_let()
            if val in ("true", "false"):
                self.advance()
                return Node("LITERAL", op=val)
            self.advance()
            return Node("IDENT", name=val)

        # Unknown token: consume to guarantee progress.
        self.advance()
        if kind == "EOF":
            return Node("EMPTY")
        self.parse_ok = False
        return Node("ATOM", op=tok.value)

    # -- compound forms ---------------------------------------------------- #
    def parse_prefix_connective(self, op: str) -> Node:
        """Parse Verus' bulleted ``&&& a &&& b`` / ``||| a ||| b`` list form.

        Builds the same left-leaning chain as the infix spelling so that both
        forms yield identical node counts and logical depth.
        """
        bp = _INFIX_BP[op]
        node: Optional[Node] = None
        guard = 0
        while self.peek().kind == "OP" and self.peek().value == op:
            guard += 1
            if guard > _MAX_TOKENS:
                self.parse_ok = False
                break
            self.advance()  # consume the leading connective
            # A run of connectives with no operand between them (`&&& &&& a`)
            # is malformed; swallow it here instead of recursing through nud(),
            # which would blow the stack on degenerate repeated-token output.
            while self.peek().kind == "OP" and self.peek().value == op:
                guard += 1
                if guard > _MAX_TOKENS:
                    break
                self.advance()
                self.parse_ok = False
            if self.at_end():
                # Dangling connective with no operand: incomplete input.
                self.parse_ok = False
                break
            operand = self.parse_expr(bp)
            if operand.kind == "EMPTY" and not operand.children:
                self.parse_ok = False
            node = operand if node is None else Node("LOGICAL", op=op, children=[node, operand])
        return node if node is not None else Node("EMPTY")

    def parse_comma_list(self, closer: str) -> List[Node]:
        items: List[Node] = []
        while not self.at_end() and self.peek().kind != closer:
            before = self.pos
            item = self.parse_expr(0.0)
            if not (item.kind == "EMPTY" and not item.children):
                items.append(item)
            if self.peek().kind == "COMMA":
                self.advance()
            elif self.peek().kind == "SEMI":
                self.advance()  # array repeat `[e; n]`
            elif self.peek().kind != closer:
                # not making sense — bail out of the list
                if self.pos == before:
                    self.advance()
                    self.parse_ok = False
                if self.peek().kind not in ("COMMA", closer):
                    break
        if not self.eat(closer):
            self.parse_ok = False
        return items

    def parse_quant(self, kw: str) -> Node:
        binders = self.parse_binders()
        body = self.parse_expr(0.0)
        return Node("QUANT", qkind=kw, binders=tuple(binders), children=[body])

    def parse_binders(self) -> List[str]:
        """Consume a ``|a: T, b: U|`` binder list, returning bound names."""
        names: List[str] = []
        if self.peek().kind != "PIPE":
            return names
        self.advance()  # opening |
        expect_name = True
        depth = 0
        guard = 0
        while not self.at_end():
            guard += 1
            if guard > _MAX_TOKENS:
                break
            tok = self.peek()
            if tok.kind == "PIPE" and depth == 0:
                self.advance()
                break
            if tok.kind in ("LPAREN", "LBRACK", "LBRACE"):
                depth += 1
            elif tok.kind in ("RPAREN", "RBRACK", "RBRACE"):
                depth = max(0, depth - 1)
            elif tok.kind == "OP" and tok.value == "<":
                depth += 1
            elif tok.kind == "OP" and tok.value == ">":
                depth = max(0, depth - 1)
            elif tok.kind == "COMMA" and depth == 0:
                expect_name = True
            elif tok.kind == "OP" and tok.value == ":" and depth == 0:
                expect_name = False
            elif tok.kind == "IDENT" and expect_name and depth == 0:
                names.append(tok.value)
                expect_name = False
            self.advance()
        return names

    def parse_closure(self) -> Node:
        self.parse_binders()
        body = self.parse_expr(0.0)
        return Node("CLOSURE", children=[body])

    def parse_if(self) -> Node:
        cond = self.parse_expr(0.0)
        then = self.parse_block() if self.peek().kind == "LBRACE" else self.parse_expr(0.0)
        children = [cond, then]
        if self.peek().kind == "IDENT" and self.peek().value == "else":
            self.advance()
            if self.peek().kind == "IDENT" and self.peek().value == "if":
                self.advance()
                children.append(self.parse_if())
            elif self.peek().kind == "LBRACE":
                children.append(self.parse_block())
            else:
                children.append(self.parse_expr(0.0))
        return Node("IF", children=children)

    def parse_match(self) -> Node:
        scrut = self.parse_expr(0.0)
        children = [scrut]
        if self.peek().kind == "LBRACE":
            self.advance()
            guard = 0
            while not self.at_end() and self.peek().kind != "RBRACE":
                guard += 1
                if guard > _MAX_TOKENS:
                    break
                self.skip_until_fat_arrow()
                arm = self.parse_expr(0.0)
                children.append(arm)
                self.eat("COMMA")
            self.eat("RBRACE")
        return Node("MATCH", children=children)

    def parse_let(self) -> Node:
        # `let pat = expr` (statement position). Skip the pattern, keep the rhs.
        guard = 0
        while not self.at_end():
            guard += 1
            if guard > _MAX_TOKENS:
                break
            tok = self.peek()
            if tok.kind == "OP" and tok.value == "=":
                self.advance()
                rhs = self.parse_expr(0.0)
                return Node("LET", children=[rhs])
            if tok.kind in ("SEMI", "RBRACE") or tok.kind == "EOF":
                break
            self.advance()
        return Node("LET")

    def parse_block(self) -> Node:
        self.eat("LBRACE")
        stmts: List[Node] = []
        guard = 0
        while not self.at_end() and self.peek().kind != "RBRACE":
            guard += 1
            if guard > _MAX_TOKENS:
                break
            before = self.pos
            stmt = self.parse_expr(0.0)
            if not (stmt.kind == "EMPTY" and not stmt.children):
                stmts.append(stmt)
            self.eat("SEMI")
            if self.pos == before:
                self.advance()  # progress guard
        self.eat("RBRACE")
        return Node("BLOCK", children=stmts)

    # -- low level skips --------------------------------------------------- #
    def consume_attribute(self) -> None:
        # `#` already at peek. Handles `#[...]` and `#![...]`.
        self.eat("HASH")
        if self.peek().kind == "OP" and self.peek().value == "!":
            self.advance()
        if self.peek().kind != "LBRACK":
            return
        self.advance()  # [
        depth = 1
        has_trigger = False
        guard = 0
        while not self.at_end() and depth > 0:
            guard += 1
            if guard > _MAX_TOKENS:
                break
            tok = self.advance()
            if tok.kind == "LBRACK":
                depth += 1
            elif tok.kind == "RBRACK":
                depth -= 1
            elif tok.kind == "IDENT" and tok.value in ("trigger", "auto"):
                has_trigger = True
        if has_trigger:
            self.triggers += 1

    def skip_generics(self) -> None:
        # peek at `<`. Balance `<` / `>` (best effort).
        if not (self.peek().kind == "OP" and self.peek().value == "<"):
            return
        self.advance()
        depth = 1
        guard = 0
        while not self.at_end() and depth > 0:
            guard += 1
            if guard > _MAX_TOKENS:
                break
            tok = self.advance()
            if tok.kind == "OP" and tok.value == "<":
                depth += 1
            elif tok.kind == "OP" and tok.value == ">":
                depth -= 1
            elif tok.kind == "OP" and tok.value == ">>":
                depth -= 2

    def skip_until_fat_arrow(self) -> None:
        depth = 0
        guard = 0
        while not self.at_end():
            guard += 1
            if guard > _MAX_TOKENS:
                break
            tok = self.peek()
            if depth == 0 and tok.kind == "OP" and tok.value == "=>":
                self.advance()
                return
            if tok.kind in ("LPAREN", "LBRACK", "LBRACE"):
                depth += 1
            elif tok.kind in ("RPAREN", "RBRACK", "RBRACE"):
                if depth == 0:
                    return
                depth -= 1
            self.advance()


def parse_expression(text: str) -> Tuple[Node, int, bool]:
    """Parse a spec expression. Returns ``(root, trigger_count, parse_ok)``."""
    cleaned = strip_comments(text)
    toks = lex(cleaned)
    parser = Parser(toks)
    root = parser.parse()
    return root, parser.triggers, parser.parse_ok


# --------------------------------------------------------------------------- #
# Complexity metrics over the AST
# --------------------------------------------------------------------------- #


@dataclass
class NodeStats:
    nodes: int = 0
    logic_depth: int = 0
    n_quant: int = 0
    alt_depth: int = 0
    operators: Dict[str, int] = field(default_factory=dict)
    variables: Dict[str, int] = field(default_factory=dict)


# All AST statistics below use explicit work stacks instead of recursion: a
# left-leaning connective chain (`a && a && ...`) or a bulleted `&&&` list can
# legally be thousands of levels deep, which would overflow Python's call
# stack long before hitting _MAX_TOKENS.


def _count_nodes(node: Node) -> int:
    total = 0
    stack = [node]
    while stack:
        current = stack.pop()
        if current.kind not in _TRANSPARENT:
            total += 1
        stack.extend(current.children)
    return total


_IF_CHAIN_CTX = "if-chain"


def _logic_depth(node: Node, ctx: Optional[str] = None) -> int:
    """Depth of the boolean/quantifier skeleton.

    A run of the *same* associative connective (``a && b && c``) counts as a
    single level; a level is added only when the connective changes, or when
    entering a quantifier body / ``if`` / ``match`` branch.  Arithmetic, calls
    and indexing never add logical depth.

    An ``else if`` chain is flat n-way dispatch, not nesting, so it folds into a
    single level the same way a connective run does. That keeps it comparable
    with the equivalent ``match``, which is always one level.
    """
    best = 0
    # (node, connective context, depth accumulated on the path above)
    stack: List[Tuple[Node, Optional[str], int]] = [(node, ctx, 0)]
    while stack:
        current, current_ctx, acc = stack.pop()
        kind = current.kind
        if kind == "GROUP":  # transparent: parentheses forward the connective ctx
            for child in current.children:
                stack.append((child, current_ctx, acc))
            continue
        if kind == "LOGICAL":
            acc += 0 if current.op == current_ctx else 1
            best = max(best, acc)
            for child in current.children:
                stack.append((child, current.op, acc))
            continue
        if kind == "IF":
            acc += 0 if current_ctx == _IF_CHAIN_CTX else 1
            best = max(best, acc)
            # children = [cond, then, else?]; only the else branch continues the chain.
            for index, child in enumerate(current.children):
                stack.append((child, _IF_CHAIN_CTX if index == 2 else None, acc))
            continue
        if kind in ("QUANT", "MATCH"):
            acc += 1
            best = max(best, acc)
            for child in current.children:
                stack.append((child, None, acc))
            continue
        if kind == "UNARY" and current.op == "!":
            for child in current.children:
                stack.append((child, current_ctx, acc))
            continue
        for child in current.children:
            stack.append((child, None, acc))
    return best


def _count_quant(node: Node) -> int:
    total = 0
    stack = [node]
    while stack:
        current = stack.pop()
        if current.kind == "QUANT":
            total += 1
        stack.extend(current.children)
    return total


def _alt_depth(node: Node, last_kind: Optional[str], blocks: int) -> int:
    best = blocks
    stack: List[Tuple[Node, Optional[str], int]] = [(node, last_kind, blocks)]
    while stack:
        current, last, acc = stack.pop()
        if current.kind == "QUANT":
            this = "A" if current.qkind == "forall" else "E"
            acc += 1 if (last is None or this != last) else 0
            last = this
            best = max(best, acc)
        for child in current.children:
            stack.append((child, last, acc))
    return best


def _op_symbol(node: Node) -> Optional[str]:
    kind = node.kind
    if kind in ("BINARY", "LOGICAL"):
        return node.op
    if kind == "UNARY":
        return "u" + node.op
    if kind == "CALL":
        return "call()"
    if kind == "METHOD":
        return "method." + node.name
    if kind == "FIELD":
        return "field." + node.name
    if kind == "INDEX":
        return "index[]"
    if kind == "VIEW":
        return "@"
    if kind == "PATH":
        return "::"
    if kind == "MACRO":
        return "macro." + node.name + "!"
    if kind == "QUANT":
        return node.qkind
    if kind in ("IF", "MATCH", "CLOSURE", "LET"):
        return kind.lower()
    return None


def _collect_vocab(
    node: Node,
    operators: Dict[str, int],
    variables: Dict[str, int],
    *,
    identifier_role: str = "value",
) -> None:
    stack: List[Tuple[Node, str]] = [(node, identifier_role)]
    while stack:
        current, role = stack.pop()
        sym = _op_symbol(current)
        if sym is not None:
            operators[sym] = operators.get(sym, 0) + 1
        if current.kind == "IDENT" and role == "value":
            variables[current.name] = variables.get(current.name, 0) + 1

        for index, child in enumerate(current.children):
            # Type position is inherited: `x as (&int)` must not surface `int`
            # as a program variable even through GROUP/UNARY wrappers.
            child_role = "type" if role == "type" else "value"
            if current.kind in {"CALL", "PATH"} and index == 0:
                child_role = "callee"
            elif current.kind == "BINARY" and current.op == "as" and index == 1:
                # `x as int`: the right operand is a type, not a program variable.
                child_role = "type"
            stack.append((child, child_role))


def analyze_expression(text: str) -> Tuple[NodeStats, int, bool]:
    """Full structural analysis of a single spec expression."""
    try:
        root, triggers, parse_ok = parse_expression(text)
        operators: Dict[str, int] = {}
        variables: Dict[str, int] = {}
        _collect_vocab(root, operators, variables)
        stats = NodeStats(
            nodes=_count_nodes(root),
            logic_depth=_logic_depth(root),
            n_quant=_count_quant(root),
            alt_depth=_alt_depth(root, None, 0),
            operators=operators,
            variables=variables,
        )
    except RecursionError:
        # The parse phase is still recursive for right-associative chains and
        # deeply nested parentheses. Degrade pathological inputs to an ordinary
        # parse failure instead of breaking the parser's totality promise.
        return NodeStats(), 0, False
    return stats, triggers, parse_ok
