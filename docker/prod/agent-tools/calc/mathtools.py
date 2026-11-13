"""Symbolic maths and physical units for <feed> coursework (sympy, pint).

calculate (calc.py) gives numbers; first-year Analysis, Linear Algebra and
Physics need derivatives, integrals, limits, equations, matrices, unit
conversions and physical constants, which gemma4-e2b gets wrong by hand.

Both functions take untrusted text from a group chat:
* symbolic_math never hands text to sympify/parse_expr (they eval). The
  text goes through sympy's tokenizer transformations (stringify_expr) to
  Python source, which is checked against a whitelist (_Checker) and only
  then evaluated with no builtins.
* physics_units uses pint's own expression parser, which builds a tree of
  numbers and units and never calls eval.
Either can still be asked for something huge (10^10^10, a hard integral):
server.py runs every call in a killable child process with a time limit.

Each answer is one or a few lines that restate the input, like calc.py's.
"""

from __future__ import annotations

import ast
import re

import pint
import sympy
from sympy.parsing.sympy_parser import (
    implicit_multiplication_application,
    standard_transformations,
    stringify_expr,
)


class MathError(ValueError):
    pass


# --- symbolic_math -------------------------------------------------------------

OPERATIONS = (
    "evaluate", "simplify", "expand", "factor", "derivative", "integral",
    "limit", "solve", "series", "determinant", "inverse", "eigenvalues",
    "eigenvectors", "rank", "rref",
)
MATRIX_OPERATIONS = {"determinant", "inverse", "eigenvalues", "eigenvectors", "rank", "rref"}
MAX_TEXT = 500
MAX_ORDER = 20
MAX_MATRIX = 6

_FUNCS = {
    name: getattr(sympy, name) for name in (
        "sin", "cos", "tan", "cot", "sec", "csc", "asin", "acos", "atan", "acot",
        "atan2", "sinh", "cosh", "tanh", "coth", "asinh", "acosh", "atanh",
        "exp", "log", "sqrt", "cbrt", "root", "Abs", "sign", "floor", "ceiling",
        "factorial", "binomial", "gamma", "erf", "re", "im", "conjugate", "arg",
        "Max", "Min", "gcd", "lcm", "Heaviside",
    )
}
_FUNCS.update({"ln": sympy.log, "abs": sympy.Abs, "arcsin": sympy.asin, "arccos": sympy.acos,
               "arctan": sympy.atan, "max": sympy.Max, "min": sympy.Min, "ceil": sympy.ceiling,
               "Matrix": sympy.Matrix})
_CONSTS = {"pi": sympy.pi, "e": sympy.E, "E": sympy.E, "i": sympy.I, "I": sympy.I,
           "oo": sympy.oo, "inf": sympy.oo, "infinity": sympy.oo}
# What stringify_expr's own output calls: numbers and symbols, by name.
_ATOMS = {"Symbol": sympy.Symbol, "Integer": sympy.Integer, "Float": sympy.Float,
          "Rational": sympy.Rational}
_NAMESPACE = {**_FUNCS, **_CONSTS, **_ATOMS}
# Names the tokenizer must keep whole (not split "pi" into p*i or "sin" into s*i*n).
_GLOBAL = {**_NAMESPACE}
_TRANSFORMS = standard_transformations + (implicit_multiplication_application,)

_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.FloorDiv, ast.MatMult)
_SYMBOL_NAME = re.compile(r"[^\W\d]\w{0,19}")


class _Checker(ast.NodeTransformer):
    """Lets through arithmetic on numbers, symbols, whitelisted functions and
    matrices only; turns an outermost [[...]] list into Matrix([[...]])."""

    def __init__(self) -> None:
        self.list_depth = 0

    def generic_visit(self, node):
        raise MathError(f"not allowed in an expression: {type(node).__name__}")

    def visit_Expression(self, node):
        node.body = self.visit(node.body)
        return node

    def visit_BinOp(self, node):
        if not isinstance(node.op, _BINOPS):
            raise MathError("only + - * / ^ are allowed")
        node.left, node.right = self.visit(node.left), self.visit(node.right)
        return node

    def visit_UnaryOp(self, node):
        if not isinstance(node.op, (ast.UAdd, ast.USub)):
            raise MathError("only + - * / ^ are allowed")
        node.operand = self.visit(node.operand)
        return node

    def visit_Constant(self, node):
        if type(node.value) not in (int, float):
            raise MathError("text is not allowed in an expression")
        return node

    def visit_Name(self, node):
        if node.id not in _NAMESPACE:
            raise MathError(f"unknown name {node.id!r}")
        return node

    def visit_Call(self, node):
        if not isinstance(node.func, ast.Name) or node.keywords:
            raise MathError("only plain function calls are allowed")
        name = node.func.id
        if name in _ATOMS:
            # Symbol('x'), Integer(2), Float('1.5'): one literal argument.
            if len(node.args) != 1 or not isinstance(node.args[0], ast.Constant):
                raise MathError(f"bad {name}")
            value = node.args[0].value
            if name == "Symbol" and not (isinstance(value, str) and _SYMBOL_NAME.fullmatch(value)):
                raise MathError("bad variable name")
            if name != "Symbol" and not isinstance(value, (int, str)):
                raise MathError("bad number")
            if isinstance(value, str) and name != "Symbol" and not re.fullmatch(r"[0-9.eE+-]{1,60}", value):
                raise MathError("bad number")
            return node
        if name not in _FUNCS:
            raise MathError(f"unknown function {name!r}")
        node.args = [self.visit(arg) for arg in node.args]
        return node

    def visit_List(self, node):
        self.list_depth += 1
        try:
            node.elts = [self.visit(elt) for elt in node.elts]
        finally:
            self.list_depth -= 1
        if self.list_depth == 0:
            return ast.Call(func=ast.Name(id="Matrix", ctx=ast.Load()), args=[node], keywords=[])
        return node

    def visit_Tuple(self, node):
        node.elts = [self.visit(elt) for elt in node.elts]
        return node


def _prepare(text: str) -> str:
    text = text.strip()
    for old, new in (("×", "*"), ("·", "*"), ("÷", "/"), ("−", "-"), ("–", "-"),
                     ("π", "pi"), ("∞", "oo"), ("√", "sqrt"), ("²", "^2"), ("³", "^3"),
                     ("^", "**")):
        text = text.replace(old, new)
    # [1, 2; 3, 4] (rows split by ;) -> [[1, 2], [3, 4]]
    match = re.fullmatch(r"\[([^\[\]]*;[^\[\]]*)\]", text)
    if match:
        text = "[" + ", ".join(f"[{row}]" for row in match.group(1).split(";")) + "]"
    return text


def parse(text: str):
    """Text -> sympy object, never through eval of the raw text."""
    if len(text) > MAX_TEXT:
        raise MathError(f"expression longer than {MAX_TEXT} characters")
    if "__" in text or "'" in text or '"' in text or "\\" in text:
        raise MathError("text is not allowed in an expression")
    prepared = _prepare(text)
    if not prepared:
        raise MathError("empty expression")
    try:
        code = stringify_expr(prepared, {}, dict(_GLOBAL), _TRANSFORMS)
        tree = _Checker().visit(ast.parse(code, mode="eval"))
        ast.fix_missing_locations(tree)
        result = eval(compile(tree, "<math>", "eval"), {"__builtins__": {}}, dict(_NAMESPACE))  # noqa: S307 (checked tree)
    except MathError:
        raise
    except (SyntaxError, TypeError, ValueError, AttributeError, sympy.SympifyError, ZeroDivisionError) as exc:
        raise MathError(f"could not read {text!r}: {type(exc).__name__}") from exc
    if isinstance(result, (list, tuple)):
        raise MathError("write a matrix as [[1, 2], [3, 4]]")
    return sympy.sympify(result) if isinstance(result, (int, float)) else result


def _equation(text: str):
    """'lhs = rhs' -> lhs - rhs; no '=' -> the expression itself (= 0)."""
    parts = re.split(r"(?<![<>=!])=(?!=)", text)
    if len(parts) > 2:
        raise MathError("one '=' per equation; separate equations with ;")
    if len(parts) == 2:
        return parse(parts[0]) - parse(parts[1]), True
    return parse(text), False


def show(value) -> str:
    """sympy's plain text, as a student writes it: ^ powers, e, i, ∞."""
    if isinstance(value, sympy.MatrixBase):
        return "[" + ", ".join("[" + ", ".join(show(v) for v in value.row(r)) + "]" for r in range(value.rows)) + "]"
    text = sympy.sstr(value)
    text = text.replace("**", "^")
    text = re.sub(r"\bE\b", "e", text)
    text = re.sub(r"\bI\b", "i", text)
    text = text.replace("zoo", "complex ∞").replace("oo", "∞")
    return text


def _approx(value) -> str:
    """' ≈ 1.41421' for an exact number that isn't plainly a whole number."""
    try:
        if isinstance(value, sympy.Basic) and value.is_number and not value.is_Integer \
                and not value.has(sympy.oo, -sympy.oo, sympy.zoo, sympy.nan):
            number = sympy.N(value, 8)
            if number.is_real or number.is_complex:
                return f" ≈ {show(number)}"
    except (TypeError, ValueError):
        pass
    return ""


def _variables(expr, variable: str) -> list:
    if variable.strip():
        names = [v.strip() for v in re.split(r"[,\s]+", variable) if v.strip()]
        out = []
        for name in names:
            if not _SYMBOL_NAME.fullmatch(name) or name in _NAMESPACE:
                raise MathError(f"bad variable {name!r}")
            out.append(sympy.Symbol(name))
        return out
    free = sorted(getattr(expr, "free_symbols", set()), key=lambda s: s.name)
    return free


def _one_variable(expr, variable: str):
    found = _variables(expr, variable)
    if len(found) == 1:
        return found[0]
    if not found:
        return sympy.Symbol(variable.strip() or "x")
    for preferred in ("x", "t"):
        for sym in found:
            if sym.name == preferred:
                return sym
    raise MathError(f"several variables ({', '.join(s.name for s in found)}): say which in variable")


def _point(text: str):
    text = text.strip().replace("∞", "oo")
    direction = "+-"
    if len(text) > 1 and text[-1] in "+-" and text[:-1].strip():
        direction, text = text[-1], text[:-1]
    return parse(text), direction


def _matrix(text: str):
    value = parse(text)
    if not isinstance(value, sympy.MatrixBase):
        raise MathError("give a matrix, e.g. [[1, 2], [3, 4]]")
    if value.rows > MAX_MATRIX or value.cols > MAX_MATRIX:
        raise MathError(f"matrices up to {MAX_MATRIX}x{MAX_MATRIX}")
    return value


def symbolic_math(operation: str, expression: str, variable: str = "", at: str = "", order: int = 0) -> str:
    operation = operation.strip().lower()
    if operation not in OPERATIONS:
        raise MathError(f"operation is one of: {', '.join(OPERATIONS)}")
    if not 0 <= order <= MAX_ORDER:
        raise MathError(f"order is 0-{MAX_ORDER}")
    expression = expression.strip()

    if operation in MATRIX_OPERATIONS:
        m = _matrix(expression)
        if operation == "rank":
            return f"rank {show(m)} = {m.rank()}"
        if operation == "rref":
            reduced, pivots = m.rref()
            return f"rref {show(m)} = {show(reduced)} (pivot columns {', '.join(str(p + 1) for p in pivots)})"
        if not m.is_square:
            raise MathError(f"{operation} needs a square matrix")
        if operation == "determinant":
            d = sympy.simplify(m.det())
            return f"det {show(m)} = {show(d)}{_approx(d)}"
        if operation == "inverse":
            d = sympy.simplify(m.det())
            if d == 0:
                return f"{show(m)} has no inverse: its determinant is 0"
            return f"inverse of {show(m)} = {show(sympy.simplify(m.inv()))}"
        if operation == "eigenvalues":
            values = m.eigenvals()
            parts = [f"{show(v)}{_approx(v)}" + (f" (multiplicity {k})" if k > 1 else "") for v, k in values.items()]
            return f"eigenvalues of {show(m)}: " + "; ".join(parts)
        lines = [f"eigenvectors of {show(m)}:"]
        for value, mult, vectors in m.eigenvects():
            vecs = ", ".join(show(v.T) for v in vectors)
            lines.append(f"λ = {show(value)}{_approx(value)} (multiplicity {mult}): {vecs}")
        return "\n".join(lines)

    if operation == "solve":
        texts = [t for t in re.split(r";|\n", expression) if t.strip()]
        if not texts:
            raise MathError("empty equation")
        equations = [_equation(t)[0] for t in texts]
        if variable.strip():
            unknowns = _variables(None, variable)
        else:
            unknowns = sorted(set().union(*(e.free_symbols for e in equations)), key=lambda s: s.name)
        if not unknowns:
            raise MathError("no unknown to solve for")
        solutions = sympy.solve(equations, unknowns, dict=True)
        names = ", ".join(s.name for s in unknowns)
        if not solutions:
            return f"{'; '.join(texts)}: no solution for {names} found (none exists, or none in closed form)"
        rendered = []
        for sol in solutions:
            rendered.append(", ".join(f"{show(k)} = {show(v)}{_approx(v)}" for k, v in sol.items()))
        return f"{'; '.join(t.strip() for t in texts)} → " + " | ".join(rendered)

    expr, is_equation = _equation(expression) if "=" in expression else (parse(expression), False)
    if is_equation:
        raise MathError("an equation needs operation solve")

    if operation == "evaluate":
        value = expr if isinstance(expr, sympy.MatrixBase) else sympy.simplify(expr)
        return f"{expression} = {show(value)}{_approx(value)}"
    if isinstance(expr, sympy.MatrixBase):
        raise MathError(f"{operation} works on an expression, not a matrix")
    if operation == "simplify":
        return f"{expression} = {show(sympy.simplify(expr))}"
    if operation == "expand":
        return f"{expression} = {show(sympy.expand(expr))}"
    if operation == "factor":
        return f"{expression} = {show(sympy.factor(expr))}"

    var = _one_variable(expr, variable)
    if operation == "derivative":
        n = order or 1
        result = sympy.diff(expr, var, n)
        simpler = sympy.simplify(result)
        mark = f"d^{n}/d{var}^{n}" if n > 1 else f"d/d{var}"
        also = f" = {show(simpler)}" if simpler != result and sympy.count_ops(simpler) < sympy.count_ops(result) else ""
        return f"{mark} [{expression}] = {show(result)}{also}"
    if operation == "integral":
        if at.strip():
            bounds = [b for b in at.split(",") if b.strip()]
            if len(bounds) != 2:
                raise MathError("at for a definite integral is 'a, b'")
            a, b = parse(bounds[0]), parse(bounds[1])
            result = sympy.integrate(expr, (var, a, b))
            if result.has(sympy.Integral):
                numeric = sympy.N(sympy.Integral(expr, (var, a, b)), 10)
                return f"∫ from {show(a)} to {show(b)} of {expression} d{var}: no closed form found ≈ {show(numeric)}"
            return f"∫ from {show(a)} to {show(b)} of {expression} d{var} = {show(sympy.simplify(result))}{_approx(result)}"
        result = sympy.integrate(expr, var)
        if result.has(sympy.Integral):
            return f"∫ {expression} d{var}: no closed form found"
        return f"∫ {expression} d{var} = {show(sympy.simplify(result))} + C"
    if operation == "limit":
        if not at.strip():
            raise MathError("at is the point, e.g. 0, oo or 0+")
        point, direction = _point(at)
        result = sympy.limit(expr, var, point, direction)
        side = {"+": " from above", "-": " from below"}.get(direction, "")
        if isinstance(result, sympy.AccumBounds):
            return f"lim {var}→{show(point)}{side} of {expression}: no limit (oscillates in {show(result)})"
        return f"lim {var}→{show(point)}{side} of {expression} = {show(result)}{_approx(result)}"
    # series
    point, _ = _point(at) if at.strip() else (sympy.Integer(0), "+-")
    n = order or 6
    result = sympy.series(expr, var, point, n)
    return f"{expression} around {var} = {show(point)}: {show(result)}"


# --- physics_units -------------------------------------------------------------

# Built once at import (~1 s on the Pi): server.py's forkserver imports this
# module before it forks a child per call, so no call pays for it.
_ureg = pint.UnitRegistry(autoconvert_offset_to_baseunit=True)


def _prepare_units(text: str) -> str:
    text = text.strip()
    for old, new in (("°C", "degC"), ("°F", "degF"), ("℃", "degC"), ("℉", "degF"), ("°", " degree"),
                     ("µ", "u"), ("μ", "u"), ("Ω", "ohm"), ("×", "*"), ("·", "*"), ("−", "-"),
                     ("²", "**2"), ("³", "**3"), ("^", "**")):
        text = text.replace(old, new)
    return text


def _units_text(q) -> str:
    text = f"{q:.6g~P}"
    return text


def physics_units(quantity: str, to: str = "") -> str:
    if len(quantity) > MAX_TEXT or len(to) > 100:
        raise MathError("too long")
    if not quantity.strip():
        raise MathError("give a quantity, e.g. 72 km/h or speed_of_light")
    if any(bad in quantity + to for bad in ("__", "'", '"', "\\", "[", "]")):
        raise MathError("give a number with units, e.g. 72 km/h")
    ureg = _ureg
    try:
        q = ureg.parse_expression(_prepare_units(quantity))
        if not isinstance(q, ureg.Quantity):
            q = ureg.Quantity(q)
        if to.strip():
            target = to.strip()
            result = q.to(_prepare_units(target))
        else:
            result = q.to_base_units()
    except pint.DimensionalityError as exc:
        raise MathError(f"can't convert: {exc.dim1 or 'dimensionless'} is not {exc.dim2 or 'dimensionless'}") from exc
    except pint.UndefinedUnitError as exc:
        raise MathError(f"unknown unit or constant {exc.unit_names}") from exc
    except (pint.PintError, SyntaxError, TypeError, ValueError, AttributeError, KeyError, ZeroDivisionError,
            AssertionError, IndexError) as exc:
        raise MathError(f"could not read {quantity!r}: {type(exc).__name__}") from exc
    return f"{quantity.strip()} = {_units_text(result)}"
