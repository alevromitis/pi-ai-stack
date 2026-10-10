"""Exact answers for what gemma4-e2b gets wrong: arithmetic, dates, counting letters.

Pure functions, no I/O; server.py exposes them as MCP tools. Each returns a
one-line answer that restates its input, so the model can't misattribute it.
"""

from __future__ import annotations

import ast
import calendar
import datetime
import math
import operator
import re
from collections import Counter

# --- calculate --------------------------------------------------------------

FUNCS = {
    "sqrt": math.sqrt, "cbrt": math.cbrt, "abs": abs, "round": round,
    "floor": math.floor, "ceil": math.ceil, "exp": math.exp,
    "log": math.log, "ln": math.log, "log10": math.log10, "log2": math.log2,
    "sin": math.sin, "cos": math.cos, "tan": math.tan, "asin": math.asin,
    "acos": math.acos, "atan": math.atan, "degrees": math.degrees,
    "radians": math.radians, "factorial": math.factorial, "gcd": math.gcd,
    "lcm": math.lcm, "comb": math.comb, "perm": math.perm, "hypot": math.hypot,
    "min": min, "max": max,
}
CONSTS = {"pi": math.pi, "e": math.e, "tau": math.tau}
BINOPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod, ast.Pow: operator.pow,
}
UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
MAX_EXPONENT = 10_000
MAX_FACTORIAL = 3_000
MAX_INT_BITS = 100_000


class CalcError(ValueError):
    pass


def _prepare(expression: str) -> str:
    text = expression.strip().rstrip("=?").strip()
    text = text.replace("×", "*").replace("·", "*").replace("÷", "/").replace("−", "-")
    text = text.replace("^", "**")
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)          # 1,000,000
    text = re.sub(r"(?<=[\d)])\s*[xX]\s*(?=[\d(])", "*", text)  # 3 x 4
    text = re.sub(r"(\d+(?:\.\d+)?)\s*%\s*of\b", r"(\1/100)*", text)  # 15% of 80
    text = re.sub(r"(\d+(?:\.\d+)?)\s*%", r"(\1/100)", text)
    return text


def _eval(node: ast.AST) -> int | float:
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.Name) and node.id in CONSTS:
        return CONSTS[node.id]
    if isinstance(node, ast.UnaryOp) and type(node.op) in UNARY:
        return UNARY[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in BINOPS:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_EXPONENT:
            raise CalcError("exponent too large")
        result = BINOPS[type(node.op)](left, right)
        if isinstance(result, int) and result.bit_length() > MAX_INT_BITS:
            raise CalcError("result too large")
        return result
    if (
        isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        and node.func.id in FUNCS and not node.keywords
    ):
        args = [_eval(a) for a in node.args]
        if node.func.id == "factorial" and args and args[0] > MAX_FACTORIAL:
            raise CalcError("factorial too large")
        return FUNCS[node.func.id](*args)
    raise CalcError(f"unsupported: {ast.unparse(node)}")


def _format(value: int | float | complex) -> str:
    if isinstance(value, bool):
        value = int(value)
    if isinstance(value, int):
        text = str(value)
        return text if len(text) <= 60 else f"{value:.15e} ({len(text)} digits)"
    if isinstance(value, float):
        if value.is_integer() and abs(value) < 1e15:
            return str(int(value))
        if 0 < abs(value) < 1e-12:
            return f"0 (rounding error: {value:.3g})"
        return f"{value:.12g}"
    raise CalcError("not a real number")


def calculate(expression: str) -> str:
    """'17*23 = 391'. Raises CalcError for anything but arithmetic."""
    text = _prepare(expression)
    if not text:
        raise CalcError("empty expression")
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        raise CalcError("not an arithmetic expression") from exc
    try:
        value = _eval(tree)
    except CalcError:
        raise
    except ZeroDivisionError as exc:
        raise CalcError("division by zero") from exc
    except (OverflowError, ValueError, TypeError) as exc:
        raise CalcError(str(exc) or type(exc).__name__) from exc
    return f"{expression.strip().rstrip('=?').strip()} = {_format(value)}"


# --- dates -----------------------------------------------------------------

FORMATS = (
    "%Y-%m-%d", "%Y/%m/%d", "%d %B %Y", "%d %b %Y", "%B %d %Y", "%b %d %Y",
    "%B %d, %Y", "%b %d, %Y", "%d %B, %Y",
)


def parse_date(text: str, today: datetime.date) -> datetime.date:
    clean = " ".join(re.sub(r"(?<=\d)(st|nd|rd|th)\b", "", text.strip()).split())
    lowered = clean.lower()
    if lowered in ("today", "now", ""):
        return today
    if lowered == "tomorrow":
        return today + datetime.timedelta(days=1)
    if lowered == "yesterday":
        return today - datetime.timedelta(days=1)
    for fmt in FORMATS:
        try:
            return datetime.datetime.strptime(clean, fmt).date()
        except ValueError:
            continue
    raise CalcError(f"can't read the date {text!r}: use YYYY-MM-DD")


def _describe(day: datetime.date) -> str:
    return f"{day.isoformat()} ({day.strftime('%A')})"


def _add_months(day: datetime.date, months: int) -> datetime.date:
    month0 = day.month - 1 + months
    year, month = day.year + month0 // 12, month0 % 12 + 1
    if not 1 <= year <= 9999:
        raise CalcError("date out of range")
    return day.replace(year=year, month=month, day=min(day.day, calendar.monthrange(year, month)[1]))


def _ymd(start: datetime.date, end: datetime.date) -> tuple[int, int, int]:
    """Whole years, months and days from start to end (start <= end)."""
    months = (end.year - start.year) * 12 + end.month - start.month
    if _add_months(start, months) > end:
        months -= 1
    days = (end - _add_months(start, months)).days
    return months // 12, months % 12, days


def _plural(n: int, unit: str) -> str:
    return f"{n} {unit}{'' if n == 1 else 's'}"


def date_diff(start: str, end: str, today: datetime.date) -> str:
    a, b = parse_date(start, today), parse_date(end, today)
    days = (b - a).days
    lo, hi = sorted((a, b))
    years, months, rest = _ymd(lo, hi)
    parts = [_plural(n, u) for n, u in ((years, "year"), (months, "month"), (rest, "day")) if n]
    weeks, wdays = divmod(abs(days), 7)
    sign = "-" if days < 0 else ""
    return (
        f"From {_describe(a)} to {_describe(b)}: {sign}{_plural(abs(days), 'day')}"
        f" ({_plural(weeks, 'week')} {_plural(wdays, 'day')}"
        + (f"; {', '.join(parts)}" if parts else "") + ")."
        + (" The end is before the start." if days < 0 else "")
    )


def date_add(date: str, days: int, weeks: int, months: int, years: int, today: datetime.date) -> str:
    start = parse_date(date, today)
    try:
        result = _add_months(start, 12 * years + months) + datetime.timedelta(days=days, weeks=weeks)
    except OverflowError as exc:
        raise CalcError("date out of range") from exc
    moved = " ".join(
        f"{'+' if n > 0 else '-'} {_plural(abs(n), u)}"
        for n, u in ((years, "year"), (months, "month"), (weeks, "week"), (days, "day")) if n
    )
    return f"{_describe(start)} {moved or '+ 0 days'} = {_describe(result)}."


# --- counting ----------------------------------------------------------------


def count_text(text: str, find: str = "") -> str:
    letters = sum(c.isalpha() for c in text)
    words = len(re.findall(r"[^\W_]+(?:['’-][^\W_]+)*", text))
    out = f"{text!r}: {_plural(len(text), 'character')}, {_plural(letters, 'letter')}, {_plural(words, 'word')}."
    if find:
        n = text.lower().count(find.lower())
        where = ""
        if len(find) == 1 and 0 < n <= 20:
            spots = [str(i + 1) for i, c in enumerate(text.lower()) if c == find.lower()]
            where = f" (at positions {', '.join(spots)})"
        out += f" {find!r} appears {_plural(n, 'time')}{where}, ignoring case."
    elif 0 < letters <= 60:
        # The model often leaves out `find`, then counts the letters itself.
        tally = Counter(c for c in text.lower() if c.isalpha())
        out += " Each letter, ignoring case: " + ", ".join(f"{c}={n}" for c, n in tally.items()) + "."
    return out
