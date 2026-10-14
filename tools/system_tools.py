"""System-level tools: clock, safe calculator and read-only OS information."""

from __future__ import annotations

import ast
import logging
import math
import operator
import os
import platform
import shutil
from collections.abc import Callable
from datetime import datetime
from typing import Any

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Safe arithmetic evaluator (AST whitelist - never uses eval/exec)
# --------------------------------------------------------------------------- #
MAX_EXPRESSION_LENGTH: int = 200
MAX_EXPONENT: int = 1000
MAX_RESULT_BITS: int = 20_000

Number = int | float


def _safe_pow(base: Number, exponent: Number) -> Number:
    """``base ** exponent`` with guards against runaway computations."""
    if abs(exponent) > MAX_EXPONENT:
        raise ValueError(f"exponent too large (max {MAX_EXPONENT})")
    if isinstance(base, int) and isinstance(exponent, int) and exponent > 0:
        if base.bit_length() * exponent > MAX_RESULT_BITS:
            raise ValueError("result would be too large")
    return base**exponent


def _safe_factorial(value: Number) -> int:
    if not float(value).is_integer() or value < 0 or value > 500:
        raise ValueError("factorial requires an integer between 0 and 500")
    return math.factorial(int(value))


_BINARY_OPS: dict[type[ast.operator], Callable[[Any, Any], Number]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: _safe_pow,
}
_UNARY_OPS: dict[type[ast.unaryop], Callable[[Any], Number]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
_FUNCTIONS: dict[str, Callable[..., Number]] = {
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "log": math.log,
    "log10": math.log10,
    "log2": math.log2,
    "exp": math.exp,
    "abs": abs,
    "round": round,
    "floor": math.floor,
    "ceil": math.ceil,
    "factorial": _safe_factorial,
}
_CONSTANTS: dict[str, float] = {"pi": math.pi, "e": math.e, "tau": math.tau}


def _eval_node(node: ast.AST) -> Number:
    """Recursively evaluate a whitelisted AST node."""
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        raise ValueError(f"unsupported literal: {node.value!r}")
    if isinstance(node, ast.BinOp):
        op = _BINARY_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"unsupported operator: {type(node.op).__name__}")
        return op(_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp):
        unary = _UNARY_OPS.get(type(node.op))
        if unary is None:
            raise ValueError(f"unsupported operator: {type(node.op).__name__}")
        return unary(_eval_node(node.operand))
    if isinstance(node, ast.Name):
        if node.id in _CONSTANTS:
            return _CONSTANTS[node.id]
        raise ValueError(f"unknown name: {node.id!r}")
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCTIONS:
            raise ValueError("unsupported function call")
        if node.keywords:
            raise ValueError("keyword arguments are not supported")
        return _FUNCTIONS[node.func.id](*[_eval_node(arg) for arg in node.args])
    raise ValueError(f"unsupported expression element: {type(node).__name__}")


def safe_eval_arithmetic(expression: str) -> Number:
    """Evaluate an arithmetic expression safely.

    Supports ``+ - * / // % **`` (``^`` is treated as power), parentheses,
    the constants ``pi``/``e``/``tau`` and common math functions.

    Raises:
        ValueError: For empty, too long, malformed or unsupported expressions.
        ZeroDivisionError, OverflowError: For arithmetic failures.
    """
    cleaned = (
        expression.strip()
        .replace("^", "**")
        .replace("×", "*")
        .replace("÷", "/")
    )
    if not cleaned:
        raise ValueError("expression is empty")
    if len(cleaned) > MAX_EXPRESSION_LENGTH:
        raise ValueError(f"expression too long (max {MAX_EXPRESSION_LENGTH} characters)")
    try:
        tree = ast.parse(cleaned, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"invalid syntax: {exc.msg}") from exc
    result = _eval_node(tree)
    if isinstance(result, complex):
        raise ValueError("result is not a real number")
    if isinstance(result, float) and not math.isfinite(result):
        raise ValueError("result is not a finite number")
    return result


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #
@tool
def get_current_time() -> str:
    """Get the current local date, time and timezone. Use for any question about 'now', 'today' or the current time."""
    try:
        now = datetime.now().astimezone()
        return now.strftime("%A, %d %B %Y, %I:%M:%S %p %Z (UTC%z)")
    except Exception as exc:
        logger.exception("get_current_time failed")
        return f"Error: could not read the clock ({type(exc).__name__}: {exc})"


@tool
def run_calculator(expression: str) -> str:
    """Evaluate an arithmetic expression exactly, e.g. '(12.5 * 4) / 3' or 'sqrt(144) + 2**10'.
    Supports + - * / // % ** ( ), pi, e and sqrt, sin, cos, tan, log, log10, exp, abs, round, floor, ceil, factorial.
    Always use this for math instead of calculating in your head."""
    try:
        result = safe_eval_arithmetic(expression)
        formatted = f"{result:.12g}" if isinstance(result, float) else str(result)
        return f"{expression.strip()} = {formatted}"
    except ZeroDivisionError:
        return "Error: division by zero."
    except OverflowError:
        return "Error: the result is too large to represent."
    except Exception as exc:
        return f"Error: cannot evaluate expression ({type(exc).__name__}: {exc})"


@tool
def get_system_info() -> str:
    """Get read-only information about this computer: operating system, Python version, CPU count, working directory and disk space."""
    try:
        cwd = os.getcwd()
        usage = shutil.disk_usage(cwd)
        gib = 1024**3
        lines = [
            f"OS: {platform.system()} {platform.release()} ({platform.version()})",
            f"Machine: {platform.machine()}",
            f"Python: {platform.python_version()}",
            f"CPU cores: {os.cpu_count()}",
            f"Working directory: {cwd}",
            f"Disk (working drive): {usage.free / gib:.1f} GiB free of {usage.total / gib:.1f} GiB",
        ]
        return "\n".join(lines)
    except Exception as exc:
        logger.exception("get_system_info failed")
        return f"Error: could not collect system info ({type(exc).__name__}: {exc})"
