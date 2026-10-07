import ast
import math
import operator
from typing import Any

from langchain_core.tools import tool

# Supported operators for safe AST evaluation
SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

SAFE_FUNCTIONS = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sqrt": math.sqrt,
    "ceil": math.ceil,
    "floor": math.floor,
}


def safe_eval(node: ast.AST) -> int | float:
    """Recursively evaluate an AST node containing mathematical expressions safely."""
    if isinstance(node, ast.Expression):
        return safe_eval(node.body)
    elif isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Unsupported constant type: {type(node.value)}")
    elif isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in SAFE_OPERATORS:
            raise ValueError(f"Unsupported operator: {op_type}")
        left = safe_eval(node.left)
        right = safe_eval(node.right)
        return SAFE_OPERATORS[op_type](left, right)
    elif isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in SAFE_OPERATORS:
            raise ValueError(f"Unsupported unary operator: {op_type}")
        operand = safe_eval(node.operand)
        return SAFE_OPERATORS[op_type](operand)
    elif isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id in SAFE_FUNCTIONS:
            args = [safe_eval(arg) for arg in node.args]
            return SAFE_FUNCTIONS[node.func.id](*args)
        raise ValueError("Unsupported function call")
    else:
        raise ValueError(f"Unsupported expression element: {type(node)}")


@tool
def calculate(expression: str) -> dict[str, Any]:
    """Perform mathematical calculations safely (supports +, -, *, /, **, %, round, abs, sqrt).

    Args:
        expression: A mathematical expression string, e.g. '(72000 - 25000) / 25000 * 100' or 'sqrt(144)'.
    """
    clean_expr = expression.strip()
    try:
        parsed = ast.parse(clean_expr, mode="eval")
        result = safe_eval(parsed)
        return {
            "expression": clean_expr,
            "result": result,
            "status": "success",
        }
    except ZeroDivisionError:
        return {
            "expression": clean_expr,
            "error": "Division by zero is not allowed.",
            "status": "error",
        }
    except Exception as e:
        return {
            "expression": clean_expr,
            "error": f"Invalid mathematical expression: {e!s}",
            "status": "error",
        }


@tool
def add(a: float, b: float) -> float:
    """Add two numbers (a + b)."""
    return a + b


@tool
def subtract(a: float, b: float) -> float:
    """Subtract two numbers (a - b)."""
    return a - b


@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers (a * b)."""
    return a * b


@tool
def divide(a: float, b: float) -> dict[str, Any]:
    """Divide a by b (a / b)."""
    if b == 0:
        return {"error": "Division by zero", "status": "error"}
    return {"result": a / b, "status": "success"}
