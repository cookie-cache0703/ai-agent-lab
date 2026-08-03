"""Tool: evaluates a basic arithmetic expression.

Expressions are parsed with `ast` and walked by hand rather than passed to
`eval`, so only numeric literals and +, -, *, /, %, ** are ever executed —
there's no way to reach names, calls, or attribute access.
"""

import ast
import operator

from pydantic import BaseModel

from tools.base import Tool

_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


class CalculatorArgs(BaseModel):
    expression: str


class CalculatorError(Exception):
    """Raised when an expression contains something other than arithmetic."""


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
        return _OPERATORS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPERATORS:
        return _OPERATORS[type(node.op)](_eval_node(node.operand))
    raise CalculatorError(f"unsupported expression near {ast.dump(node)}")


def _calculate(args: CalculatorArgs) -> str:
    try:
        tree = ast.parse(args.expression, mode="eval")
        result = _eval_node(tree.body)
    except (SyntaxError, CalculatorError, ZeroDivisionError, TypeError) as e:
        return f"error: could not evaluate {args.expression!r}: {e}"
    return str(result)


calculator_tool = Tool(
    name="calculator",
    description="Evaluate a basic arithmetic expression (+, -, *, /, %, **, parentheses).",
    args_model=CalculatorArgs,
    handler=_calculate,
)
