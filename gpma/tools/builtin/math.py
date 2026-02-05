"""
Math Tools

Safe mathematical expression evaluation.

Uses AST parsing to prevent code injection while allowing
complex mathematical operations.
"""

from __future__ import annotations

import ast
import logging
import operator
from typing import TYPE_CHECKING, Any, Dict, List, Union

from ..base import BaseTool, ToolCategory, ToolParameter, ToolResult

if TYPE_CHECKING:
    from ..registry import ToolRegistry

logger = logging.getLogger(__name__)


class SafeCalculator:
    """
    Safe mathematical expression evaluator.

    Uses AST parsing instead of eval() to prevent code injection.
    Supports: +, -, *, /, **, %, //, abs, round, min, max, sqrt, pow, sum
    """

    ALLOWED_OPERATORS = {
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

    ALLOWED_FUNCTIONS = {
        'abs': abs,
        'round': round,
        'min': min,
        'max': max,
        'sqrt': lambda x: x ** 0.5,
        'pow': pow,
        'sum': sum,
        'int': int,
        'float': float,
    }

    MAX_VALUE = 10 ** 100  # Prevent overflow attacks

    def evaluate(self, expression: str) -> Union[int, float, str]:
        """
        Safely evaluate a mathematical expression.

        Args:
            expression: Mathematical expression string

        Returns:
            Numeric result or error message
        """
        try:
            # Clean the expression
            expression = expression.strip()

            # Parse to AST
            tree = ast.parse(expression, mode='eval')

            # Evaluate safely
            result = self._eval_node(tree.body)

            # Check for overflow
            if isinstance(result, (int, float)) and abs(result) > self.MAX_VALUE:
                return "Error: Result too large"

            return result

        except SyntaxError:
            return "Error: Invalid expression syntax"
        except ZeroDivisionError:
            return "Error: Division by zero"
        except Exception as e:
            return f"Error: {str(e)}"

    def _eval_node(self, node: ast.AST) -> Union[int, float]:
        """Recursively evaluate an AST node."""
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError(f"Unsupported constant type: {type(node.value)}")

        elif isinstance(node, ast.Num):  # Python 3.7 compatibility
            return node.n

        elif isinstance(node, ast.BinOp):
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            op = self.ALLOWED_OPERATORS.get(type(node.op))
            if op is None:
                raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
            return op(left, right)

        elif isinstance(node, ast.UnaryOp):
            operand = self._eval_node(node.operand)
            op = self.ALLOWED_OPERATORS.get(type(node.op))
            if op is None:
                raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
            return op(operand)

        elif isinstance(node, ast.Call):
            func_name = node.func.id if isinstance(node.func, ast.Name) else None
            if func_name not in self.ALLOWED_FUNCTIONS:
                raise ValueError(f"Function not allowed: {func_name}")

            args = [self._eval_node(arg) for arg in node.args]
            return self.ALLOWED_FUNCTIONS[func_name](*args)

        elif isinstance(node, ast.List):
            return [self._eval_node(elem) for elem in node.elts]

        elif isinstance(node, ast.Tuple):
            return tuple(self._eval_node(elem) for elem in node.elts)

        else:
            raise ValueError(f"Unsupported expression type: {type(node).__name__}")


# Global calculator instance
_calculator = SafeCalculator()


async def calculate(expression: str) -> str:
    """
    Safely evaluate a mathematical expression.

    Args:
        expression: Math expression (e.g., "25 * 4", "sqrt(16)", "max(1, 2, 3)")

    Returns:
        Result string or error message
    """
    result = _calculator.evaluate(expression)

    if isinstance(result, str) and result.startswith("Error"):
        return result

    return f"Result: {result}"


def create_calculator_tool() -> BaseTool:
    """Create the calculator tool."""
    return BaseTool(
        name="calculator",
        description=(
            "Safely evaluate mathematical expressions. "
            "Supports: +, -, *, /, **, %, //, sqrt(), abs(), round(), min(), max(), pow(), sum(). "
            "Example: 'sqrt(16) + 5 * 2'"
        ),
        parameters=[
            ToolParameter.string(
                name="expression",
                description="Mathematical expression to evaluate",
                required=True
            )
        ],
        function=calculate,
        category=ToolCategory.MATH,
        timeout=5.0,
        tags=["math", "calculator", "arithmetic"]
    )


def register_math_tools(registry: "ToolRegistry") -> None:
    """Register all math tools in the registry."""
    registry.register(create_calculator_tool())
    logger.debug("Registered math tools")
