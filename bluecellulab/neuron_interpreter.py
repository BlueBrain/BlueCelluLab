"""Module to interpret NEURON code strings."""

from __future__ import annotations
import ast
import sys
from typing import Any

from bluecellulab.exceptions import NeuronEvalError

PY39_PLUS = sys.version_info >= (3, 9)


def _recursive_evaluate(node: ast.AST, context: dict[str, Any]) -> Any:
    """A limited evaluator for evaluating NEURON code string."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return context[node.id]
    if isinstance(node, ast.Attribute):
        base = _recursive_evaluate(node.value, context)
        return getattr(base, node.attr)
    if isinstance(node, ast.Call):
        func = _recursive_evaluate(node.func, context)
        args = [_recursive_evaluate(arg, context) for arg in node.args]
        return func(*args)
    if isinstance(node, ast.Subscript):
        base = _recursive_evaluate(node.value, context)
        if PY39_PLUS:
            index = _recursive_evaluate(node.slice, context)
        else:
            index = _recursive_evaluate(node.slice.value, context)  # type: ignore
        return base[index]
    raise NeuronEvalError("Unexpected code!")


def eval_neuron(source: str, **context) -> Any:
    """A limited interpreter for evaluating NEURON code."""
    tree = ast.parse(source)

    if len(tree.body) != 1:
        raise NeuronEvalError("NEURON code should be a single expression")

    [node] = tree.body
    if not isinstance(node, ast.Expr):
        raise NeuronEvalError("NEURON code should be an expression")

    return _recursive_evaluate(node.value, context)
