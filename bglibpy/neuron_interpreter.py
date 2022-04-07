"""Module to interpret neuron code strings."""

import ast
from typing import Dict, Any

from bglibpy import NeuronEvalError


def _recursive_evaluate(node: ast.AST, context: Dict[str, Any]) -> Any:
    """A limited evaluator for evaluating neuron code string."""
    if isinstance(node, ast.Constant):
        return node.value
    elif isinstance(node, ast.Name):
        return context[node.id]
    elif isinstance(node, ast.Attribute):
        base = _recursive_evaluate(node.value, context)
        return getattr(base, node.attr)
    elif isinstance(node, ast.Call):
        func = _recursive_evaluate(node.func, context)
        args = [_recursive_evaluate(arg, context) for arg in node.args]
        return func(*args)
    elif isinstance(node, ast.Subscript):
        base = _recursive_evaluate(node.value, context)
        index = _recursive_evaluate(node.slice.value, context)
        return base[index]
    else:
        raise NeuronEvalError("Unexpected code!")


def eval_neuron(source: str, **context) -> Any:
    """A limited interpreter for evaluating neuron code."""
    tree = ast.parse(source)

    if len(tree.body) != 1:
        raise NeuronEvalError("Neuron code should be a single expression")

    [node] = tree.body
    if not isinstance(node, ast.Expr):
        raise NeuronEvalError("Neuron code should be an expression")

    return _recursive_evaluate(node.value, context)
