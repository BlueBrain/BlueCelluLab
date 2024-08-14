# Copyright 2023-2024 Blue Brain Project / EPFL

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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
