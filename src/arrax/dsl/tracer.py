"""DAG capture and shape inference for traced array expressions."""

from __future__ import annotations

import inspect
from typing import Callable

from arrax.dsl.array import Array


def trace(
    fn: Callable[..., Array],
    shapes: dict[str, tuple[int, ...]],
) -> tuple[Array, list[str]]:
    """Call fn with placeholder Arrays and return (result_dag, param_names).

    The param_names list preserves the original function signature order,
    which downstream passes use for function argument ordering.
    """
    params = list(inspect.signature(fn).parameters.keys())
    for name in params:
        if name not in shapes:
            raise ValueError(
                f"missing shape for parameter '{name}'; "
                f"shapes has keys {list(shapes.keys())}"
            )
    inputs = {name: Array(name, shapes[name]) for name in params}
    result = fn(**inputs)
    return result, params
