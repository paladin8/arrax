"""Lazy Array class with operator overloading for DAG capture."""

from __future__ import annotations


class Array:
    """A lazy array that records operations into a DAG instead of executing them.

    An Array with op=None is a leaf (function parameter).
    An Array with op set is a DAG node whose operands are other Arrays.
    """

    def __init__(self, name: str, shape: tuple[int, ...]) -> None:
        self.name = name
        self.shape = shape
        self.op: str | None = None
        self.operands: list[Array] = []

    def __add__(self, other: Array) -> Array:
        result = Array(name="", shape=self.shape)
        result.op = "add"
        result.operands = [self, other]
        return result

    def __sub__(self, other: Array) -> Array:
        result = Array(name="", shape=self.shape)
        result.op = "sub"
        result.operands = [self, other]
        return result

    @property
    def is_leaf(self) -> bool:
        return self.op is None


def relu(x: Array) -> Array:
    """Elementwise ReLU: max(x, 0)."""
    result = Array(name="", shape=x.shape)
    result.op = "relu"
    result.operands = [x]
    return result


def exp(x: Array) -> Array:
    """Elementwise exponential."""
    result = Array(name="", shape=x.shape)
    result.op = "exp"
    result.operands = [x]
    return result
