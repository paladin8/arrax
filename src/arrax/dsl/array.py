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
        self.scalar: float | None = None

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

    def __mul__(self, other: float) -> Array:
        result = Array(name="", shape=self.shape)
        result.op = "mul_scalar"
        result.operands = [self]
        result.scalar = float(other)
        return result

    def __rmul__(self, other: float) -> Array:
        return self.__mul__(other)

    def __truediv__(self, other: float) -> Array:
        result = Array(name="", shape=self.shape)
        result.op = "div_scalar"
        result.operands = [self]
        result.scalar = float(other)
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


def sum(x: Array) -> Array:
    """Sum-reduction across all elements. Returns a rank-0 Array."""
    result = Array(name="", shape=())
    result.op = "sum"
    result.operands = [x]
    return result


def dot(a: Array, b: Array) -> Array:
    """Dot product of two 1D arrays. Returns a rank-0 Array.

    Requires both inputs to be 1D with matching shapes.
    """
    if len(a.shape) != 1 or len(b.shape) != 1:
        raise ValueError(
            f"dot requires 1D inputs, got shapes {a.shape} and {b.shape}"
        )
    if a.shape != b.shape:
        raise ValueError(
            f"dot requires matching shapes, got {a.shape} and {b.shape}"
        )
    result = Array(name="", shape=())
    result.op = "dot"
    result.operands = [a, b]
    return result


def amax(x: Array) -> Array:
    """Max-reduction across all elements. Returns a rank-0 Array.

    Named after NumPy's ``amax`` to avoid shadowing Python's ``max`` builtin.
    """
    result = Array(name="", shape=())
    result.op = "amax"
    result.operands = [x]
    return result


def mean(x: Array) -> Array:
    """Mean-reduction across all elements. Returns a rank-0 Array.

    Equivalent to ``sum(x) / len(x)`` but compiled as a single reduction with
    a trailing divide.
    """
    result = Array(name="", shape=())
    result.op = "mean"
    result.operands = [x]
    return result


def softmax(x: Array) -> Array:
    """Softmax: exp(x - max(x)) / sum(exp(x - max(x))). Returns same shape.

    Decomposed in the compiler into amax, broadcast-sub, exp, sum, broadcast-div.
    Numerically stable via max-subtraction before exp.
    """
    result = Array(name="", shape=x.shape)
    result.op = "softmax"
    result.operands = [x]
    return result


def rmsnorm(x: Array) -> Array:
    """RMSNorm (no gamma): x / sqrt(mean(x^2) + eps). Returns same shape.

    Decomposed in the compiler into dot(x,x) reduction + scalar math
    (div by N, add eps, rsqrt) + broadcast-mul. Hardcoded eps=1e-5.
    """
    result = Array(name="", shape=x.shape)
    result.op = "rmsnorm"
    result.operands = [x]
    return result
