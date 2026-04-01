"""Lazy Array class with operator overloading for DAG capture."""

from __future__ import annotations


class Array:
    """A lazy array that records operations into a DAG instead of executing them.

    Operator overloads create DAG nodes. Call compile() on the resulting
    expression to lower through the compiler pipeline.
    """
