"""Tests for arrax.dsl.array — lazy Array class."""

from __future__ import annotations

from arrax.dsl.array import Array


class TestArray:
    def test_leaf_construction(self) -> None:
        a = Array("A", (1024,))
        assert a.name == "A"
        assert a.shape == (1024,)
        assert a.is_leaf
        assert a.op is None
        assert a.operands == []

    def test_add_creates_dag_node(self) -> None:
        a = Array("A", (1024,))
        b = Array("B", (1024,))
        c = a + b

        assert c.op == "add"
        assert not c.is_leaf
        assert len(c.operands) == 2
        assert c.operands[0] is a
        assert c.operands[1] is b

    def test_add_propagates_shape(self) -> None:
        a = Array("A", (512,))
        b = Array("B", (512,))
        c = a + b
        assert c.shape == (512,)

    def test_add_result_has_empty_name(self) -> None:
        a = Array("A", (64,))
        b = Array("B", (64,))
        c = a + b
        assert c.name == ""

    def test_chained_add(self) -> None:
        """(A + B) + C builds a two-level DAG."""
        a = Array("A", (100,))
        b = Array("B", (100,))
        c = Array("C", (100,))
        result = (a + b) + c

        assert result.op == "add"
        assert result.operands[1] is c
        inner = result.operands[0]
        assert inner.op == "add"
        assert inner.operands[0] is a
        assert inner.operands[1] is b

    def test_sub_creates_dag_node(self) -> None:
        a = Array("A", (1024,))
        b = Array("B", (1024,))
        c = a - b

        assert c.op == "sub"
        assert not c.is_leaf
        assert len(c.operands) == 2
        assert c.operands[0] is a
        assert c.operands[1] is b

    def test_sub_propagates_shape(self) -> None:
        a = Array("A", (512,))
        b = Array("B", (512,))
        c = a - b
        assert c.shape == (512,)

    def test_mixed_add_sub(self) -> None:
        """(A + B) - C builds a mixed DAG."""
        a = Array("A", (100,))
        b = Array("B", (100,))
        c = Array("C", (100,))
        result = (a + b) - c

        assert result.op == "sub"
        assert result.operands[1] is c
        inner = result.operands[0]
        assert inner.op == "add"

    def test_2d_shape(self) -> None:
        a = Array("A", (32, 64))
        b = Array("B", (32, 64))
        c = a + b
        assert c.shape == (32, 64)
