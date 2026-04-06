"""Tests for arrax.dsl.array — lazy Array class."""

from __future__ import annotations

import pytest

from arrax.dsl.array import Array, amax, dot, exp, mean, relu, sum


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

    def test_relu_creates_dag_node(self) -> None:
        a = Array("A", (1024,))
        r = relu(a)

        assert r.op == "relu"
        assert not r.is_leaf
        assert len(r.operands) == 1
        assert r.operands[0] is a
        assert r.shape == (1024,)

    def test_exp_creates_dag_node(self) -> None:
        a = Array("A", (512,))
        e = exp(a)

        assert e.op == "exp"
        assert len(e.operands) == 1
        assert e.operands[0] is a
        assert e.shape == (512,)

    def test_relu_of_add(self) -> None:
        """relu(A + B) builds a two-level DAG."""
        a = Array("A", (100,))
        b = Array("B", (100,))
        result = relu(a + b)

        assert result.op == "relu"
        inner = result.operands[0]
        assert inner.op == "add"

    def test_mul_scalar(self) -> None:
        a = Array("A", (64,))
        r = a * 3.0
        assert r.op == "mul_scalar"
        assert r.scalar == 3.0
        assert len(r.operands) == 1
        assert r.operands[0] is a

    def test_rmul_scalar(self) -> None:
        a = Array("A", (64,))
        r = 2.0 * a
        assert r.op == "mul_scalar"
        assert r.scalar == 2.0

    def test_div_scalar(self) -> None:
        a = Array("A", (64,))
        r = a / 4.0
        assert r.op == "div_scalar"
        assert r.scalar == 4.0

    def test_2d_shape(self) -> None:
        a = Array("A", (32, 64))
        b = Array("B", (32, 64))
        c = a + b
        assert c.shape == (32, 64)

    def test_rank0_construction(self) -> None:
        """Array accepts shape=() for scalar (rank-0) values."""
        a = Array("s", ())
        assert a.shape == ()
        assert a.is_leaf
        assert a.op is None

    def test_sum_creates_dag_node(self) -> None:
        """sum(A) builds a rank-0 DAG node."""
        a = Array("A", (1024,))
        s = sum(a)
        assert s.op == "sum"
        assert not s.is_leaf
        assert s.shape == ()
        assert len(s.operands) == 1
        assert s.operands[0] is a

    def test_sum_of_add(self) -> None:
        """sum(A + B) is a rank-0 reduction of an elementwise node."""
        a = Array("A", (100,))
        b = Array("B", (100,))
        s = sum(a + b)
        assert s.op == "sum"
        assert s.shape == ()
        inner = s.operands[0]
        assert inner.op == "add"

    def test_amax_creates_dag_node(self) -> None:
        """amax(A) builds a rank-0 DAG node."""
        a = Array("A", (1024,))
        m = amax(a)
        assert m.op == "amax"
        assert not m.is_leaf
        assert m.shape == ()
        assert len(m.operands) == 1
        assert m.operands[0] is a

    def test_amax_of_sub(self) -> None:
        """amax(A - B) is a rank-0 reduction of an elementwise node."""
        a = Array("A", (100,))
        b = Array("B", (100,))
        m = amax(a - b)
        assert m.op == "amax"
        assert m.shape == ()
        inner = m.operands[0]
        assert inner.op == "sub"

    def test_dot_creates_dag_node(self) -> None:
        """dot(A, B) builds a rank-0 DAG node with two operands."""
        a = Array("A", (1024,))
        b = Array("B", (1024,))
        d = dot(a, b)
        assert d.op == "dot"
        assert not d.is_leaf
        assert d.shape == ()
        assert len(d.operands) == 2
        assert d.operands[0] is a
        assert d.operands[1] is b

    def test_dot_of_add(self) -> None:
        """dot(A + B, C) is a rank-0 reduction of an elementwise node."""
        a = Array("A", (100,))
        b = Array("B", (100,))
        c = Array("C", (100,))
        d = dot(a + b, c)
        assert d.op == "dot"
        assert d.shape == ()
        assert d.operands[0].op == "add"
        assert d.operands[1] is c

    def test_dot_shape_mismatch_raises(self) -> None:
        """dot(A, B) raises ValueError when shapes differ."""
        a = Array("A", (32,))
        b = Array("B", (64,))
        with pytest.raises(ValueError, match="shape"):
            dot(a, b)

    def test_dot_non_1d_raises(self) -> None:
        """dot requires 1D inputs."""
        a = Array("A", (4, 8))
        b = Array("B", (4, 8))
        with pytest.raises(ValueError, match="1D"):
            dot(a, b)

    def test_mean_creates_dag_node(self) -> None:
        a = Array("A", (64,))
        result = mean(a)
        assert result.shape == ()
        assert result.op == "mean"
        assert result.operands == [a]

    def test_mean_of_add(self) -> None:
        a = Array("A", (64,))
        b = Array("B", (64,))
        result = mean(a + b)
        assert result.op == "mean"
        assert result.operands[0].op == "add"
