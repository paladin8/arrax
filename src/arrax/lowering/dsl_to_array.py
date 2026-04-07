"""Lowering: traced DAG -> array dialect IR."""

from __future__ import annotations

from xdsl.dialects.builtin import Float32Type, ModuleOp, TensorType
from xdsl.dialects.func import FuncOp, ReturnOp
from xdsl.ir import SSAValue

from arrax.dialects.array_dialect import (
    AddOp,
    AmaxOp,
    DivScalarOp,
    DotOp,
    ExpOp,
    MeanOp,
    MulScalarOp,
    RMSNormOp,
    ReluOp,
    SoftmaxOp,
    SubOp,
    SumOp,
)
from arrax.dsl.array import Array


def visited_nodes(root: Array) -> list[Array]:
    """Return every unique DAG node reachable from root."""
    seen: set[int] = set()
    order: list[Array] = []

    def walk(node: Array) -> None:
        if id(node) in seen:
            return
        seen.add(id(node))
        order.append(node)
        for child in node.operands:
            walk(child)

    walk(root)
    return order


def dsl_to_array(
    result: Array,
    param_names: list[str],
    shapes: dict[str, tuple[int, ...]],
) -> ModuleOp:
    """Convert traced DAG to array dialect IR wrapped in a func.func.

    Walks the DAG bottom-up, mapping each Array node to an SSA value.
    Leaf nodes become function arguments; op nodes become dialect operations.
    """
    f32 = Float32Type()

    # Build function argument types in signature order
    input_types = [TensorType(f32, shapes[name]) for name in param_names]

    # Determine result type from the root node
    result_type = TensorType(f32, result.shape)

    # Create the function with an entry block whose args match input_types
    func_op = FuncOp("kernel", (input_types, [result_type]))
    entry_block = func_op.body.blocks.first
    assert entry_block is not None

    # Map parameter names to block arguments
    param_to_arg: dict[str, SSAValue] = {}
    for name, block_arg in zip(param_names, entry_block.args):
        param_to_arg[name] = block_arg

    # Walk DAG bottom-up, mapping each Array to an SSA value
    value_map: dict[int, SSAValue] = {}

    def lower(node: Array) -> SSAValue:
        node_id = id(node)
        if node_id in value_map:
            return value_map[node_id]

        if node.is_leaf:
            val = param_to_arg[node.name]
        elif node.op == "add":
            lhs = lower(node.operands[0])
            rhs = lower(node.operands[1])
            add_op = AddOp(lhs, rhs)
            entry_block.add_op(add_op)
            val = add_op.result
        elif node.op == "sub":
            lhs = lower(node.operands[0])
            rhs = lower(node.operands[1])
            sub_op = SubOp(lhs, rhs)
            entry_block.add_op(sub_op)
            val = sub_op.result
        elif node.op == "relu":
            operand = lower(node.operands[0])
            relu_op = ReluOp(operand)
            entry_block.add_op(relu_op)
            val = relu_op.result
        elif node.op == "exp":
            operand = lower(node.operands[0])
            exp_op = ExpOp(operand)
            entry_block.add_op(exp_op)
            val = exp_op.result
        elif node.op == "mul_scalar":
            assert node.scalar is not None
            operand = lower(node.operands[0])
            mul_op = MulScalarOp(operand, node.scalar)
            entry_block.add_op(mul_op)
            val = mul_op.result
        elif node.op == "div_scalar":
            assert node.scalar is not None
            operand = lower(node.operands[0])
            div_op = DivScalarOp(operand, node.scalar)
            entry_block.add_op(div_op)
            val = div_op.result
        elif node.op == "sum":
            operand = lower(node.operands[0])
            sum_op = SumOp(operand)
            entry_block.add_op(sum_op)
            val = sum_op.result
        elif node.op == "amax":
            operand = lower(node.operands[0])
            amax_op = AmaxOp(operand)
            entry_block.add_op(amax_op)
            val = amax_op.result
        elif node.op == "dot":
            lhs = lower(node.operands[0])
            rhs = lower(node.operands[1])
            dot_op = DotOp(lhs, rhs)
            entry_block.add_op(dot_op)
            val = dot_op.result
        elif node.op == "mean":
            operand = lower(node.operands[0])
            mean_op = MeanOp(operand)
            entry_block.add_op(mean_op)
            val = mean_op.result
        elif node.op == "softmax":
            operand = lower(node.operands[0])
            softmax_op = SoftmaxOp(operand)
            entry_block.add_op(softmax_op)
            val = softmax_op.result
        elif node.op == "rmsnorm":
            operand = lower(node.operands[0])
            rmsnorm_op = RMSNormOp(operand)
            entry_block.add_op(rmsnorm_op)
            val = rmsnorm_op.result
        else:
            raise ValueError(f"unsupported operation: {node.op}")

        value_map[node_id] = val
        return val

    result_val = lower(result)

    # Add return
    entry_block.add_op(ReturnOp(result_val))

    return ModuleOp([func_op])
