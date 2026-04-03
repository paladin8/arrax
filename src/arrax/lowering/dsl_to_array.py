"""Lowering: traced DAG -> array dialect IR."""

from __future__ import annotations

from xdsl.dialects.builtin import Float32Type, ModuleOp, TensorType
from xdsl.dialects.func import FuncOp, ReturnOp
from xdsl.ir import SSAValue

from arrax.dialects.array_dialect import AddOp, SubOp
from arrax.dsl.array import Array


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
        else:
            raise ValueError(f"unsupported operation: {node.op}")

        value_map[node_id] = val
        return val

    result_val = lower(result)

    # Add return
    entry_block.add_op(ReturnOp(result_val))

    return ModuleOp([func_op])
