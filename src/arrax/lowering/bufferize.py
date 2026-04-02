"""Bufferization: tensor types -> memref types.

Custom pass scoped to the IR shapes produced by arrax's pipeline.
xDSL 0.59.0 has no general-purpose one-shot bufferization.
"""

from __future__ import annotations

from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects import func, linalg, memref, tensor
from xdsl.dialects.builtin import MemRefType, ModuleOp, TensorType
from xdsl.ir import Block, Region, SSAValue
from xdsl.passes import ModulePass


def _tensor_to_memref(t: TensorType) -> MemRefType:
    """Convert tensor<...xf32> to memref<...xf32>."""
    return MemRefType(t.element_type, t.get_shape())


@dataclass(frozen=True)
class BufferizePass(ModulePass):
    """Convert tensor-semantic IR to memref-semantic IR.

    Rebuilds each FuncOp with memref-typed arguments, destination-passing
    style (output memrefs as function args), and void return. Intermediate
    tensors become memref.alloc; the final output becomes a function arg.
    """

    name = "bufferize"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        for func_op in list(op.body.block.ops):
            if isinstance(func_op, func.FuncOp):
                new_func = self._bufferize_func(func_op)
                parent_block = func_op.parent
                assert parent_block is not None
                parent_block.insert_op_after(new_func, func_op)
                func_op.detach()
                func_op.erase()

    def _bufferize_func(self, func_op: func.FuncOp) -> func.FuncOp:
        old_block = func_op.body.blocks.first
        assert old_block is not None

        # Collect old types
        old_input_types = [
            t for t in func_op.function_type.inputs.data
            if isinstance(t, TensorType)
        ]
        old_output_types = [
            t for t in func_op.function_type.outputs.data
            if isinstance(t, TensorType)
        ]

        memref_inputs = [_tensor_to_memref(t) for t in old_input_types]
        memref_outputs = [_tensor_to_memref(t) for t in old_output_types]
        all_arg_types = memref_inputs + memref_outputs

        # Find the final tensor.empty: trace back from func.return
        # NOTE: only single-result functions are supported (Milestone 1 scope)
        return_op = self._find_return(old_block)
        final_empty_op: tensor.EmptyOp | None = None
        if return_op.arguments:
            returned_val = return_op.arguments[0]
            assert isinstance(returned_val.owner, linalg.GenericOp)
            final_outs_val = list(returned_val.owner.outputs)[0]
            assert isinstance(final_outs_val.owner, tensor.EmptyOp)
            final_empty_op = final_outs_val.owner

        # Build new block with memref args
        new_block = Block(arg_types=all_arg_types)

        # SSA value mapping: old → new
        value_map: dict[SSAValue, SSAValue] = {}
        for old_arg, new_arg in zip(old_block.args, new_block.args):
            value_map[old_arg] = new_arg

        # Map final tensor.empty result → output memref function arg
        if final_empty_op is not None:
            output_arg = new_block.args[len(memref_inputs)]
            value_map[final_empty_op.tensor] = output_arg

        # Walk old ops and emit bufferized versions
        for op in old_block.ops:
            if isinstance(op, tensor.EmptyOp):
                if op is final_empty_op:
                    continue  # mapped to output arg
                # Intermediate buffer: allocate
                memref_type = _tensor_to_memref(op.tensor.type)
                alloc = memref.AllocOp([], [], memref_type)
                new_block.add_op(alloc)
                value_map[op.tensor] = alloc.memref

            elif isinstance(op, linalg.GenericOp):
                new_ins = [value_map[v] for v in op.inputs]
                new_outs = [value_map[v] for v in op.outputs]

                # Clone the body region — scalar types (f32) are unchanged
                new_body = op.body.clone()

                new_generic = linalg.GenericOp(
                    inputs=new_ins,
                    outputs=new_outs,
                    body=new_body,
                    indexing_maps=op.indexing_maps,
                    iterator_types=op.iterator_types,
                    result_types=[],  # memref semantics: no results
                )
                new_block.add_op(new_generic)

                # Map old tensor result → outs memref
                # (in memref mode, the result is written to the outs buffer)
                if op.res:
                    value_map[op.res[0]] = new_outs[0]

            elif isinstance(op, func.ReturnOp):
                new_block.add_op(func.ReturnOp())

            else:
                raise ValueError(
                    f"bufferize: unsupported op {op.name} in function body"
                )

        return func.FuncOp(
            name=func_op.sym_name.data,
            function_type=(all_arg_types, []),
            region=Region([new_block]),
        )

    @staticmethod
    def _find_return(block: Block) -> func.ReturnOp:
        """Find the func.return terminator in a block."""
        for op in block.ops:
            if isinstance(op, func.ReturnOp):
                return op
        raise ValueError("no func.return found in block")
