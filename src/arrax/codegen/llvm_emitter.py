"""LLVM IR generation for RISC-V + NPU intrinsics via llvmlite.

Walks NPU-dialect IR (same input as asm_emitter) and builds LLVM IR using
llvmlite. The output is a `.ll` text string compilable by an llc built
with the Xnpu vendor extension patches (see llvm-npu/).

This emitter walks the xDSL IR directly rather than going through xDSL's
LLVM dialect, because xDSL lacks the conversion passes for arith/memref/
scf/func → LLVM. The direct approach mirrors asm_emitter's traversal
pattern but outputs llvmlite IR instead of assembly text.

Each NPU op maps to one or more @llvm.riscv.npu.* intrinsic calls.
LLVM handles register allocation, prologue/epilogue, and scalar codegen.
"""

from __future__ import annotations

import llvmlite.ir as ir

from xdsl.dialects import arith, func, memref, scf
from xdsl.dialects.builtin import (
    Float32Type,
    FloatAttr,
    IntegerAttr,
    MemRefType,
    ModuleOp,
)
from xdsl.ir import Block, Operation, SSAValue

from arrax.dialects.npu_dialect import (
    FRsqrtOp,
    FVAddOp,
    FVDivOp,
    FVExpOp,
    FVMacOp,
    FVMaxOp,
    FVMulOp,
    FVReduceOp,
    FVReluOp,
    FVSubOp,
    FVSubScalarOp,
)

# LLVM IR types
_i32 = ir.IntType(32)
_f32 = ir.FloatType()
_void = ir.VoidType()
_ptr = ir.PointerType()  # opaque pointer


def emit_llvm_ir(module: ModuleOp) -> str:
    """Walk a module containing npu ops and emit LLVM IR text."""
    emitter = _LlvmEmitter()
    emitter.emit_module(module)
    return str(emitter.get_module())


class _LlvmEmitter:
    """Stateful LLVM IR emitter that walks xDSL IR and builds llvmlite IR."""

    def __init__(self) -> None:
        self._module = ir.Module(name="arrax")
        self._module.triple = "riscv32-unknown-none-elf"
        self._module.data_layout = "e-m:e-p:32:32-i64:64-n32-S128"
        self._val: dict[int, ir.Value] = {}  # id(SSAValue) -> llvmlite value
        self._builder: ir.IRBuilder | None = None
        self._intrinsics: dict[str, ir.Function] = {}

    def get_module(self) -> ir.Module:
        return self._module

    def emit_module(self, module: ModuleOp) -> None:
        for op in module.body.block.ops:
            if isinstance(op, func.FuncOp):
                self._emit_func(op)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _v(self, val: SSAValue) -> ir.Value:
        """Look up the llvmlite value for an xDSL SSA value."""
        return self._val[id(val)]

    def _set(self, val: SSAValue, llvm_val: ir.Value) -> None:
        """Record an SSA value mapping."""
        self._val[id(val)] = llvm_val

    def _convert_type(self, ty: object) -> ir.Type:
        """Convert an xDSL type to an llvmlite type."""
        if isinstance(ty, Float32Type):
            return _f32
        if isinstance(ty, MemRefType):
            return _ptr
        # Index type
        from xdsl.dialects.builtin import IndexType
        if isinstance(ty, IndexType):
            return _i32
        raise ValueError(f"llvm_emitter: unsupported type {ty}")

    def _get_or_declare_intrinsic(
        self, name: str, ret: ir.Type, arg_types: list[ir.Type]
    ) -> ir.Function:
        """Get or declare an LLVM intrinsic function."""
        if name not in self._intrinsics:
            ftype = ir.FunctionType(ret, arg_types)
            fn = ir.Function(self._module, ftype, name=name)
            fn.attributes.add("nounwind")
            self._intrinsics[name] = fn
        return self._intrinsics[name]

    def _const_i32(self, val: int) -> ir.Constant:
        return ir.Constant(_i32, val)

    def _const_f32(self, val: float) -> ir.Constant:
        return ir.Constant(_f32, val)

    # ------------------------------------------------------------------
    # Function emission
    # ------------------------------------------------------------------

    def _emit_func(self, func_op: func.FuncOp) -> None:
        name = func_op.sym_name.data
        block = func_op.body.blocks.first
        assert block is not None

        # Build function type: all args converted
        arg_types = [self._convert_type(arg.type) for arg in block.args]
        ret_type = _void  # arrax functions return void
        ftype = ir.FunctionType(ret_type, arg_types)
        fn = ir.Function(self._module, ftype, name=name)

        # Map function args
        entry = fn.append_basic_block("entry")
        self._builder = ir.IRBuilder(entry)
        for xdsl_arg, llvm_arg in zip(block.args, fn.args):
            self._set(xdsl_arg, llvm_arg)

        # Walk body ops
        for op in block.ops:
            if isinstance(op, func.ReturnOp):
                self._builder.ret_void()
            else:
                self._emit_op(op)

    # ------------------------------------------------------------------
    # Op dispatch
    # ------------------------------------------------------------------

    def _emit_op(self, op: object) -> None:
        """Dispatch to the appropriate emission method."""
        if isinstance(op, arith.ConstantOp):
            self._emit_constant(op)
        elif isinstance(op, memref.AllocOp):
            self._emit_alloc(op)
        elif isinstance(op, memref.AllocaOp):
            self._emit_alloca(op)
        elif isinstance(op, FVAddOp):
            self._emit_fv_binop(op, "llvm.riscv.npu.fvadd")
        elif isinstance(op, FVSubOp):
            self._emit_fv_binop(op, "llvm.riscv.npu.fvsub")
        elif isinstance(op, FVReluOp):
            self._emit_fv_unop(op, "llvm.riscv.npu.fvrelu")
        elif isinstance(op, FVExpOp):
            self._emit_fv_unop(op, "llvm.riscv.npu.fvexp")
        elif isinstance(op, FVMulOp):
            self._emit_fv_scalar_op(op, "llvm.riscv.npu.fvmul")
        elif isinstance(op, FVDivOp):
            self._emit_fv_scalar_op(op, "llvm.riscv.npu.fvdiv")
        elif isinstance(op, FVSubScalarOp):
            self._emit_fv_scalar_op(op, "llvm.riscv.npu.fvsub.scalar")
        elif isinstance(op, FRsqrtOp):
            self._emit_frsqrt(op)
        elif isinstance(op, FVReduceOp):
            self._emit_fv_reduce(op)
        elif isinstance(op, FVMaxOp):
            self._emit_fv_max(op)
        elif isinstance(op, FVMacOp):
            self._emit_fv_mac(op)
        elif isinstance(op, scf.ForOp):
            self._emit_for(op)
        elif isinstance(op, memref.SubviewOp):
            self._emit_subview(op)
        elif isinstance(op, memref.StoreOp):
            self._emit_store(op)
        elif isinstance(op, memref.LoadOp):
            self._emit_load(op)
        elif isinstance(op, arith.SubiOp):
            self._emit_subi(op)
        elif isinstance(op, arith.MinSIOp):
            self._emit_minsi(op)
        elif isinstance(op, arith.DivfOp):
            self._emit_divf(op)
        elif isinstance(op, arith.AddfOp):
            self._emit_addf(op)
        elif isinstance(op, scf.YieldOp):
            pass  # handled by _emit_for
        elif isinstance(op, Operation):
            raise ValueError(f"llvm_emitter: unsupported op {op.name}")

    # ------------------------------------------------------------------
    # Constants
    # ------------------------------------------------------------------

    def _emit_constant(self, op: arith.ConstantOp) -> None:
        if isinstance(op.value, IntegerAttr):
            self._set(op.result, self._const_i32(op.value.value.data))
        elif isinstance(op.value, FloatAttr) and isinstance(
            op.value.type, Float32Type
        ):
            self._set(op.result, self._const_f32(op.value.value.data))
        else:
            raise ValueError(
                f"llvm_emitter: unsupported constant type {op.value}"
            )

    # ------------------------------------------------------------------
    # Memory ops
    # ------------------------------------------------------------------

    def _emit_alloc(self, op: memref.AllocOp) -> None:
        """Emit alloca for a buffer (flat memory model, alloc = alloca)."""
        self._emit_mem_alloc(op.memref)

    def _emit_alloca(self, op: memref.AllocaOp) -> None:
        """Emit alloca for a stack buffer."""
        self._emit_mem_alloc(op.memref)

    def _emit_mem_alloc(self, memref_val: SSAValue) -> None:
        """Emit alloca for a memref, mapping the SSA value to the pointer."""
        assert isinstance(memref_val.type, MemRefType)
        shape = memref_val.type.get_shape()
        num_elements = 1
        for dim in shape:
            num_elements *= dim
        if num_elements == 0:
            num_elements = 1  # rank-0 memref = 1 element
        alloca = self._builder.alloca(_f32, size=self._const_i32(num_elements),
                                       name="buf")
        self._set(memref_val, alloca)

    def _emit_subview(self, op: memref.SubviewOp) -> None:
        """Emit GEP for a 1D subview: base + offset (f32 elements)."""
        base = self._v(op.source)
        offset = self._v(op.offsets[0])
        gep = self._builder.gep(base, [offset], source_etype=_f32, name="sv")
        self._set(op.result, gep)

    def _emit_store(self, op: memref.StoreOp) -> None:
        """Emit f32 store to a rank-0 memref."""
        val = self._v(op.value)
        ptr = self._v(op.memref)
        self._builder.store(val, ptr)

    def _emit_load(self, op: memref.LoadOp) -> None:
        """Emit f32 load from a rank-0 memref."""
        ptr = self._v(op.memref)
        result = self._builder.load(ptr, name="ld")
        self._set(op.res, result)

    # ------------------------------------------------------------------
    # Arith ops
    # ------------------------------------------------------------------

    def _emit_subi(self, op: arith.SubiOp) -> None:
        self._set(op.result, self._builder.sub(self._v(op.lhs), self._v(op.rhs),
                                                name="sub"))

    def _emit_minsi(self, op: arith.MinSIOp) -> None:
        lhs = self._v(op.lhs)
        rhs = self._v(op.rhs)
        cmp = self._builder.icmp_signed("<", lhs, rhs, name="cmp")
        self._set(op.result, self._builder.select(cmp, lhs, rhs, name="min"))

    def _emit_addf(self, op: arith.AddfOp) -> None:
        self._set(op.result, self._builder.fadd(self._v(op.lhs), self._v(op.rhs),
                                                 name="addf"))

    def _emit_divf(self, op: arith.DivfOp) -> None:
        self._set(op.result, self._builder.fdiv(self._v(op.lhs), self._v(op.rhs),
                                                 name="divf"))

    # ------------------------------------------------------------------
    # SCF loop
    # ------------------------------------------------------------------

    def _emit_for(self, op: scf.ForOp) -> None:
        """Emit a counted loop with optional f32 iter_args.

        Lowers scf.for to LLVM basic blocks:
          preheader → header (cond_br) → body → latch → header / exit
        """
        body = op.body.blocks.first
        assert body is not None

        fn = self._builder.function
        header = fn.append_basic_block("for.header")
        body_bb = fn.append_basic_block("for.body")
        exit_bb = fn.append_basic_block("for.exit")

        lb = self._v(op.lb)
        ub = self._v(op.ub)
        step = self._v(op.step)

        # Detect FVMacOp — bracket loop with FRSTACC
        has_fvmac = any(isinstance(bop, FVMacOp) for bop in body.ops)
        if has_fvmac:
            frstacc = self._get_or_declare_intrinsic(
                "llvm.riscv.npu.frstacc", _f32, []
            )
            self._builder.call(frstacc, [], name="facc.discard")

        # Collect iter_arg init values
        init_vals = [self._v(init) for init in op.iter_args]

        # Branch to header
        self._builder.branch(header)

        # Header: IV phi + iter_arg phis + condition check
        self._builder = ir.IRBuilder(header)
        iv_phi = self._builder.phi(_i32, name="iv")
        iv_phi.add_incoming(lb, fn.blocks[-4])  # preheader is 4 blocks back

        iter_phis: list[ir.PhiInstr] = []
        for i, init in enumerate(init_vals):
            phi = self._builder.phi(init.type, name=f"acc.{i}")
            phi.add_incoming(init, fn.blocks[-4])
            iter_phis.append(phi)

        # Map IV and iter_args to phi nodes
        self._set(body.args[0], iv_phi)
        for i, phi in enumerate(iter_phis):
            self._set(body.args[i + 1], phi)

        cond = self._builder.icmp_signed(">=", iv_phi, ub, name="done")
        self._builder.cbranch(cond, exit_bb, body_bb)

        # Body
        self._builder = ir.IRBuilder(body_bb)
        for body_op in body.ops:
            if isinstance(body_op, scf.YieldOp):
                # Yield: collect values for phi incoming, advance IV, branch back
                yield_vals = [self._v(v) for v in body_op.operands]
                next_iv = self._builder.add(iv_phi, step, name="iv.next")
                self._builder.branch(header)

                # Add incoming to phis from body
                iv_phi.add_incoming(next_iv, body_bb)
                for phi, yv in zip(iter_phis, yield_vals):
                    phi.add_incoming(yv, body_bb)
            else:
                self._emit_op(body_op)

        # Exit block
        self._builder = ir.IRBuilder(exit_bb)

        # Map loop results to the iter_arg phis (read after loop)
        for i, result in enumerate(op.results):
            self._set(result, iter_phis[i])

        # For FVMac loops: read facc after loop
        if has_fvmac:
            frstacc = self._get_or_declare_intrinsic(
                "llvm.riscv.npu.frstacc", _f32, []
            )
            facc_val = self._builder.call(frstacc, [], name="facc.read")
            # Override the loop result with the facc value
            if op.results:
                self._val[id(op.results[0])] = facc_val

    # ------------------------------------------------------------------
    # NPU vector ops
    # ------------------------------------------------------------------

    def _get_n(self, n_val: SSAValue) -> ir.Value:
        """Get the i32 element count value."""
        return self._v(n_val)

    def _emit_fv_binop(self, op: FVAddOp | FVSubOp, intrinsic_name: str) -> None:
        """Emit binary in-place vector op with optional copy.

        Hardware writes in-place to rs2. If src2 != dst, copy first.
        """
        src1 = self._v(op.src1)
        src2 = self._v(op.src2)
        dst = self._v(op.dst)
        n = self._get_n(op.n)

        # Copy src2 → dst if different buffers
        if op.src2 is not op.dst:
            self._emit_memcpy(dst, src2, n)

        fn = self._get_or_declare_intrinsic(
            intrinsic_name, _void, [_ptr, _ptr, _i32]
        )
        self._builder.call(fn, [src1, dst, n])

    def _emit_fv_unop(
        self, op: FVReluOp | FVExpOp, intrinsic_name: str
    ) -> None:
        """Emit unary vector op: dst[i] = f(src[i])."""
        src = self._v(op.src)
        dst = self._v(op.dst)
        n = self._get_n(op.n)

        fn = self._get_or_declare_intrinsic(
            intrinsic_name, _void, [_ptr, _ptr, _i32]
        )
        self._builder.call(fn, [src, dst, n])

    def _emit_fv_scalar_op(
        self, op: FVMulOp | FVDivOp | FVSubScalarOp, intrinsic_name: str
    ) -> None:
        """Emit scalar-vector op: load scalar into facc, then run vector op."""
        src = self._v(op.src)
        dst = self._v(op.dst)
        n = self._get_n(op.n)
        scalar = self._v(op.scalar)

        # Load scalar into facc: frstacc (zero), fmacc(scalar, 1.0)
        self._emit_facc_load(scalar)

        fn = self._get_or_declare_intrinsic(
            intrinsic_name, _void, [_ptr, _ptr, _i32]
        )
        self._builder.call(fn, [src, dst, n])

    def _emit_frsqrt(self, op: FRsqrtOp) -> None:
        """Emit FRSQRT: result = 1/sqrt(src)."""
        src = self._v(op.src)
        fn = self._get_or_declare_intrinsic(
            "llvm.riscv.npu.frsqrt", _f32, [_f32]
        )
        result = self._builder.call(fn, [src], name="rsqrt")
        self._set(op.result, result)

    def _emit_fv_reduce(self, op: FVReduceOp) -> None:
        """Emit FVREDUCE: partial = sum(src[0..n]), result = acc_in + partial."""
        src = self._v(op.src)
        n = self._get_n(op.n)
        acc_in = self._v(op.acc_in)

        fn = self._get_or_declare_intrinsic(
            "llvm.riscv.npu.fvreduce", _f32, [_ptr, _i32]
        )
        partial = self._builder.call(fn, [src, n], name="partial.sum")
        result = self._builder.fadd(acc_in, partial, name="acc.sum")
        self._set(op.result, result)

    def _emit_fv_max(self, op: FVMaxOp) -> None:
        """Emit FVMAX: partial = max(src[0..n]), result = max(acc_in, partial).

        Includes NaN propagation: if partial is NaN, use partial (not
        fmax.s which is NaN-suppressing).
        """
        src = self._v(op.src)
        n = self._get_n(op.n)
        acc_in = self._v(op.acc_in)

        fn = self._get_or_declare_intrinsic(
            "llvm.riscv.npu.fvmax", _f32, [_ptr, _i32]
        )
        partial = self._builder.call(fn, [src, n], name="partial.max")

        # maxnum(acc_in, partial) — NaN-suppressing
        maxnum = self._get_or_declare_intrinsic(
            "llvm.maxnum.f32", _f32, [_f32, _f32]
        )
        combined = self._builder.call(maxnum, [acc_in, partial], name="max.combined")

        # NaN check: if partial is NaN, override with partial
        is_nan = self._builder.fcmp_unordered("uno", partial, partial, name="isnan")
        result = self._builder.select(is_nan, partial, combined, name="acc.max")
        self._set(op.result, result)

    def _emit_fv_mac(self, op: FVMacOp) -> None:
        """Emit FVMAC: facc += dot(lhs[0..n], rhs[0..n]).

        The FRSTACC bracket is handled by _emit_for for tiled loops,
        or emitted inline for untiled dot products.
        """
        lhs = self._v(op.lhs)
        rhs = self._v(op.rhs)
        n = self._get_n(op.n)

        in_loop = isinstance(op.parent_op(), scf.ForOp)

        if not in_loop:
            # Untiled: full FRSTACC bracket
            frstacc = self._get_or_declare_intrinsic(
                "llvm.riscv.npu.frstacc", _f32, []
            )
            self._builder.call(frstacc, [], name="facc.zero")

        fvmac = self._get_or_declare_intrinsic(
            "llvm.riscv.npu.fvmac", _void, [_ptr, _ptr, _i32]
        )
        self._builder.call(fvmac, [lhs, rhs, n])

        if not in_loop:
            frstacc = self._get_or_declare_intrinsic(
                "llvm.riscv.npu.frstacc", _f32, []
            )
            facc_val = self._builder.call(frstacc, [], name="facc.read")
            self._set(op.result, facc_val)
        else:
            # Inside loop: result is cosmetic (facc holds real state).
            # Map result to acc_in so downstream yield sees the phi.
            self._set(op.result, self._v(op.acc_in))

    # ------------------------------------------------------------------
    # Facc helpers
    # ------------------------------------------------------------------

    def _emit_facc_load(self, scalar: ir.Value) -> None:
        """Load a scalar f32 value into the hardware facc register.

        Sequence: frstacc (zero facc), fmacc(scalar, 1.0).
        """
        frstacc = self._get_or_declare_intrinsic(
            "llvm.riscv.npu.frstacc", _f32, []
        )
        self._builder.call(frstacc, [], name="facc.zero")

        fmacc = self._get_or_declare_intrinsic(
            "llvm.riscv.npu.fmacc", _void, [_f32, _f32]
        )
        one = self._const_f32(1.0)
        self._builder.call(fmacc, [scalar, one])

    def _emit_memcpy(self, dst: ir.Value, src: ir.Value, n_elements: ir.Value) -> None:
        """Emit an element-wise copy loop: dst[0..n] = src[0..n].

        Uses llvm.memcpy intrinsic with byte count = n_elements * 4.
        """
        # n_bytes = n_elements * 4
        n_bytes = self._builder.mul(n_elements, self._const_i32(4), name="nbytes")

        memcpy = self._get_or_declare_intrinsic(
            "llvm.memcpy.p0.p0.i32", _void,
            [_ptr, _ptr, _i32, ir.IntType(1)]
        )
        is_volatile = ir.Constant(ir.IntType(1), 0)
        self._builder.call(memcpy, [dst, src, n_bytes, is_volatile])
