"""Run comma.ai's openpilot driving model pkl (QCOM/AMD-compiled) on a local
CUDA GPU, using comma's own pinned tinygrad (vendored at
openpilot-release-chestnut/tinygrad_repo).

Pipeline:
  1. alias the pickled QCOM/AMD device strings to the local CUDA device
  2. per kernel: strip the pickled PROGRAM to its SINK AST, apply small graph
     fixes (FDIV, half-rcp, bool ops, ALU operand dtypes, 2D image index
     flatten), then to_program() under Context(IMAGE=1) -- which reproduces
     comma's per-texel image semantics on CUDA
  3. renderer patches: texel addressing (row*W+x)*C*itemsize, storage-width
     vector loads/stores with per-lane cvt widening/narrowing, vector CASTs,
     SUB/NEG opcodes
  4. launch config: NOLOCALS kernels fold local_size into global_size
"""
import io, math, os, re, struct, pickle, sys, time, types as _types
import numpy as np

TARGET = os.getenv('TG_DEV', 'CUDA')
PKL = sys.argv[1] if len(sys.argv) > 1 else 'models/driving.pkl'

from tinygrad import Device as _DevMod, Tensor
from tinygrad.dtype import dtypes, AddrSpace as _AS
from tinygrad.uop.ops import Ops, UOp, GroupOp, graph_rewrite, PatternMatcher, UPat
from tinygrad.codegen import to_program
from tinygrad.renderer.ptx import asm_for_op
from tinygrad.helpers import Context as _Ctx, NOOPT
from tinygrad.engine.realize import run_linear, get_call_arg_uops, get_call_outs_ins
from tinygrad.engine.jit import _prepare_jit_inputs
import tinygrad.renderer.ptx as _ptxmod
import inspect as _inspect, textwrap as _tw

# --- 1) renderer patches --------------------------------------------------------
# 1a. missing ALU opcodes (comma's table lacks them; their pipeline rewrites)
asm_for_op[Ops.SUB] = lambda d, a, b, dt, name: f"sub.{name} {d}, {a}, {b};"
asm_for_op[Ops.NEG] = lambda d, a, dt, name: f"neg.{name} {d}, {a};"

# 1b. helpers used by the patched render (injected into its globals below)
def _pbuf(u):
    d = u
    for _ in range(8):
        if d.op in (Ops.PARAM, Ops.BUFFER): return d
        if not d.src: return None
        d = d.src[0]
    return None

def _pimg(u):
    p = _pbuf(u)
    if p is None or not p.src or p.src[0].op is not Ops.STACK: return None
    dims = [s.arg for s in p.src[0].src]
    if len(dims) == 3 and all(isinstance(d, int) for d in dims): return dims
    return None

# 1c. addressing: flat element index * storage itemsize; for 3D image params
# the SINK index is a TEXEL index, so scale by C*itemsize
def _addrflat(ctx, x, buf, idx):
    p = _pbuf(buf)
    dims = _pimg(buf) if p is not None else None
    # image branch: 3-src INDEX(param(A,B,C), y, x) — OpenCL CLK_ADDRESS_CLAMP
    # semantics at register level: address edge-clamped for safety, and a
    # per-axis validity predicate is emitted for the LOAD hook to zero-mask
    if p is not None and dims is not None and len(x.src) >= 3:
        A, B = dims[0], dims[1]
        yreg = ctx.r[idx]
        xreg = ctx.r[x.src[2]]
        yreg = yreg if isinstance(yreg, str) else yreg[0]
        xreg = xreg if isinstance(xreg, str) else xreg[0]
        fl, fc, vv = ctx._img_sc[id(x)]
        v1, v2, v3, v4, v5, v6, a1, a2, v = vv
        ctx._img_valid[id(x)] = v
        return [f"mad.lo.s32 {fl}, {yreg}, {B}, {xreg};",
                f"setp.lt.s32 {v1}, {fl}, 0;",
                f"selp.s32 {fc}, 0, {fl}, {v1};",
                f"setp.lt.s32 {v2}, {fc}, {A*B};",
                f"selp.s32 {fc}, {fc}, {A*B}, {v2};",
                f"setp.gt.s32 {v3}, {xreg}, -1;",
                f"setp.lt.s32 {v4}, {xreg}, {B};",
                f"setp.gt.s32 {v5}, {yreg}, -1;",
                f"setp.lt.s32 {v6}, {yreg}, {A};",
                f"and.pred {a1}, {v3}, {v4};",
                f"and.pred {a2}, {v5}, {v6};",
                f"and.pred {v}, {a1}, {a2};",
                f"cvt.s64.s32 {ctx.r[x]}, {fc};",
                f"mad.lo.s64 {ctx.r[x]}, {ctx.r[x]}, {dims[2] * p.dtype.itemsize}, {ctx.r[buf]};"]
    if p is not None and dims is not None:
        its = dims[2] * p.dtype.itemsize
    elif p is not None:
        its = p.dtype.itemsize
    else:
        its = x.dtype.itemsize
    return [f"cvt.s64.{ctx.types[idx.dtype]} {ctx.r[x]}, {ctx.r[idx]};",
            f"mad.lo.s64 {ctx.r[x]}, {ctx.r[x]}, {its}, {ctx.r[buf]};"]

_addr_pm = PatternMatcher([
    (UPat((Ops.INDEX, Ops.SHRINK), src=(UPat(name="buf"), UPat(name="idx")), name="x", allow_any_len=True),
     lambda ctx, x, buf, idx: _addrflat(ctx, x, buf, idx)),
])
_ptxmod.string_rewrite = _addr_pm + _ptxmod.string_rewrite

# 1f. WMMA lowering (big model). comma's pkl linearized the attention GEMMs
# with the AMD RDNA4 tensor-core spec: WMMA arg ((16,16,16), half, 'AMD', 32),
# 32-lane wave, per-thread fragments A/B=8 halves, C/D=8 floats (the len==8
# ops_python branch; NOT rdna3's 16). The stock PTX renderer would emit
# mma.sync.m16n16k16, which ptxas rejects on EVERY NVIDIA arch (probed
# sm_75/80/86/89 — that shape only exists in wmma.sync). Emission-time
# lowering: stage each lane's A/B fragments in shared memory (thread-major
# slots), then compute the lane's 8 outputs with plain FFMA. Layout per
# ops_python generic_wmma_helper (tinygrad's executable semantics):
#   A[row][k] = A-frag elem jA(k) of thread row%16 + 16*g(k)
#   B[k][col] = B-frag elem jA(k) of thread col%16 + 16*g(k)
#   D: lane element e = D[(lane//16)*8 + e][lane%16]
#   jA(k) = k - [0,4,4,8][k//4];  g(k) = (k//4)%2
# Shared: per-warp 1KB slice — A slots [0,512), B slots [512,1024); slot =
# (thread%32)*16 + elem*2 bytes. All consumer addresses are affine in tid
# with compile-time constants, so no address arithmetic per access.
def _ja(k): return k - (0, 4, 4, 8)[k // 4]
def _gk(k): return (k // 4) % 2

def _wm_emit(ctx, x):
    if x.arg[0] != (16, 16, 16) or x.arg[1] != dtypes.half or x.arg[2] != 'AMD' or x.arg[3] != 32:
        raise RuntimeError(f'unsupported WMMA spec {x.arg!r}')
    def _fl(l): return [t for e in l for t in (e if isinstance(e, list) else [e])]
    A, B, C, D = _fl(ctx.r[x.src[0]]), _fl(ctx.r[x.src[1]]), _fl(ctx.r[x.src[2]]), _fl(ctx.r[x])
    assert len(A) == len(B) == 8 and len(C) == len(D) == 8, \
        (len(A), len(B), len(C), len(D),
         [(s.op.name, s.dtype.name, len(s.src)) for s in x.src[0].src],
         [('list', len(e)) if isinstance(e, list) else 'str' for e in ctx.r[x.src[0]]])
    if os.getenv('WMMA_SKIP'):   # isolation probe: D = C, no shared traffic
        return [f'mov.f32 {D[e]}, {C[e]};' for e in range(8)]
    tz, b0, b1 = ctx._wm_t[:3]
    ua, ub, uc, ud = ctx._wm_t[3:7]
    wq = ctx._wm_t[7:12]           # u64: base, stage, (+mode-specific temps)
    W = ctx._wm_warps
    L = []
    if ctx._wm_shared is None:
        ctx._wm_shared = [f'.shared .align 4 .b8 _wm[{2048*W}];']
    # stage: my 8 A halves -> slot (tid.x, j) at [0,512), B at [512,1024)
    # addr = tid.y*2048 + tid.x*16 + off
    L.append(f'mov.u32 {ua}, %tid.y;')
    L.append(f'mov.u32 {ub}, %tid.x;')
    L.append(f'mad.lo.u32 {ua}, {ua}, 128, {ub};')
    L.append(f'shl.b32 {ua}, {ua}, 4;')
    L.append(f'cvt.u64.u32 {wq[1]}, {ua};')
    L.append(f'mov.u64 {wq[0]}, _wm;')
    L.append(f'add.s64 {wq[1]}, {wq[0]}, {wq[1]};')
    for j in range(0, 8, 2):
        L.append(f'mov.b16 {b0}, {A[j]};')
        L.append(f'mov.b16 {b1}, {A[j+1]};')
        L.append(f'mov.b32 {tz}, {{{b0}, {b1}}};')
        L.append(f'st.shared.b32 [{wq[1]}+{j*2}], {tz};')
    for j in range(0, 8, 2):
        L.append(f'mov.b16 {b0}, {B[j]};')
        L.append(f'mov.b16 {b1}, {B[j+1]};')
        L.append(f'mov.b32 {tz}, {{{b0}, {b1}}};')
        L.append(f'st.shared.b32 [{wq[1]}+{512 + j*2}], {tz};')
    L.append('bar.warp.sync 0xffffffff;')

    if os.getenv('WMMA_FFMA'):
        # --- reference consumer: per-lane FFMA over shared (validated) ---
        bf = ctx._wm_t[12:28]          # f32 B-column cache
        tf0 = ctx._wm_t[28]
        # consume A rows: row m_e = (tid.x>>4)*8 + e -> thread slot base
        # wqA = warp*2048 + (tid.x>>4)*128; imm = e*16 + g(k)*256 + jA(k)*2
        # (warp base and lane component computed separately — mixing them puts
        # tid.x bytes into the base and odd b16 addresses = MISALIGNED_ADDRESS)
        L.append(f'mov.u32 {ub}, %tid.y;')
        L.append(f'shl.b32 {ub}, {ub}, 11;')
        L.append(f'mov.u32 {ua}, %tid.x;')
        L.append(f'shr.u32 {ua}, {ua}, 4;')
        L.append(f'shl.b32 {ua}, {ua}, 7;')
        L.append(f'add.u32 {uc}, {ub}, {ua};')
        L.append(f'cvt.u64.u32 {wq[2]}, {uc};')
        L.append(f'add.s64 {wq[2]}, {wq[0]}, {wq[2]};')
        # consume B column: n = tid.x%16 -> wqB = warp*2048 + 512 + (tid.x%16)*16
        L.append(f'mov.u32 {ua}, %tid.x;')
        L.append(f'and.b32 {ua}, {ua}, 15;')
        L.append(f'shl.b32 {ua}, {ua}, 4;')
        L.append(f'add.u32 {ud}, {ub}, {ua};')
        L.append(f'add.u32 {ud}, {ud}, 512;')
        L.append(f'cvt.u64.u32 {wq[3]}, {ud};')
        L.append(f'add.s64 {wq[3]}, {wq[0]}, {wq[3]};')
        # B column cache: B[k][n] = slot(n + 16*g(k), jA(k))
        for k in range(16):
            L.append(f'ld.shared.b16 {b0}, [{wq[3]}+{_gk(k)*256 + _ja(k)*2}];')
            L.append(f'cvt.f32.f16 {bf[k]}, {b0};')
        # outputs: D[m_e][n] = C + sum_k A[m_e][k]*B[k][n]
        for e in range(8):
            for k in range(16):
                L.append(f'ld.shared.b16 {b0}, [{wq[2]}+{e*16 + _gk(k)*256 + _ja(k)*2}];')
                L.append(f'cvt.f32.f16 {tf0}, {b0};')
                L.append(f'fma.rn.f32 {D[e]}, {tf0}, {bf[k]}, {C[e] if k == 0 else D[e]};')
        return L

    # --- tensor-core consumer: 4x mma.sync.m16n8k8 (sm_75's only f16 shape;
    # k-chained via the C accumulator, n-split). NV fragment lane layout per
    # the M16N8 family: A/B operand address = slot base + (l>>2)*16 + off(l&3),
    # D per (mma-op, elem): row = (l>>2)+8*(en>>1), col = (en&1)+2*(l&3)+8*op.
    a0, a1, a2, a3, q0, q1, q2, q3 = ctx._wm_t[11:19]      # b32 fragments
    wz, t00, t01, t02, t03, t10, t11, t12, t13 = ctx._wm_t[19:28]  # f32
    # r0 = l>>2, c = l&3; off(c) = 4c + 248*(u&1) - 8*(u>>1), u = c>>1
    L.append(f'mov.u32 {ub}, %tid.y;')
    L.append(f'shl.b32 {ub}, {ub}, 11;')                   # warp*2048
    L.append(f'mov.u32 {ua}, %tid.x;')
    L.append(f'shr.u32 {uc}, {ua}, 2;')                    # r0
    L.append(f'shl.b32 {ud}, {uc}, 4;')                    # r0*16
    L.append(f'and.b32 {ua}, {ua}, 3;')                    # c
    L.append(f'shr.u32 {a0}, {ua}, 1;')                    # u (borrow a0)
    L.append(f'and.b32 {a1}, {a0}, 1;')                    # u&1
    L.append(f'shr.u32 {a2}, {a0}, 1;')                    # u>>1
    L.append(f'shl.b32 {a3}, {ua}, 2;')                    # 4c
    L.append(f'mad.lo.u32 {ud}, {a1}, 248, {ud};')         # + r0*16 + 248*(u&1)
    L.append(f'shl.b32 {a0}, {a2}, 3;')                    # 8*(u>>1)
    L.append(f'sub.u32 {ud}, {ud}, {a0};')
    L.append(f'add.u32 {uc}, {ub}, {ud};')                 # + warp*2048
    L.append(f'cvt.u64.u32 {wq[2]}, {uc};')
    L.append(f'add.s64 {wq[2]}, {wq[0]}, {wq[2]};')        # wqA
    L.append(f'add.u32 {ud}, {uc}, 512;')
    L.append(f'cvt.u64.u32 {wq[3]}, {ud};')
    L.append(f'add.s64 {wq[3]}, {wq[0]}, {wq[3]};')        # wqB
    # fragment loads (A pairs: [0,128] k0-7, [8,136] k8-15; B same bases)
    L.append(f'ld.shared.b32 {a0}, [{wq[2]}+0];')
    L.append(f'ld.shared.b32 {a1}, [{wq[2]}+128];')
    L.append(f'ld.shared.b32 {a2}, [{wq[2]}+8];')
    L.append(f'ld.shared.b32 {a3}, [{wq[2]}+136];')
    L.append(f'ld.shared.b32 {q0}, [{wq[3]}+0];')
    L.append(f'ld.shared.b32 {q1}, [{wq[3]}+128];')
    L.append(f'ld.shared.b32 {q2}, [{wq[3]}+8];')
    L.append(f'ld.shared.b32 {q3}, [{wq[3]}+136];')
    L.append(f'mov.f32 {wz}, 0f00000000;')
    MM = 'mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32'
    L.append(f'{MM} {{{t00}, {t01}, {t02}, {t03}}}, {{{a0}, {a1}}}, {{{q0}}}, {{{wz}, {wz}, {wz}, {wz}}};')
    L.append(f'{MM} {{{t10}, {t11}, {t12}, {t13}}}, {{{a0}, {a1}}}, {{{q1}}}, {{{wz}, {wz}, {wz}, {wz}}};')
    L.append(f'{MM} {{{t00}, {t01}, {t02}, {t03}}}, {{{a2}, {a3}}}, {{{q2}}}, {{{t00}, {t01}, {t02}, {t03}}};')
    L.append(f'{MM} {{{t10}, {t11}, {t12}, {t13}}}, {{{a2}, {a3}}}, {{{q3}}}, {{{t10}, {t11}, {t12}, {t13}}};')
    # D reshuffle: store NV layout to shared [1024,2048), reload RDNA4 layout
    # store addr = 1024 + (l>>2)*64 + 8*(l&3) + imm; reload = 1024 + (l>>4)*512 + (l&15)*4 + e*64
    L.append(f'mov.u32 {ua}, %tid.x;')
    L.append(f'shr.u32 {ub}, {ua}, 2;')
    L.append(f'shl.b32 {ub}, {ub}, 6;')                    # (l>>2)*64
    L.append(f'and.b32 {ua}, {ua}, 3;')
    L.append(f'shl.b32 {ua}, {ua}, 3;')                    # 8*(l&3)
    L.append(f'add.u32 {uc}, {ub}, {ua};')
    L.append(f'mov.u32 {ua}, %tid.y;')
    L.append(f'mad.lo.u32 {uc}, {ua}, 2048, {uc};')
    L.append(f'add.u32 {uc}, {uc}, 1024;')
    L.append(f'cvt.u64.u32 {wq[2]}, {uc};')
    L.append(f'add.s64 {wq[2]}, {wq[0]}, {wq[2]};')        # wqD-store
    for _v, _imm in ((t00, 0), (t01, 4), (t02, 512), (t03, 516),
                     (t10, 32), (t11, 36), (t12, 544), (t13, 548)):
        L.append(f'st.shared.f32 [{wq[2]}+{_imm}], {_v};')
    L.append('bar.warp.sync 0xffffffff;')
    L.append(f'mov.u32 {ua}, %tid.x;')
    L.append(f'shr.u32 {ub}, {ua}, 4;')
    L.append(f'shl.b32 {ub}, {ub}, 9;')                    # (l>>4)*512
    L.append(f'and.b32 {ua}, {ua}, 15;')
    L.append(f'shl.b32 {ua}, {ua}, 2;')                    # (l&15)*4
    L.append(f'add.u32 {uc}, {ub}, {ua};')
    L.append(f'mov.u32 {ua}, %tid.y;')
    L.append(f'mad.lo.u32 {uc}, {ua}, 2048, {uc};')
    L.append(f'add.u32 {uc}, {uc}, 1024;')
    L.append(f'cvt.u64.u32 {wq[2]}, {uc};')
    L.append(f'add.s64 {wq[2]}, {wq[0]}, {wq[2]};')        # wqD-load
    for e in range(8):
        L.append(f'ld.shared.f32 {a0}, [{wq[2]}+{e*64}];')
        L.append(f'add.f32 {D[e]}, {a0}, {C[e]};')
    return L

_wm_pm = PatternMatcher([(UPat(Ops.WMMA, name='x'), lambda ctx, x: _wm_emit(ctx, x))])
_ptxmod.string_rewrite = _wm_pm + _ptxmod.string_rewrite

# 1d. render(): vector CAST, STACK flatten, widening loads / narrowing stores
_msrc = _tw.dedent(_inspect.getsource(_ptxmod.PTXRenderer.render))
_OLD1 = '    if prefix: r[u] = ssa(prefix, u, dtype)'
_NEW1 = '''    if prefix:
        if (u.op in (Ops.CAST,) or u.op in GroupOp.ALU) and u.max_numel() > 1:
            if any(isinstance(r.get(s), list) for s in u.src):
                r[u] = [ssa(prefix, u, dtype) for _ in range(u.max_numel())]
            else: r[u] = ssa(prefix, u, dtype)
        else: r[u] = ssa(prefix, u, dtype)
        if u.op is Ops.INDEX and len(u.src) >= 3 and u.src[0].op is Ops.PARAM and _pimg(u.src[0]) is not None:
            if not hasattr(self, '_img_sc'): self._img_sc = {}
            if not hasattr(self, '_img_valid'): self._img_valid = {}
            self._img_sc[id(u)] = (ssa('imgf', u, 's32'), ssa('imgf', u, 's32'),
                                   [ssa('imgv', u, 'pred') for _ in range(9)])'''
_OLD2 = '    l: str|list[str]|None = string_rewrite.rewrite(u, ctx=self)'
_NEW2 = '''    if u.op is Ops.CAST and isinstance(r[u], list):
        _a = u.src[0]
        _ars = r[_a] if isinstance(r[_a], list) else [r[_a]] * len(r[u])
        kernel.extend([f"cvt{modifier(u.dtype, _a.dtype)}.{self.cast_types[u.dtype]}.{self.cast_types[_a.dtype]} {rd}, {rs};"
                       for rd, rs in zip(r[u], _ars)])
        continue
    if u.op is Ops.CAST and isinstance(r[u.src[0]], list):
        _a = u.src[0]
        kernel.append(f"cvt{modifier(u.dtype, _a.dtype)}.{self.cast_types[u.dtype]}.{self.cast_types[_a.dtype]} {r[u]}, {r[_a][0]};")
        continue
    if u.op in GroupOp.ALU and isinstance(r[u], list):
        _srcs = []
        for _v in u.src:
            _rv = r[_v]
            _srcs.append(_rv if isinstance(_rv, list) else [_rv] * len(r[u]))
        _dt = u.src[0].dtype if u.op in (Ops.CMPLT, Ops.CMPNE, Ops.CMPEQ) else u.dtype
        _nm = self.types[_dt]
        _out = []
        for _i in range(len(r[u])):
            _args = [s[_i] if isinstance(s, list) else s for s in _srcs]
            _out.append(asm_for_op[u.op](r[u][_i], *_args, _dt, _nm))
        kernel.extend(_out)
        continue
    if u.op is Ops.STORE and u.src[0].addrspace == AddrSpace.REG and isinstance(r[u.src[1]], list):
        _loc = r[u.src[0]]
        _rv = r[u.src[1]]
        _buf = u.src[0].src[0]
        while _buf.op is Ops.AFTER: _buf = _buf.src[0]
        _rb = r.get(_buf)
        _k = _rb.index(_loc) if isinstance(_rb, list) and _loc in _rb else 0
        _vd = u.src[1].dtype
        kernel.append(f"mov.{'pred' if _vd == dtypes.bool else 'b' + self.types[_vd][1:]} {_loc}, {_rv[_k]};")
        continue
    if u.op is Ops.LOAD and u.src[0].op in (Ops.INDEX, Ops.SHRINK) and u.src[0].addrspace != AddrSpace.REG:
        _p = _pbuf(u.src[0])
        if _p is not None and (u.max_numel() > 1 or (_p.dtype != u.dtype and dtypes.is_float(_p.dtype) and dtypes.is_float(u.dtype))) and not (u.dtype == dtypes.bool):
            _bd = _p.dtype; _n = u.max_numel()
            if _n > 4 and _pimg(u.src[0]) is not None: _n = 4  # image: one texel per thread
            _dsts = r[u] if isinstance(r[u], list) else [r[u]]
            if len(_dsts) != _n: _dsts = ([_dsts[0]] * _n) if _dsts else [ssa('val', u)] * _n
            _widen = _bd != u.dtype
            _tmps = [ssa('ldt', dtype=self.types[_bd]) for _ in range(_n)] if _widen else _dsts
            _base = self.r[u.src[0]]
            _gate = r[u.src[2]] if len(u.src) >= 3 else None
            if _gate is not None:
                _al = r[u.src[1]]
                _al = _al if isinstance(_al, list) else [_al]
                if len(_al) != _n:
                    print(f'[gatefix] {name}: wide alt regn {len(_al)} != {_n} -> zero-init fallback (comma semantics)')
                    _al = None
            else:
                _al = None
            _lds = []
            _done = 0
            while _done < _n:
                if _n - _done >= 4:
                    _lds.append(f"ld.{mem_type(u.src[0])}.v4.{self.mem_types[_bd]} {{{', '.join(_tmps[_done:_done+4])}}}, [{_base}+{_done*_bd.itemsize}];")
                    _done += 4
                else:
                    _lds.append(f"ld.{mem_type(u.src[0])}.{self.mem_types[_bd]} {_tmps[_done]}, [{_base}+{_done*_bd.itemsize}];")
                    _done += 1
            _valid = getattr(self, '_img_valid', {}).get(id(u.src[0]))
            if _gate is not None:
                if _al is None:
                    kernel.extend([f"mov.{self.mem_types[u.dtype]} {v}, {render_val(0, u.dtype)};" for v in _dsts])
                kernel.extend([f"@{_gate} {ln}" for ln in _lds])
                if _widen:
                    for _i in range(_n):
                        kernel.append(f"@{_gate} cvt{modifier(u.dtype, _bd)}.{self.types[u.dtype]}.{self.types[_bd]} {_dsts[_i]}, {_tmps[_i]};")
                if _al is not None:
                    for _i in range(_n):
                        kernel.append(f"@!{_gate} mov.{'b16' if u.dtype.itemsize == 2 else self.types[u.dtype]} {_dsts[_i]}, {_al[_i]};")
                if _valid is not None:
                    _nv = ssa('nv', None, 'pred'); _mz = ssa('mz', None, 'pred')
                    kernel.append(f"not.pred {_nv}, {_valid};")
                    kernel.append(f"and.pred {_mz}, {_gate}, {_nv};")
                    for _i in range(_n):
                        kernel.append(f"@{_mz} mov.{'b16' if u.dtype.itemsize == 2 else self.types[u.dtype]} {_dsts[_i]}, {render_val(0, u.dtype)};")
            else:
                if _widen and _valid is not None:
                    kernel.extend([f"mov.{self.mem_types[u.dtype]} {v}, {render_val(0, u.dtype)};" for v in _dsts])
                kernel.extend(_lds)
                if _widen:
                    for _i in range(_n):
                        _cvt = f"cvt{modifier(u.dtype, _bd)}.{self.types[u.dtype]}.{self.types[_bd]} {_dsts[_i]}, {_tmps[_i]};"
                        kernel.append(f"@{_valid} {_cvt}" if _valid is not None else _cvt)
                if _valid is not None and not _widen:
                    for _i in range(_n):
                        kernel.append(f"@!{_valid} mov.{'b16' if u.dtype.itemsize == 2 else self.types[u.dtype]} {_dsts[_i]}, {render_val(0, u.dtype)};")
            continue
    if u.op is Ops.STORE and len(u.src) >= 2 and u.src[0].op in (Ops.INDEX, Ops.SHRINK) and u.src[0].addrspace != AddrSpace.REG:
        _p = _pbuf(u.src[0])
        _v = u.src[1]
        _narrow = _p is not None and _p.dtype != _v.dtype and dtypes.is_float(_p.dtype) and dtypes.is_float(_v.dtype)
        if _p is not None and (_v.max_numel() > 1 or _narrow):
            _vr = r[_v]; _vr = _vr if isinstance(_vr, list) else [_vr]
            _vr = [x for c in _vr for x in (c if isinstance(c, list) else [c])]
            if len(_vr) > 4 and _pimg(u.src[0]) is not None: _vr = _vr[:4]  # image: one texel per thread
            _n = len(_vr)
            _tmps = [ssa('stt', dtype=self.types[_p.dtype]) for _ in range(_n)] if _narrow else _vr
            if _narrow:
                for _i in range(_n):
                    kernel.append(f"cvt{modifier(_p.dtype, _v.dtype)}.{self.types[_p.dtype]}.{self.types[_v.dtype]} {_tmps[_i]}, {_vr[_i]};")
            _base = self.r[u.src[0]]
            _gate = r[u.src[2]] if len(u.src) >= 3 else None
            _done = 0
            while _done < _n:
                if _n - _done >= 4:
                    _ln = f"st.{mem_type(u.src[0])}.v4.{self.mem_types[_p.dtype]} [{_base}+{_done*_p.dtype.itemsize}], {{{', '.join(_tmps[_done:_done+4])}}};"
                    _done += 4
                else:
                    _ln = f"st.{mem_type(u.src[0])}.{self.mem_types[_p.dtype]} [{_base}+{_done*_p.dtype.itemsize}], {_tmps[_done]};"
                    _done += 1
                kernel.append(f"@{_gate} {_ln}" if _gate is not None else _ln)
            continue
    l: str|list[str]|None = string_rewrite.rewrite(u, ctx=self)'''
_OLD3 = '    if u.op is Ops.STACK:\n      r[u] = [cast(str,r[x]) for x in u.src]\n      continue'
_NEW3 = '''    if u.op is Ops.STACK:
      _flat = []
      for x in u.src:
        _rx = r[x]
        _flat.extend(_rx) if isinstance(_rx, list) else _flat.append(_rx)
      r[u] = _flat
      continue'''
# image-texel ops carry the WHOLE image shape as logical numel; per-thread
# emission width is always ONE texel (4 lanes), like cstyle's read/write_imagef
_OLD4 = "    elif u.op is Ops.LOAD:\n      r[u] = [ssa('val', dtype=self.types[u.dtype]) for _ in range(u.max_numel())] if u.max_numel() > 1 else ssa('val', u)"
_NEW4 = """    elif u.op is Ops.LOAD:
      _n4 = u.max_numel()
      if _n4 > 4 and u.src and u.src[0].op in (Ops.INDEX, Ops.SHRINK) and _pimg(u.src[0]) is not None: _n4 = 4
      r[u] = [ssa('val', dtype=self.types[u.dtype]) for _ in range(_n4)] if _n4 > 1 else ssa('val', u)"""
assert _OLD1 in _msrc and _OLD2 in _msrc and _OLD3 in _msrc and _OLD4 in _msrc, 'render patch anchors missing'
# lane pick: INDEX/SHRINK over a VALUE whose registers are a list (vector load /
# ALU result / stack) with a CONST index = pick that lane, no code emitted.
# (vanilla only handles this for REG/ALU addrspace; IMAGE=1 graphs also pick
# lanes of global loads and computed vectors)
_OLD5 = '    if u.op is Ops.SPECIAL: r[u] = "%" + u.arg'
_NEW5 = '''    if u.op in (Ops.INDEX, Ops.SHRINK) and u.src[0] in r and isinstance(r[u.src[0]], list) \\
      and len(u.src) > 1 and u.src[1].op is Ops.CONST and u.src[0].addrspace not in (AddrSpace.REG, AddrSpace.ALU):
      r[u] = r[u.src[0]][u.src[1].val]
      continue
    if u.op is Ops.SPECIAL and isinstance(u.arg, str) and u.arg[0] == 'i':
      _d = int(u.arg[-1]) if u.arg[-1].isdigit() else 0
      _ls = (getattr(self, '_i_launch_local', [1, 1, 1]) + [1, 1, 1])[_d]
      _g, _t = ssa('cta', u, 'u32'), ssa('tid', u, 'u32')
      _dst = ssa('gid', u, 's32')
      kernel.append(f"mov.u32 {_g}, %ctaid.{chr(120 + _d)};")
      kernel.append(f"mov.u32 {_t}, %tid.{chr(120 + _d)};")
      kernel.append(f"mad.lo.s32 {_dst}, {_g}, {_ls}, {_t};")
      r[u] = _dst
      continue
    if u.op is Ops.SPECIAL: r[u] = "%" + u.arg'''
_OLD8 = '  kernel:list[str] = []\n  bufs = []'
_NEW8 = '''  kernel:list[str] = []
  bufs = []
  self._img_sc, self._img_valid = {}, {}   # per-kernel image-INDEX state'''
_OLD6 = '    if u.op is Ops.SPECIAL: kernel = [f".reg .u32 %{u.arg};"] + kernel'
_NEW6 = '''    if u.op is Ops.SPECIAL and not (isinstance(u.arg, str) and u.arg[0] == 'i'):
      kernel = [f".reg .u32 %{u.arg};"] + kernel'''
_OLD7 = '    kernel.extend([l] if isinstance(l, str) else l)'
_NEW7 = '''    kernel.extend([l] if isinstance(l, str) else l)
    if isinstance(l, str) and "['" in l:
      print(f'[GARBAGE] after uop {u.op} numel={u.max_numel()} srcs={[x.op for x in u.src]}: {l[:100]}')'''
_OLD9 = '''    elif u.op is Ops.WMMA:
      # registers for packing/unpacking input and acc
      self.wmma_r = [[ssa("wmma_in", dtype="b32") for _ in range(0, len(r[u.src[0]]), 4 // u.src[0].dtype.itemsize)],
                     [ssa("wmma_in", dtype="b32") for _ in range(0, len(r[u.src[1]]), 4 // u.src[0].dtype.itemsize)],
                     [ssa("wmma_acc", dtype="b32") for _ in range(0, len(r[u.src[2]]), 4 // u.dtype.itemsize)]]
      r[u] = [ssa("wmma", dtype=self.types[u.dtype]) for _ in range(u.max_numel())]'''
_NEW9 = '''    elif u.op is Ops.WMMA:
      r[u] = [ssa("wmma", dtype=self.types[u.dtype]) for _ in range(u.max_numel())]
      # WMMA lowering temps + per-kernel shared state (consumed by _wm_emit)
      if getattr(self, '_wm_for', None) != id(uops):
        self._wm_for = id(uops)
        self._wm_shared = None
        _lp = 1
        for _d in (getattr(self, '_i_launch_local', None) or [1, 1, 1]):
          _lp *= int(_d) if isinstance(_d, (int, float)) else 1
        self._wm_warps = max(1, _lp // 32)
        if os.getenv('WMMA_FFMA') or os.getenv('WMMA_SKIP'):
          self._wm_t = ([ssa('wmz', dtype='b32'), ssa('wmz', dtype='b16'), ssa('wmz', dtype='b16')] +
                        [ssa('wmu', dtype='u32') for _ in range(4)] + [ssa('wmq', dtype='u64') for _ in range(4)] +
                        [ssa('wmb', dtype='f32') for _ in range(16)] + [ssa('wmf', dtype='f32') for _ in range(2)])
        else:
          self._wm_t = ([ssa('wmz', dtype='b32'), ssa('wmz', dtype='b16'), ssa('wmz', dtype='b16')] +
                        [ssa('wmu', dtype='u32') for _ in range(4)] + [ssa('wmq', dtype='u64') for _ in range(4)] +
                        [ssa('wma', dtype='b32') for _ in range(4)] + [ssa('wmb', dtype='b32') for _ in range(4)] +
                        [ssa('wmz', dtype='f32')] + [ssa('wmt', dtype='f32') for _ in range(8)])'''
_OLDK = '+ kernel + ["ret;"]))'
_NEWK = '+ (getattr(self, \'_wm_shared\', None) or []) + kernel + ["ret;"]))'
assert _OLD5 in _msrc and _OLD6 in _msrc and _OLD7 in _msrc and _OLD8 in _msrc, 'render patch anchors missing'
assert _OLD9 in _msrc, 'wmma pre-pass anchor missing'
_msrc = _msrc.replace(_OLD1, _NEW1).replace(_OLD2, _NEW2).replace(_OLD3, _NEW3).replace(_OLD4, _NEW4) \
             .replace(_OLD5, _NEW5).replace(_OLD6, _NEW6).replace(_OLD7, _NEW7).replace(_OLD8, _NEW8) \
             .replace(_OLD9, _NEW9)
_ns = dict(_ptxmod.__dict__)
_ns['math'] = math
_ns['os'] = os
open('research/patched_render.py', 'w').write(_msrc)
exec(compile(_msrc, '<patched_ptx_render>', 'exec'), _ns)
_ns['_pbuf'] = _pbuf
_ns['_pimg'] = _pimg
_ptxmod.PTXRenderer.render = _ns['render']
_ksrc = _tw.dedent(_inspect.getsource(_ptxmod.PTXRenderer.render_kernel))
assert _OLDK in _ksrc, 'render_kernel anchor missing'
_ksrc = _ksrc.replace(_OLDK, _NEWK)
exec(compile(_ksrc, '<patched_ptx_render_kernel>', 'exec'), _ns)
_ptxmod.PTXRenderer.render_kernel = _ns['render_kernel']

# --- 2) device aliasing ----------------------------------------------------------
if TARGET == 'CL':
    # run kernels as NATIVE OpenCL: QCOMDevice/AMDDevice = the real CLDevice.
    # keep the pickled device strings ('QCOM:0') so buffers/kernels share one
    # context; only the class behind them changes.
    from tinygrad.runtime.ops_cl import CLDevice as _RealCL
    import tinygrad.runtime.ops_cl as _oclc
    from tinygrad.runtime.autogen import opencl as _cl
    import ctypes as _ct
    def _cl_offset(self, buf, size: int, offset: int):
        # OpenCL views = sub-buffers (CUDA does pointer arithmetic; cl_mem can't)
        region = _cl.cl_buffer_region()
        region.origin, region.size = offset, size
        st = _ct.c_long(0)
        sub = _cl.clCreateSubBuffer(buf, _cl.CL_MEM_READ_WRITE, _cl.CL_BUFFER_CREATE_TYPE_REGION,
                                    _ct.cast(_ct.byref(region), _ct.c_void_p), st)
        return _oclc.checked(sub, st)
    _oclc.CLAllocator._offset = _cl_offset

    # NVIDIA desktop OpenCL lacks cl_khr_image2d_from_buffer: back image args
    # with standalone images + explicit buffer<->image copies around the kernel
    from tinygrad.helpers import is_image_shape as _is_img
    _SRC_BY_NAME: dict[str, str] = {}
    import hashlib as _hl
    _CL_PROG_CACHE: dict[str, object] = {}
    from tinygrad.helpers import to_char_p_p as _tcpp
    def _clp_init(self, device, obj):
        # NV's clCreateProgramWithBinary+rebuild is flaky (-5); build straight
        # from source once and cache the cl_program object
        ct, cl = _ct, _cl
        self.dev, self.signature = device, obj.signature
        src = obj.lib.decode()
        self._img_dirs = [d for d, _ in re.findall(r'(read_only|write_only)\s+image2d_t\s+(data\d+)', src)]
        key = _hl.md5(src.encode()).hexdigest()
        if key not in _CL_PROG_CACHE:
            st = ct.c_int32()
            import time as _t
            program = _oclc.checked(cl.clCreateProgramWithSource(device.context, 1, _tcpp([src.encode()]), None, st), st)
            for _try in range(4):
                try:
                    _oclc.check(cl.clBuildProgram(program, 1, device.cl_dev, None, _oclc.BP_CB(), None))
                    break
                except Exception as _e:
                    print(f'[clbuild] retry {_try} for {obj.name} ({len(src)//1024}KB src): {str(_e)[:60]}', flush=True)
                    if _try == 3: raise
                    _t.sleep(1)
            _CL_PROG_CACHE[key] = program
        self.program = _CL_PROG_CACHE[key]
        st2 = ct.c_int32()
        self.kernel = _oclc.checked(cl.clCreateKernel(self.program, obj.name.encode(), st2), st2)
        self._cl_name = obj.name
    _oclc.CLProgram.__init__ = _clp_init

    _IMG_CACHE: dict = {}
    def _cl_call(self, *bufs, global_size=(1, 1, 1), local_size=(1, 1, 1), vals=(), wait=False, **kw):
        cl, ct = _cl, _ct
        q = self.dev.queue
        dirs = getattr(self, '_img_dirs', [])
        tmp = []      # (cl_mem img, cl_mem buf, is_write, w, h)
        ki = 0
        for i, (_, slot, dt, shape) in enumerate(self.signature):
            if slot < len(bufs):
                b = bufs[slot]
                if _is_img(shape):
                    fmt = cl.cl_image_format(cl.CL_RGBA, {2: cl.CL_HALF_FLOAT, 4: cl.CL_FLOAT}[dt.itemsize])
                    w, h = shape[1], shape[0]
                    key = (ct.cast(b, ct.c_void_p).value, dt.itemsize, w, h)
                    img = _IMG_CACHE.get(key)
                    if img is None:
                        desc = cl.cl_image_desc(cl.CL_MEM_OBJECT_IMAGE2D, w, h)
                        st = ct.c_int32()
                        img = _oclc.checked(cl.clCreateImage(self.dev.context, cl.CL_MEM_READ_WRITE, fmt, desc, None, st), st)
                        _IMG_CACHE[key] = img
                    is_wr = ki < len(dirs) and dirs[ki] == 'write_only'
                    ki += 1
                    if not is_wr:   # read_only: buffer -> image
                        _oclc.check(cl.clEnqueueCopyBufferToImage(q, b, img, 0, (cl.size_t * 3)(0, 0, 0),
                                                                  (cl.size_t * 3)(w, h, 1), 0, None, None))
                    _oclc.check(cl.clSetKernelArg(self.kernel, i, ct.sizeof(img), ct.byref(img)))
                    tmp.append((img, b, is_wr, w, h))
                else:
                    _oclc.check(cl.clSetKernelArg(self.kernel, i, ct.sizeof(b), ct.byref(b)))
            else:
                v = getattr(ct, f'c_int{dt.bitsize}')(vals[slot - len(bufs)])
                _oclc.check(cl.clSetKernelArg(self.kernel, i, ct.sizeof(v), ct.byref(v)))
        if local_size is not None:
            # exact thread counts: fractional grid dims encode g*l = exact
            gws = tuple(g * l for g, l in zip(global_size, local_size))
            assert all(float(x).is_integer() for x in gws), f'{getattr(self, "_cl_name", "?")}: non-integral gws {gws}'
            global_size = tuple(int(x) for x in gws)
            # only pass lws when it divides gws exactly (else let OpenCL pick;
            # such kernels never read local ids)
            local_size = local_size if all(g % l == 0 for g, l in zip(global_size, local_size)) else None
        _oclc.check(cl.clEnqueueNDRangeKernel(q, self.kernel, len(global_size), None,
                                              (cl.size_t * len(global_size))(*global_size),
                                              (cl.size_t * len(local_size))(*local_size) if local_size else None,
                                              0, None, None))
        for img, b, is_wr, w, h in tmp:
            if is_wr:      # write_only: image -> buffer
                _oclc.check(cl.clEnqueueCopyImageToBuffer(q, img, b, (cl.size_t * 3)(0, 0, 0),
                                                          (cl.size_t * 3)(w, h, 1), 0, 0, None, None))
        if os.getenv('CLDEBUG'):   # per-kernel fence: find the faulting kernel
            try:
                _oclc.check(cl.clFinish(q))
            except Exception as _e:
                print(f'[CLFAULT] kernel {getattr(self, "_cl_name", "?")}: {_e}', flush=True)
                raise
        if wait: self.dev.synchronize()
        return None
    _oclc.CLProgram.__call__ = _cl_call
    for _x in ('qcom', 'amd'):
        _m = _types.ModuleType(f'tinygrad.runtime.ops_{_x}')
        setattr(_m, _x.upper() + 'Device', _RealCL)
        sys.modules[_m.__name__] = _m
else:
    class _AliasDevice:
        def __init__(self, device):
            self.device = device
            self._real = _DevMod[TARGET]
        def __getattr__(self, name): return getattr(self._real, name)

    for _x in ('qcom', 'amd'):
        _m = _types.ModuleType(f'tinygrad.runtime.ops_{_x}')
        setattr(_m, _x.upper() + 'Device', _AliasDevice)
        sys.modules[_m.__name__] = _m

# --- 3) graph fixups (non-addressing) ---------------------------------------------
pm_div = PatternMatcher([
    # rcp in f32 then round back (pm_rcp semantics): the FDIV-created half
    # RECIPROCAL would otherwise render as rcp.approx.f16, which ptxas
    # rejects on sm_75, and rewrite_uop_list never revisits children
    (UPat(Ops.FDIV, name="dv"),
     lambda dv: dv.src[0] * UOp(Ops.RECIPROCAL, dtypes.float32, (dv.src[1].cast(dtypes.float32),)).cast(dv.dtype)),
])
pm_rcp = PatternMatcher([
    (UPat(Ops.RECIPROCAL, dtype=dtypes.half, name="u"),
     lambda u: UOp(Ops.RECIPROCAL, dtypes.float32, (u.src[0].cast(dtypes.float32),)).cast(dtypes.half)),
])
pm_nohalf = PatternMatcher([
    (UPat.var('x', dtype=dtypes.bool).ne(UPat.var('y')), lambda x, y: (x^y)),
    (UPat.var('x', dtype=dtypes.bool).alu(Ops.CMPEQ, UPat.var('y')), lambda x, y: (x^y)^True),
    (UPat.var('x', dtype=dtypes.bool) < UPat.var('y'), lambda x, y: (x^True)&y),
    (UPat.var("x") << UPat.var("y"), lambda x, y: UOp(Ops.SHL, x.dtype, (x, y.cast(dtypes.uint))) if y.dtype != dtypes.uint else None),
    (UPat.var("x") >> UPat.var("y"), lambda x, y: UOp(Ops.SHR, x.dtype, (x, y.cast(dtypes.uint))) if y.dtype != dtypes.uint else None),
    (UPat(Ops.LOAD, dtypes.bool, src=(UPat(name="idx"),), name="x", allow_any_len=True),
     lambda x, idx: UOp(x.op, dtypes.uint8, x.src[0:1] + ((x.src[1].cast(dtypes.uint8),) if len(x.src) >= 2 else ()) + x.src[2:]).cast(dtypes.bool)
        if idx.addrspace != _AS.REG else None),
    (UPat(Ops.STORE, src=(UPat(name="idx"), UPat(dtype=dtypes.bool)), name="x", allow_any_len=True),
     lambda x, idx: UOp(x.op, src=(x.src[0], x.src[1].cast(dtypes.uint8)) + x.src[2:]) if idx.addrspace != _AS.REG else None),
])
pm_bool = PatternMatcher([
    (UPat(Ops.CAST, dtype=dtypes.bool, name="x"),
     lambda x: None if x.src[0].dtype == dtypes.bool
        else x.src[0].ne(UOp(Ops.CONST, x.src[0].dtype, arg=0))),
])
def _alu_fix(a):
    if a.dtype not in (dtypes.float, dtypes.half): return None
    def needs(s): return dtypes.is_float(s.dtype) and s.dtype != a.dtype
    if any(needs(s) for s in a.src):
        return a.replace(src=tuple(s.cast(a.dtype) if needs(s) else s for s in a.src))
    return None
pm_alu = PatternMatcher([(UPat(GroupOp.ALU, name="a"), _alu_fix)])

# IMAGE=1 2D access: INDEX(param, row, texel_x) over (A,B,C) image params.
# Flatten to a flat TEXEL index (row*B + x); the renderer scales by C*itemsize.
# Clamp row/texel to the image bounds FIRST — comma's OpenCL sampler uses
# CLK_ADDRESS_CLAMP, so conv halo reads clamp to the edge instead of wrapping
# into the neighbouring row (raw flat addressing reads garbage there).
def _img_clamped(x, dt, lo, hi):
    loc = UOp(Ops.CONST, dt, arg=lo)
    hic = UOp(Ops.CONST, dt, arg=hi)
    x = UOp(Ops.WHERE, dt, (UOp(Ops.CMPLT, dtypes.bool, (x, loc)), loc, x))    # max(x, lo)
    return UOp(Ops.WHERE, dt, (UOp(Ops.CMPLT, dtypes.bool, (x, hic)), x, hic)) # min(x, hi)

_IMGLOAD_IDX = set()   # ids of INDEX nodes that are the address of an image LOAD

def _rule_index(ix):
    if len(ix.src) < 3: return None
    p = ix.src[0]
    if p.op is not Ops.PARAM: return None
    dims = _pimg(p)
    if dims is None: return None
    A, B, _C = dims
    y, x = ix.src[1], ix.src[2]
    dt = y.dtype
    xc = _img_clamped(x, dt, 0, B - 1)
    yc = _img_clamped(y, dt, 0, A - 1)
    flat = yc * UOp(Ops.CONST, dt, arg=B) + xc
    return ix.replace(src=(p, flat) + tuple(ix.src[3:]))
pm_storage = PatternMatcher([(UPat(Ops.INDEX, name="ix"), _rule_index)])

# tier 1 (comma-LINEAR): image LOAD with INDEX(param(A,B,C), x, y) — OpenCL
# CLK_ADDRESS_CLAMP semantics: out-of-range reads return the BORDER color
# (zeros), so mask the loaded texel to zero when the original coords are OOB;
# the address itself stays edge-clamped to keep memory access in bounds
def _rule_img2d(ld):
    if ld.op is not Ops.LOAD or len(ld.src) != 1: return None
    if ld.max_numel() > 4: return None   # single-texel loads only (render width)
    ix = ld.src[0]
    if ix.op is not Ops.INDEX or len(ix.src) < 3: return None
    p = ix.src[0]
    if p.op is not Ops.PARAM: return None
    dims = _pimg(p)
    if dims is None: return None
    A, B, _C = dims
    y, x = ix.src[1], ix.src[2]
    dt = y.dtype
    xc = _img_clamped(x, dt, 0, B - 1)
    yc = _img_clamped(y, dt, 0, A - 1)
    flat = UOp(Ops.ADD, dt, (UOp(Ops.MUL, dt, (yc, UOp(Ops.CONST, dt, arg=B))), xc))
    new_ix = ix.replace(src=(p, flat) + tuple(ix.src[3:]))
    new_load = ld.replace(src=(new_ix,) + tuple(ld.src[1:]))
    b0 = UOp(Ops.CONST, dt, arg=0)
    bt = UOp(Ops.CONST, dtypes.bool, arg=True)
    ge_x = UOp(Ops.XOR, dtypes.bool, (UOp(Ops.CMPLT, dtypes.bool, (x, b0)), bt))
    lt_x = UOp(Ops.CMPLT, dtypes.bool, (x, UOp(Ops.CONST, dt, arg=B)))
    ge_y = UOp(Ops.XOR, dtypes.bool, (UOp(Ops.CMPLT, dtypes.bool, (y, b0)), bt))
    lt_y = UOp(Ops.CMPLT, dtypes.bool, (y, UOp(Ops.CONST, dt, arg=A)))
    valid = UOp(Ops.MUL, dtypes.bool, (UOp(Ops.MUL, dtypes.bool, (ge_x, lt_x)), UOp(Ops.MUL, dtypes.bool, (ge_y, lt_y))))
    zero = UOp(Ops.CONST, ld.dtype, arg=0)
    return UOp(Ops.WHERE, ld.dtype, (valid, new_load, zero))
pm_img2d = PatternMatcher([(UPat(Ops.LOAD, name="ld"), _rule_img2d)])

# --- 4) pkl loading ----------------------------------------------------------------
def load_oob_from(data):
    n = struct.unpack('<q', data[:8])[0]
    stream = io.BytesIO(data[8:8+n]); buf = io.BytesIO(data[8+n:])
    def buffers():
        while True:
            h = buf.read(8)
            if not h: return
            (nb,) = struct.unpack('<q', h)
            yield pickle.PickleBuffer(buf.read(nb))
    return pickle.load(stream, buffers=buffers())

def pi_name(u):
    return re.sub(r'\x1b\[[0-9;]*m', '', getattr(u.arg, 'name', str(u.arg)))

# offline PTX->cubin with the external ptxas (driver's in-process JIT hangs on
# some kernels; ptxas is fast and reports real errors)
import hashlib, subprocess as _sp
_PTXAS = os.path.abspath(os.path.join(sys.prefix, 'Lib', 'site-packages', 'nvidia', 'cuda_nvcc', 'bin', 'ptxas.exe'))
_ARCH = _DevMod[TARGET].renderer.target.arch
os.makedirs('ptxas_cache', exist_ok=True)
def ptxas_compile(ptx: str) -> bytes:
    _v = int(_ARCH[3:])
    ptx = ptx.replace('TARGET', _ARCH).replace('VERSION', '8.7' if _v >= 120 else ('7.8' if _v >= 89 else '7.5'))
    h = hashlib.sha1(ptx.encode()).hexdigest()[:16]
    cub = os.path.join('ptxas_cache', f'{_ARCH}_{h}.cubin')
    if not os.path.exists(cub):
        srcf = os.path.join('ptxas_cache', f'{h}.ptx')
        open(srcf, 'w').write(ptx)
        r = _sp.run([_PTXAS, f'-arch={_ARCH}', '-o', cub, srcf], capture_output=True, text=True, timeout=120)
        if r.returncode != 0:
            print(r.stdout[-2000:]); print(r.stderr[-2000:])
            raise RuntimeError(f'ptxas failed; PTX saved to {srcf}')
    return open(cub, 'rb').read()

# --- k-split for serial-K WMMA kernels --------------------------------------
# r_6_2 walks a 384-trip K-reduction loop (loads at ridx k-offsets, mma-accum
# into 128 loop-carried f32 fragments, single f16 store burst AFTER the loop)
# on just 4 warps/block. Split the loop across tid.z (Z=2): each z-half does
# every other k, the z=1 partials go through a 16KB shared buffer, z=0 adds
# them into its fragments and runs the unchanged epilogue. f32 summation
# order changes (reassociation-class noise, as everywhere else here).
# The _wm staging slices (indexed by %tid.y) are remapped to the linear warp
# id and doubled: with Z=2 the two z-warps of a row run concurrently.
def _ksplit_ptx(ptx: str, Z: int) -> str:
    if Z != 2:   # only the z-pair scheme is implemented
        return ptx
    lines = ptx.split('\n')
    try:
        i_init = next(i for i, l in enumerate(lines) if 'mov.u32' in l and '%ridx_s32_0, -1;' in l)
        i_step = next(i for i, l in enumerate(lines) if 'add.s32' in l and '%ridx_s32_0, %ridx_s32_0, 1;' in l)
        i_decl = next(i for i, l in enumerate(lines) if '.shared' in l and '_wm[' in l)
        i_back = next(i for i, l in enumerate(lines) if 'bra' in l and 'LOOP_ridx_s32_0' in l)
        i_ret = next(i for i, l in enumerate(lines) if l.strip() == 'ret;')
    except StopIteration:
        return ptx
    if '%tid.z' in ptx:
        return ptx
    m = re.search(r'\.reg\s+\.f32\s+%reg_f32_<(\d+)>;', ptx)
    if not m:
        return ptx
    NACC = int(m.group(1))
    if NACC * 32 * 4 > 16384:   # shared chunk budget
        return ptx
    # remap %tid.y -> linear warp id inside _wm contexts, double _wm
    out = []
    for l in lines:
        if '%tid.y' in l and re.search(r'(%wm\w+)\s*,\s*%tid\.y', l):
            l = l.replace('%tid.y', '%wslid')
        out.append(l)
    wm = int(re.search(r'_wm\[(\d+)\]', out[i_decl]).group(1))
    out[i_decl] = f'\t.shared\t\t.align 4 .b8 _wm[{wm*2}];\n\t.shared\t\t.align 4 .b8 _ksp[16384];'
    i_init = next(i for i, l in enumerate(out) if 'mov.u32' in l and '%ridx_s32_0, -1;' in l)
    i_step = next(i for i, l in enumerate(out) if 'add.s32' in l and '%ridx_s32_0, %ridx_s32_0, 1;' in l)
    out[i_init] = (f'\tmov.u32\t\t%ksz, %tid.z;\n'
                   f'\tmov.u32\t\t%wslid, %tid.y;\n'
                   f'\tmad.lo.u32\t%wslid, %ksz, 4, %wslid;\n'
                   f'\tmov.u32\t\t%ridx_s32_0, %ksz;\n'
                   f'\tadd.s32\t\t%ridx_s32_0, %ridx_s32_0, -{Z};')
    out[i_step] = lines[i_step].replace(', 1;', f', {Z};')
    # shared partial exchange, chunked: [tid.y][tid.x][32] floats per round,
    # NACC/32 rounds; both warps hit both barriers of every round
    NC = 32
    nchunk = (NACC + NC - 1) // NC
    ins = ['\tmov.u32\t\t%kso, %tid.x;',
           '\tmov.u32\t\t%wsz, %tid.y;',
           '\tmad.lo.u32\t%kso, %wsz, 32, %kso;',
           f'\tmul.lo.u32\t%kso, %kso, {NC*4};',
           '\tcvt.u64.u32\t%ksp, %kso;',
           '\tmov.u64\t\t%ksp2, _ksp;',
           '\tadd.s64\t\t%ksp, %ksp2, %ksp;',
           '\tsetp.gt.u32\t%kp, %ksz, 0;']
    for c in range(nchunk):
        base = c * NC
        ins += [f'\t@%kp\tst.shared.f32\t[%ksp+{j*4}], %reg_f32_{base+j};' for j in range(NC)]
        ins.append('\tbar.sync\t\t0;')
        ins += [f'\t@!%kp\tld.shared.f32\t%kt, [%ksp+{j*4}];\n\t@!%kp\tadd.f32\t\t%reg_f32_{base+j}, %reg_f32_{base+j}, %kt;' for j in range(NC)]
        ins.append('\tbar.sync\t\t0;')
    text = '\n'.join(out)
    text = text.replace(lines[i_back], lines[i_back] + '\n' + '\n'.join(ins) + '\n\t@%kp\tbra\t\tKSKIP;', 1)
    text = text.replace('\tret;', '\tKSKIP:\n\tret;', 1)
    text = text.replace('.maxntid 128', '.maxntid 256')
    i_reg = text.find('\t.reg')
    text = text[:i_reg] + '\t.reg \t.u32 %wslid;\n\t.reg \t.u32 %wsz;\n\t.reg \t.u32 %ksz;\n\t.reg \t.u32 %kso;\n\t.reg \t.u64 %ksp;\n\t.reg \t.u64 %ksp2;\n\t.reg \t.f32 %kt;\n\t.reg \t.pred %kp;\n' + text[i_reg:]
    return text

# --- z-split for serial-loop WMMA kernels ----------------------------------
# r_6_2 / r_2_24 walk a 384-trip do-while loop whose iterations write
# DISJOINT ridx-addressed output tiles (no cross-iteration accumulation),
# with only 4 warps per block doing tile work (12 blocks x 128 threads on
# 48 SMs). Splitting the loop across the UNUSED tid.z dimension gives each
# tile the identical mma sequence -> outputs stay bit-identical.
# The _wm shared staging slices are indexed by %tid.y inside %wm* register
# contexts; remap those to the linear warp id (tid.y + 4*tid.z) and scale
# the shared declaration accordingly.
def _zsplit_ptx(ptx: str, Z: int) -> str:
    if Z <= 1: return ptx
    lines = ptx.split('\n')
    try:
        i_init = next(i for i, l in enumerate(lines) if 'mov.u32' in l and '%ridx_s32_0, -1;' in l)
        i_step = next(i for i, l in enumerate(lines) if 'add.s32' in l and '%ridx_s32_0, %ridx_s32_0, 1;' in l)
        i_decl = next(i for i, l in enumerate(lines) if '.shared' in l and '_wm[' in l)
        i_loop = next(i for i, l in enumerate(lines) if l.strip() == 'LOOP_ridx_s32_0:')
        i_back = next(i for i, l in enumerate(lines) if 'bra' in l and 'LOOP_ridx_s32_0' in l)
    except StopIteration:
        return ptx
    # safety: iterations must write global outputs inside the loop (disjoint
    # tiles); a loop whose stores happen only after it accumulates across
    # iterations cannot be z-split without a reduction
    if not any('st.global' in l for l in lines[i_loop:i_back]):
        return ptx
    if '%tid.z' in ptx:   # tid.z already meaningfully used: unsafe
        return ptx
    import re as _re
    m = _re.search(r'_wm\[(\d+)\]', lines[i_decl])
    if not m: return ptx
    warpbytes = int(m.group(1)) // 4   # per-warp slice (rendered for 4 warps)
    # remap %tid.y -> linear warp id inside _wm register contexts only
    out = []
    for i, l in enumerate(lines):
        if '%tid.y' in l and _re.search(r'(%wm\w+)\s*,\s*%tid\.y', l):
            l = l.replace('%tid.y', '%wslid')
        out.append(l)
    out[i_decl] = f'\t.shared\t\t.align 4 .b8 _wm[{warpbytes*4*Z}];'
    i_init = next(i for i, l in enumerate(out) if 'mov.u32' in l and '%ridx_s32_0, -1;' in l)
    i_step = next(i for i, l in enumerate(out) if 'add.s32' in l and '%ridx_s32_0, %ridx_s32_0, 1;' in l)
    out[i_init] = (f'\tmov.u32\t\t%wsz, %tid.z;\n'
                   f'\tmov.u32\t\t%wslid, %tid.y;\n'
                   f'\tmad.lo.u32\t%wslid, %wsz, 4, %wslid;\n'
                   f'\tmov.u32\t\t%ridx_s32_0, %tid.z;\n'
                   f'\tadd.s32\t\t%ridx_s32_0, %ridx_s32_0, -{Z};')
    out[i_step] = lines[i_step].replace(', 1;', f', {Z};')
    # wslid/wsz register declarations alongside the others
    i_reg = next(i for i, l in enumerate(out) if l.startswith('\t.reg'))
    out[i_reg] = ('\t.reg \t.u32 %wslid;\n\t.reg \t.u32 %wsz;\n') + out[i_reg]
    text = '\n'.join(out).replace('.maxntid 128', f'.maxntid {128*Z}')
    return text


# r_32_32_3_12288_32 loops k=0..12287 step 1 on ONE warp (gs=32x1, ls=32x1):
# ~8ms of pure serialization. Split the loop across W warps (local y 1->W),
# each warp strides k by W; partial sums go through shared, warp 0 reduces.
# The only semantic change is f32 summation order (same noise class as the
# mma reassociation, measured max 1.5e-5 absolute on real inputs).
def _warp_split_ptx(ptx: str, W: int) -> str:
    lines = ptx.split('\n')
    KNAME = 'r_32_32_3_12288_32'
    if f'.entry {KNAME} ' not in ptx and f'.entry {KNAME}(' not in ptx:
        return ptx
    defs = {}
    for i, ln in enumerate(lines):
        m = re.match(r'\s*([\w.]+)\s+(%[\w.]+)\s*,', ln)
        if m and m.group(2) not in defs: defs[m.group(2)] = i
    op_of = lambda i: re.match(r'\s*([\w.]+)', lines[i]).group(1)
    def srcs_of(i):
        body = lines[i].split('\t', 1)[1].rsplit(';', 1)[0]
        return re.findall(r'%[\w.]+', body)
    DESCEND = {'add.f32', 'mul.f32', 'sub.f32', 'fma.rn.f32',
               'mul.f16', 'add.f16', 'cvt.f32.f16', 'cvt.rn.f16.f32'}
    def bases_from(reg, seen):
        if reg in seen: return []
        seen.add(reg)
        i = defs.get(reg)
        if i is None: return [reg]
        if op_of(i) in DESCEND:
            out = []
            for s in srcs_of(i):
                if s.startswith('%const_'): continue
                out += bases_from(s, seen)
            return out
        return [reg]
    bases = []
    for s in ('%alu_f32_190', '%alu_f32_191', '%alu_f32_189'):
        for b in bases_from(s, set()):
            if b.startswith('%reg_f32') and b not in bases: bases.append(b)
    if len(bases) != 96:
        print(f'[warp-split] unexpected chain bases ({len(bases)}); skipping')
        return ptx
    NB = len(bases)
    try:
        i_init = next(i for i, l in enumerate(lines) if 'mov.u32' in l and '%ridx_s32_0, -1;' in l)
        i_loopback = next(i for i, l in enumerate(lines) if 'bra' in l and 'LOOP_ridx_s32_0' in l)
    except StopIteration:
        print('[warp-split] anchors not found; skipping'); return ptx
    WS = W * 32 * NB * 4
    newregs = ('\t.reg \t.u32 %ws_w;\n\t.reg \t.u32 %ws_off;\n\t.reg \t.u32 %ws_l;\n'
               '\t.reg \t.u64 %ws_p;\n\t.reg \t.u64 %ws_p2;\n'
               '\t.reg \t.f32 %ws_t;\n\t.reg \t.pred %ws_p0;')
    ins = ['\tmov.u32\t\t%ws_w, %tid.y;',
           '\tmov.u32\t\t%ws_off, %tid.x;',
           f'\tmul.lo.u32\t%ws_l, %ws_off, {NB*4};',
           '\tmad.lo.u32\t%ws_off, %ws_w, 32, %ws_off;',
           f'\tmul.lo.u32\t%ws_off, %ws_off, {NB*4};',
           '\tcvt.u64.u32\t%ws_p, %ws_off;',
           '\tmov.u64\t\t%ws_p2, _ws;',
           '\tadd.s64\t\t%ws_p, %ws_p2, %ws_p;']
    ins += [f'\tst.shared.f32\t[%ws_p+{j*4}], {b};' for j, b in enumerate(bases)]
    ins += ['\tbar.sync\t\t0;',
            '\tsetp.gt.s32\t%ws_p0, %ws_w, 0;',
            '\t@%ws_p0\tbra\t\tWS_DONE;',
            '\tcvt.u64.u32\t%ws_p, %ws_l;',
            '\tadd.s64\t\t%ws_p, %ws_p2, %ws_p;']
    for w in range(1, W):
        for j, b in enumerate(bases):
            ins.append(f'\tld.shared.f32\t%ws_t, [%ws_p+{w*32*NB*4 + j*4}];')
            ins.append(f'\tadd.f32\t\t{b}, {b}, %ws_t;')
    out = list(lines)
    i_decl = next(i for i, l in enumerate(lines) if '.shared' in l)
    # the rendered _wm[...] shared decl is dead for this kernel; replace it
    out[i_decl] = f'\t.shared\t\t.align 4 .b8 _ws[{WS}];' + '\n' + newregs
    out[i_init] = (f'\tmov.u32\t\t%ridx_s32_0, %tid.y;\n'
                   f'\tadd.s32\t\t%ridx_s32_0, %ridx_s32_0, -{W};')
    i_step = next(i for i, l in enumerate(out) if 'add.s32' in l and '%ridx_s32_0, %ridx_s32_0, 1;' in l)
    out[i_step] = lines[i_step].replace(', 1;', f', {W};')
    text = '\n'.join(out).replace('.maxntid 32', f'.maxntid {32*W}')
    text = text.replace(lines[i_loopback], lines[i_loopback] + '\n' + '\n'.join(ins), 1)
    return text.replace('\tret;', '\tWS_DONE:\n\tret;', 1)

# in-place rewrite of a linearized uop LIST, preserving emission order (incl
# control flow). Pattern replacements that grow the graph inline their new
# intermediate nodes right before the rewritten node.
def rewrite_uop_list(uops, pms):
    subs, out, emitted = {}, [], set()
    for u in uops:
        nu = u
        if nu.src:
            nsrc = tuple(subs.get(s, s) for s in nu.src)
            if any(a is not b for a, b in zip(nsrc, nu.src)): nu = nu.replace(src=nsrc)
        for pm in pms:
            r = pm.rewrite(nu)
            if r is not None and r is not nu:
                nu = r
                break
        subs[u] = nu
        if nu is not u:
            for n in nu.toposort():
                if id(n) not in emitted:
                    out.append(n); emitted.add(id(n))
        elif id(u) not in emitted:
            out.append(u); emitted.add(id(u))
    return out

_LIN_BY_NAME = {}

def reprepare(cap_lin):
    """CL mode: compile comma's own OpenCL SOURCE verbatim (zero re-rendering).
    CUDA mode: two-tier re-render per kernel:
    1. comma's own pickled LINEAR uops (their per-thread distribution is the
       ground truth) -- best fidelity.
    2. fallback: re-linearize from the pickled SINK under NOOPT=1 IMAGE=1 (the
       path that ran end-to-end before), used when comma's LINEAR carries
       patterns the PTX renderer can't express (vector REG-array stores) or
       when ptxas rejects the result.
    Offline ptxas -> cubin everywhere (driver JIT hangs on wide kernels)."""
    if TARGET == 'CL':
        # verbatim OpenCL: swap BINARY to comma's SOURCE text (CLProgram
        # compiles it via clBuildProgram), ceil fractional grid dims
        subs2 = {}
        for u in cap_lin.toposort():
            if u.op is not Ops.PROGRAM or not u.src or u.src[-1].op is not Ops.BINARY: continue
            if not any(s.op is Ops.SINK for s in u.src): continue
            sink = next(s for s in u.src if s.op is Ops.SINK)
            srcu = next((s for s in u.src if s.op is Ops.SOURCE), None)
            assert srcu is not None and isinstance(srcu.arg, str) and '__kernel' in srcu.arg, \
                f'{pi_name(sink)}: no OpenCL SOURCE in pickle'
            pi = u.arg
            _SRC_BY_NAME[pi.name] = srcu.arg
            _SRC_BY_NAME[getattr(sink.arg, 'function_name', '?')] = srcu.arg
            subs2[u] = u.replace(
                                 src=tuple(s for s in u.src if s.op is not Ops.BINARY) +
                                     (UOp(Ops.BINARY, arg=srcu.arg.encode()),))
        new_lin = cap_lin.substitute(subs2, walk=True, enter_calls=True)
        return new_lin, len(subs2), 0
    subs2 = {}
    renderer = _DevMod[TARGET].renderer
    n_fb = 0
    for u in cap_lin.toposort():
        if u.op is not Ops.PROGRAM or not u.src or u.src[-1].op is not Ops.BINARY: continue
        if not any(s.op is Ops.SINK for s in u.src): continue
        sink = next(s for s in u.src if s.op is Ops.SINK)
        lin_u = next((s for s in u.src if s.op is Ops.LINEAR), None)
        kname = getattr(sink.arg, 'function_name', '?') if sink.arg is not None else '?'
        pi = u.arg
        # global-id ('i' SPECIAL) decomposition constant = this kernel's block size
        renderer._i_launch_local = [int(d) if isinstance(d, (int, float)) and d else 1
                                    for d in (pi.local_size or (1, 1, 1))] + [1, 1]

        def finish(uops, ptx, pi):
            # warp-split the serial reduce kernel (ALU_SPLIT warps, 0=off)
            W = int(os.environ.get('ALU_SPLIT', '4') or 0)
            if W and kname == 'r_32_32_3_12288_32':
                ptx2 = _warp_split_ptx(ptx, W)
                if ptx2 is not ptx:
                    ptx = ptx2
                    pi = _dcr(pi, local_size=(32, W, 1))
            # z-split serial-loop WMMA kernels (WMMA_ZSPLIT, 0=off); the
            # transform is structural: only fires on _wm kernels with the
            # ridx do-while and no existing tid.z use
            Z = int(os.environ.get('WMMA_ZSPLIT', '2') or 0)
            if Z > 1:
                ptx2 = _zsplit_ptx(ptx, Z)
                if ptx2 is not ptx:
                    ptx = ptx2
                    ls0 = (list(pi.local_size or (1, 1, 1)) + [1, 1, 1])[:3]
                    pi = _dcr(pi, local_size=(ls0[0], ls0[1], Z))
            # k-split serial-K WMMA kernels (WMMA_KSPLIT, 0=off): ridx loop
            # accumulates into register fragments, stores once after the loop.
            # Whitelist only: the transform assumes the _wm staging shape and
            # verified K-reduction semantics of these kernels.
            # MEASURED NET-NEGATIVE (2026-10-07): correct to 1.8e-12 but
            # r_6_2 flat and r_2_24 16% slower — both kernels sit at the
            # 255-register occupancy wall (128 live f32 fragments), so the
            # extra warps only add reduction traffic. Keep off; the lever
            # there is accumulator storage, not warp count.
            K = int(os.environ.get('WMMA_KSPLIT', '0') or 0)
            if K > 1 and kname in ('r_6_2_32_4_2_2_2_4_4_384',
                                   'r_2_24_32_4_2_2_2_4_4_96'):
                ptx2 = _ksplit_ptx(ptx, K)
                if ptx2 is not ptx:
                    ptx = ptx2
                    ls0 = (list(pi.local_size or (1, 1, 1)) + [1, 1, 1])[:3]
                    pi = _dcr(pi, local_size=(ls0[0], ls0[1], K))
            # reconcile .maxntid with the launch block
            lp = 1
            for d in (pi.local_size or (1, 1, 1)):
                lp *= d if isinstance(d, (int, float)) else 1
            ptx = re.sub(r'\.maxntid \d+', f'.maxntid {int(lp)}', ptx)
            lib = ptxas_compile(ptx)
            # QCOM launcher casts fractional (symbolic-division) grid dims with ceil
            gs = tuple(d if not isinstance(d, (int, float)) else math.ceil(d) for d in pi.global_size)
            return u.replace(arg=_dcr(pi, global_size=gs),
                             src=(sink, UOp(Ops.LINEAR, src=tuple(uops)),
                                  UOp(Ops.SOURCE, arg=ptx), UOp(Ops.BINARY, arg=lib)))

        new = None
        if lin_u is not None:
            # zero-border masking only for kernels whose image loads are all
            # plain (no gated WHERE cones — those render whole-image width and
            # explode the per-thread register clamp); gated ones keep edge-clamp
            # NOTE: comma's CLK_ADDRESS_CLAMP returns BORDER ZERO for OOB image
            # reads (probed on NV OpenCL), not edge-clamp. A zero-mask rewrite
            # (pm_img2d) was attempted but destabilizes kernels whose INDEX
            # coords / gated-load cones carry whole-image shapes (register
            # width mismatch at render). Kept as documented future work; the
            # edge-clamp error affects only border-touching conv taps.
            # image INDEXes stay 3-src: the renderer's image branch emits
            # clamp + border-zero validity at register level (CLK_ADDRESS_CLAMP)
            uops = rewrite_uop_list(list(lin_u.src), (pm_div, pm_rcp, pm_nohalf, pm_bool, pm_alu))
            # rewrites can pull PARAMs forward, changing their appearance order
            # vs comma's pickle; the PTX binds params by emission order while
            # exec_kernel passes buffers by pi.globals (int indices) — realign
            pi = u.arg
            em = tuple(x for x in uops if x.op is Ops.PARAM)
            old_params = [x for x in lin_u.src if x.op is Ops.PARAM]
            if len(em) == len(old_params) and any(a is not b for a, b in zip(em, old_params)):
                posof = {id(p): i for i, p in enumerate(old_params)}
                if all(id(p) in posof for p in em) and len(set(id(p) for p in em)) == len(em):
                    pi = _dcr(pi, globals=tuple(pi.globals[posof[id(p)]] for p in em))
            print(f'[reprep] {kname} (comma-LINEAR)', flush=True)
            _LIN_BY_NAME[kname] = (uops, pi.local_size or (1, 1, 1))
            try:
                new = finish(uops, renderer.render(uops), pi)
            except Exception as e:
                print(f'[reprep] {kname}: comma-LINEAR failed ({str(e)[:60]}), falling back', flush=True)
        if new is None:
            n_fb += 1
            print(f'[reprep] {kname} (re-linearized)', flush=True)
            # fallback's 'idx0' SPECIALs are pure global ids (no locals; the
            # launch folds to 1 thread/block), so the decomposition constant is 1
            renderer._i_launch_local = [1, 1, 1]
            fixed = sink
            for pm in (pm_storage, pm_div, pm_rcp, pm_nohalf, pm_bool, pm_alu):
                fixed = graph_rewrite(fixed, pm, walk=True)
            with _Ctx(NOOPT=1, IMAGE=1):
                prg = to_program(u.replace(src=(fixed,)), renderer)
            pi = prg.arg
            ptx = next((s.arg for s in prg.src if s.op is Ops.SOURCE), '')
            lnu = next(s for s in prg.src if s.op is Ops.LINEAR)
            if any(x.op is Ops.WMMA for x in lnu.src):
                print(f'[reprep] {kname}: WARNING fallback carries WMMA — folded 1-thread blocks break the wave layout', flush=True)
            m = re.search(r'\.maxntid (\d+)', ptx)
            maxntid = int(m.group(1)) if m else 1
            if maxntid <= 1:
                l = pi.local_size or (1, 1, 1)
                def _fold(dims):
                    out = []
                    for a, b in zip(dims, l):
                        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                            out.append(int(a * b))
                        elif b == 1:
                            out.append(a)
                        else:
                            out.append(a * b)
                    return tuple(out)
                pi = _dcr(pi, global_size=_fold(pi.global_size), local_size=(1, 1, 1))
            new = finish(list(lnu.src), ptx, pi)
        subs2[u] = new
    new_lin = cap_lin.substitute(subs2, walk=True, enter_calls=True)
    return new_lin, len(subs2), n_fb

# module-load failure diagnostics
from dataclasses import replace as _dcr
import tinygrad.runtime.ops_cuda as _oc

if os.getenv('KALLOC'):
    _ka = {'n': 0, 't': 0.0, 'fn': 0, 'ft': 0.0, 'cin': 0, 'ct': 0.0, 'cout': 0, 'cot': 0.0, 'cb': 0, 'cob': 0}
    def _wrap(name):
        _orig = getattr(_oc.CUDAAllocator, name)
        def w(self, *a, **kw):
            _t0 = time.perf_counter()
            r = _orig(self, *a, **kw)
            _k = _ka; _key = {'_alloc': ('n', 't'), '_free': ('fn', 'ft'),
                              '_copyin': ('cin', 'ct'), '_copyout': ('cout', 'cot')}[name]
            _k[_key[0]] += 1; _k[_key[1]] += time.perf_counter() - _t0
            if name == '_copyin' and a: _ka['cb'] += len(a[1]) if hasattr(a[1], '__len__') else 0
            if name == '_copyout' and a: _ka['cob'] += len(a[0]) if hasattr(a[0], '__len__') else 0
            return r
        return w
    for _m in ('_alloc', '_free', '_copyin', '_copyout'):
        setattr(_oc.CUDAAllocator, _m, _wrap(_m))
    import atexit as _atx
    def _ka_dump():
        k = _ka
        print(f'--- CUDAAllocator: alloc {k["n"]}x {k["t"]*1e3:.0f}ms, free {k["fn"]}x {k["ft"]*1e3:.0f}ms, '
              f'copyin {k["cin"]}x {k["ct"]*1e3:.0f}ms {k["cb"]/1e6:.1f}MB, '
              f'copyout {k["cout"]}x {k["cot"]*1e3:.0f}ms {k["cob"]/1e6:.1f}MB ---', flush=True)
    _atx.register(_ka_dump)

_orig_cp_init = _oc.CUDAProgram.__init__
def _spy_cp_init(self, dev, obj, *a, **kw):
    nm = pi_name(type('X', (), {'arg': obj})())
    print(f'[loading] {nm}', flush=True)
    t0 = time.perf_counter()
    try:
        _orig_cp_init(self, dev, obj, *a, **kw)
    except Exception as e:
        print(f'LOAD FAIL {nm}: {str(e)[:70]}')
        open(f'loadfail_{nm}.ptx', 'wb').write(obj.lib if isinstance(obj.lib, bytes) else str(obj.lib).encode())
        raise
    dt = time.perf_counter() - t0
    if dt > 0.5: print(f'[load] {nm} took {dt:.1f}s', flush=True)
_oc.CUDAProgram.__init__ = _spy_cp_init

if os.getenv('CUDASYNC') or os.getenv('KTIME'):   # per-launch fence / kernel timing
    _orig_cpcall = _oc.CUDAProgram.__call__
    _kt = {}
    def _spy_call(self, *a, **kw):
        _t0 = time.perf_counter()
        if os.getenv('KTIME') and not os.getenv('CUDASYNC'):
            _ms = _oc.cu_time_execution(lambda: _orig_cpcall(self, *a, **kw), enable=True)
            _kt[self.name] = _kt.get(self.name, 0.0) + _ms
            return None
        r = _orig_cpcall(self, *a, **kw)
        st = _oc.cuda.cuCtxSynchronize()
        if st != 0:
            print(f'[CUDAFAULT] {self.name}: {_oc.cuda.enum_cudaError_enum.get(st, st)}', flush=True)
            raise RuntimeError(f'{self.name} fault')
        return r
    _oc.CUDAProgram.__call__ = _spy_call
    import atexit as _atexit
    def _kt_dump():
        if not _kt: return
        print('--- per-kernel GPU time (sum over run, events) ---', flush=True)
        for _n, _t in sorted(_kt.items(), key=lambda kv: -kv[1])[:20]:
            print(f'  {_t:8.1f}ms  {_n}', flush=True)
    _atexit.register(_kt_dump)

    # also fence the copy ops (they can be the faulting op between launches)
    _orig_alloc_methods = {}
    def _fence(name):
        _orig = getattr(_oc.CUDAAllocator, name)
        def w(self, *a, **kw):
            r = _orig(self, *a, **kw)
            st = _oc.cuda.cuCtxSynchronize()
            if st != 0:
                print(f'[CUDAFAULT-copy] {name} args={a!r}: {_oc.cuda.enum_cudaError_enum.get(st, st)}', flush=True)
                raise RuntimeError(f'{name} fault')
            return r
        setattr(_oc.CUDAAllocator, name, w)
    for _m in ('_transfer', '_copyin', '_copyout'):
        _fence(_m)

# --- load model ---------------------------------------------------------------------
print(f'loading {PKL} (target={TARGET}) with pinned tinygrad ...')
t0 = time.perf_counter()
obj = load_oob_from(open(PKL, 'rb').read())
print(f'unpickled in {time.perf_counter()-t0:.1f}s')
jit = obj['run_policy']
cap = jit.captured
meta = obj['metadata']
ish, out_slices = meta['input_shapes'], meta['output_slices']

t0 = time.perf_counter()
lin, nk, nl = reprepare(cap._linear)
print(f'policy: recompiled {nk} kernels for {TARGET} in {time.perf_counter()-t0:.1f}s ({nl} re-linearized fallbacks)')

# flatten batched graph calls into individual kernel calls
flat = []
for c in lin.src:
    cf = c.src[0]
    if cf.op is Ops.CUSTOM_FUNCTION and cf.arg == 'graph' and cf.src and cf.src[0].op is Ops.LINEAR:
        flat.extend(cf.src[0].src)
    else:
        flat.append(c)
lin = UOp(Ops.LINEAR, src=tuple(flat))
cap._linear = lin
cap.__dict__.pop('linear', None)

# --- input queues (devices from the JIT's own expectations) ---------------------------
FS = 4
img = ish['img']; n_frames = img[1] // 6
img_buf = (FS * (n_frames - 1) + 1, 6, img[2], img[3])
fb = ish['features_buffer']; dp = ish['desire_pulse']
_exp = dict(zip(cap.expected_names, cap.expected_input_info))
def _dev(name): return _exp[name][3]
queues = {
    'img_q':     Tensor(np.zeros(img_buf, np.uint8), device=_dev('img_q')).contiguous().realize(),
    'big_img_q': Tensor(np.zeros(img_buf, np.uint8), device=_dev('big_img_q')).contiguous().realize(),
    'feat_q':    Tensor(np.zeros((FS*fb[1], fb[0], fb[2]), np.float32), device=_dev('feat_q')).contiguous().realize(),
    'desire_q':  Tensor(np.zeros((FS*dp[1], dp[0], dp[2]), np.float32), device=_dev('desire_q')).contiguous().realize(),
}
PACKED = np.zeros(8 + 2 + 2 + fb[2], np.float32)  # desire8 | tc2 | action_t2 | prev_feat512
queues['packed_npy_inputs'] = Tensor(PACKED, device='NPY').realize()

def make_warped(val=128):
    return Tensor(np.full((2, 6, img[2], img[3]), val, np.uint8), device=_dev('warped')).contiguous().realize()

# --- REPLAY + CUDA Graph fast path ------------------------------------------------
# Stock CapturedJit re-resolves the 48K-uop graph in Python every frame AND
# each input copy stages through a synchronous D2H (6 pipeline drains/frame).
# But the dominant cost on Windows/WDDM is per-launch submission (~0.5ms x 497
# launches = 255ms; GPU busy is <1ms — proven with event pairs). Shapes and
# buffer addresses are static here, so: record the resolved launch sequence
# once, capture it into a CUDA graph, then each frame = 2 tiny H2D/D2D copies
# + ONE cuGraphLaunch + one sync. REPLAY=0 disables everything.
_REPLAY = os.getenv('REPLAY', '1') == '1'
if _REPLAY:
    import ctypes as _ct
    import tinygrad.runtime.autogen.cuda as _cug

    _rec, _rec_on = [], False
    _cp_call0 = _oc.CUDAProgram.__call__
    _ca_in0 = _oc.CUDAAllocator._copyin

    def _cp_spy(self, *a, **kw):
        if _rec_on: _rec.append((self, a, kw))
        return _cp_call0(self, *a, **kw)

    def _ci_spy(self, dest, src):
        if _rec_on:
            if getattr(src, 'obj', None) is PACKED: _rec.append(('ci_packed', dest, src))
            elif len(src) == 2 * 6 * img[2] * img[3]: _rec.append(('ci_warped_d2d', dest, None))
            else: _rec.append(('ci', dest, src))
        return _ca_in0(self, dest, src)

    _oc.CUDAProgram.__call__ = _cp_spy
    _oc.CUDAAllocator._copyin = _ci_spy

    WQ = make_warped(128)
    _wbuf = WQ.uop.base.realized
    _walloc = _DevMod[_dev('warped')].allocator
    _plan = None
    _ret = None
    _gstream = _gexec = None
    _WNBYTES = 2 * 6 * img[2] * img[3]
    _STREAM = [None]
    _wstage = np.zeros((2, 6, img[2], img[3]), np.uint8)   # persistent H2D staging (async copy reads it later)

    def _cp_stream_call(self, *args, global_size=(1, 1, 1), local_size=(1, 1, 1), vals=(), wait=False, **kw):
        # stream-directed launch used inside graph capture (no events allowed);
        # must update the cached param struct per call — the same program is
        # launched many times per frame with DIFFERENT buffer sets
        if not hasattr(self, 'vargs'):
            return _cp_call0(self, *args, global_size=global_size, local_size=local_size, vals=vals, wait=wait, **kw)
        for _i in range(len(args)): self.c_args.__setattr__(f'f{_i}', args[_i])
        for _i in range(len(vals)): self.c_args.__setattr__(f'v{_i}', vals[_i])
        _oc.check(_cug.cuLaunchKernel(self.prg, *global_size, *local_size, self.smem, _STREAM[0], None, self.vargs))
        return 0.0

    def _mvptr(mv):
        o = mv.obj
        if hasattr(o, 'ctypes'): return o.ctypes.data_as(_ct.c_void_p)
        return _ct.cast(_ct.c_char * len(mv)).from_buffer(o) if False else _ct.cast(
            _ct.pointer((_ct.c_char * len(mv)).from_buffer(o)), _ct.c_void_p)

    def _fire_on_stream(stream):
        for item in _plan:
            if item[0] == 'ci_packed':
                _oc.check(_cug.cuMemcpyHtoDAsync_v2(item[1], _mvptr(item[2]), len(item[2]), stream))
            elif item[0] == 'ci_warped_d2d':
                _oc.check(_cug.cuMemcpyDtoDAsync_v2(item[1], _wbuf._buf, _WNBYTES, stream))
            elif item[0] == 'ci':
                _oc.check(_cug.cuMemcpyHtoDAsync_v2(item[1], _mvptr(item[2]), len(item[2]), stream))
            else:
                _STREAM[0] = stream
                try: item[0](*item[1], **item[2])
                finally: _STREAM[0] = None

    def run_frame(warped_np, packed):
        global _plan, _ret, _rec_on, _gstream, _gexec
        PACKED[:] = packed
        _walloc._copyin(_wbuf._buf, memoryview(np.ascontiguousarray(warped_np)).cast('B'))
        if _plan is None:
            _rec_on = True; _rec.clear()
            try:
                _ret = jit(**{k: queues[k] for k in ('img_q', 'big_img_q', 'feat_q', 'desire_q')},
                           packed_npy_inputs=queues['packed_npy_inputs'], warped=WQ)
            finally:
                _rec_on = False
            _plan = list(_rec); _rec.clear()
            _nl = sum(1 for it in _plan if not isinstance(it[0], str))
            print(f'[replay] recorded {len(_plan)} ops (launches={_nl}, ci_packed={sum(it[0]=="ci_packed" for it in _plan)}, ci_d2d={sum(it[0]=="ci_warped_d2d" for it in _plan)})', flush=True)
            # capture: replay the whole plan on a capture stream, instantiate
            _oc.CUDAProgram.__call__ = _cp_stream_call
            _gstream = _cug.CUstream()
            _oc.check(_cug.cuStreamCreate(_ct.byref(_gstream), 0))
            _STREAM[0] = _gstream
            _oc.check(_cug.cuStreamBeginCapture(_gstream))
            _gr = _cug.CUgraph()
            try:
                _kcap = int(os.getenv('GCAP', '0'))
                _saved = _plan
                if _kcap: _plan = _plan[:_kcap]
                _fire_on_stream(_gstream)
                if _kcap: _plan = _saved
                _oc.check(_cug.cuStreamEndCapture(_gstream, _ct.byref(_gr)))
            except BaseException:
                _cug.cuStreamEndCapture(_gstream, _ct.byref(_gr))
                _STREAM[0] = None
                _oc.CUDAProgram.__call__ = _cp_spy
                raise
            _nn = _ct.c_size_t(0)
            _oc.check(_cug.cuGraphGetNodes(_gr, _ct.cast(0, _ct.POINTER(_cug.CUgraphNode)), _nn))
            print(f'[replay] graph nodes = {_nn.value}', flush=True)
            _gexec = _cug.CUgraphExec(); _gerr = _cug.CUgraphNode()
            _oc.check(_cug.cuGraphInstantiate(_ct.byref(_gexec), _gr, _ct.byref(_gerr), None, 0))
            _STREAM[0] = None
            _oc.CUDAProgram.__call__ = _cp_spy
            print('[replay] captured into CUDA graph', flush=True)
        # frame = refresh inputs on the graph stream, one graph launch, sync
        _wstage[:] = warped_np
        _oc.check(_cug.cuMemcpyHtoDAsync_v2(_wbuf._buf, _wstage.ctypes.data_as(_ct.c_void_p), _WNBYTES, _gstream))
        for item in _plan:
            if item[0] == 'ci_packed':
                _oc.check(_cug.cuMemcpyHtoDAsync_v2(item[1], _mvptr(item[2]), len(item[2]), _gstream))
        if os.getenv('GSPAN'):
            _ge0 = _cug.CUevent(); _ge1 = _cug.CUevent()
            _cug.cuEventCreate(_ct.byref(_ge0), 0); _cug.cuEventCreate(_ct.byref(_ge1), 0)
            _cug.cuEventRecord(_ge0, _gstream)
        if os.getenv('KSYNC'):
            for item in _plan:
                if isinstance(item[0], str): continue
                _oc.check(_cug.cuGraphLaunch(_gexec, _gstream)) if False else None
            import ctypes as _ct3
            _per = []
            import hashlib as _hl
            for _ii, item in enumerate(_plan):
                if isinstance(item[0], str): continue
                _h = _hl.md5()
                for _b in item[1]:
                    try:
                        _nb = min(getattr(_b, 'size', 0) or 0, 65536)
                        import numpy as _np
                        _host = bytearray(max(_nb, 4))
                        _oc.check(_cug.cuMemcpyDtoH_v2(_host, _b.get_buf(_dev('warped')) if hasattr(_b, 'get_buf') else _b._buf, _nb))
                        _h.update(_host)
                    except Exception:
                        _h.update(b'?')
                _e0 = _cug.CUevent(); _e1 = _cug.CUevent()
                _cug.cuEventCreate(_ct3.byref(_e0), 0); _cug.cuEventCreate(_ct3.byref(_e1), 0)
                _STREAM[0] = _gstream
                _cug.cuEventRecord(_e0, _gstream)
                item[0](*item[1], **item[2])
                _cug.cuEventRecord(_e1, _gstream)
                _STREAM[0] = None
                _cug.cuEventSynchronize(_e1)
                _ms = _ct3.c_float(); _cug.cuEventElapsedTime(_ct3.byref(_ms), _e0, _e1)
                _per.append((_ms.value, item[0].name, _h.hexdigest()[:8]))
                print(f'[ksync] {_ii:3d} {item[0].name:32} {_ms.value:7.3f}ms pre={_h.hexdigest()[:8]}', flush=True)
                if os.getenv('KDUMP') and item[0].name.startswith(os.getenv('KDUMP')) and _ii < 85:
                    for _j, _ptr in enumerate(item[1]):
                        _base = _cug.CUdeviceptr(); _sz = _ct.c_size_t()
                        _oc.check(_cug.cuMemGetAddressRange_v2(_ct.byref(_base), _ct.byref(_sz), _ptr))
                        _nb = min(_sz.value, 1 << 22)
                        _host = bytearray(_nb)
                        _oc.check(_cug.cuMemcpyDtoH_v2(_ct.cast(( _ct.c_char * _nb).from_buffer(_host), _ct.c_void_p), _ptr, _nb))
                        open(f'kd_{_ii}_{_j}_{_sz.value}.bin', 'wb').write(_host)
                        if _j == 0: open(f'kd_{_ii}_meta.txt', 'w').write(item[0].name)
                    print(f'[kdump] launch {_ii}: {len(item[1])} bufs dumped', flush=True)
            _per.sort(reverse=True)
            print('[ksync] ' + ' | '.join(f'{n}:{t:.2f}' for t, n in _per[:10]), flush=True)
            _tot = sum(t for t, _ in _per)
            print(f'[ksync] total kernel GPU time = {_tot:.2f}ms over {len(_per)} launches', flush=True)
            return _ret[0].numpy()[0]
        _oc.check(_cug.cuGraphLaunch(_gexec, _gstream))
        if os.getenv('GSPAN'):
            _cug.cuEventRecord(_ge1, _gstream)
        _oc.check(_cug.cuStreamSynchronize(_gstream))
        if os.getenv('GSPAN'):
            _gms = _ct.c_float(); _cug.cuEventElapsedTime(_ct.byref(_gms), _ge0, _ge1)
            print(f'[gspan] {_gms.value:.2f}ms', flush=True)
        return _ret[0].numpy()[0]
else:
    def run_frame(warped_t, packed):
        PACKED[:] = packed
        ret = jit(**{k: queues[k] for k in ('img_q', 'big_img_q', 'feat_q', 'desire_q')},
                  packed_npy_inputs=queues['packed_npy_inputs'], warped=warped_t)
        return ret[0].numpy()[0]
        return ret[0].numpy()[0]

# --- smoke run + timing ---------------------------------------------------------------
print('--- run 1 (cold) ---')
t0 = time.perf_counter()
p0 = np.zeros(524, np.float32)
out = run_frame(np.full((2, 6, img[2], img[3]), 128, np.uint8), p0)
print(f'cold call {time.perf_counter()-t0:.2f}s, finite={np.isfinite(out).all()} absmax={np.abs(out).max():.4f}')
p0[12:] = out[out_slices['hidden_state']]

print('--- timing (10 iters) ---')
ts = []
for i in range(10):
    t0 = time.perf_counter()
    out = run_frame(np.full((2, 6, img[2], img[3]), 128, np.uint8), p0)
    p0[12:] = out[out_slices['hidden_state']]
    ts.append((time.perf_counter()-t0)*1e3)
print('per-call ms:', [f'{t:.1f}' for t in ts])

for name in ('plan', 'action', 'lead', 'pose', 'lane_lines', 'meta', 'hidden_state'):
    if name in out_slices:
        s = out_slices[name]
        seg = out[s]
        print(f'  {name:14} absmax={np.abs(seg).max():9.4f} mean={seg.mean():9.4f}')

# --- input-sensitivity A/B test -------------------------------------------------------
print('--- A/B sensitivity (bright vs dark road) ---')
def frames(val, n=4):
    pk = np.zeros(524, np.float32)
    outs = []
    wf = np.full((2, 6, img[2], img[3]), val, np.uint8)
    for i in range(n):
        o = run_frame(wf, pk)
        pk[12:] = o[out_slices['hidden_state']]
        outs.append(o)
    return outs
A = frames(40); B = frames(220)
for name in ('plan', 'lead', 'hidden_state', 'lane_lines'):
    if name in out_slices:
        s = out_slices[name]
        d = np.abs(A[-1][s] - B[-1][s]).max()
        print(f'  {name:14} A/B maxdiff = {d:.4f}')
