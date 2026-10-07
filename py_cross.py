"""Cross-validate one WMMA kernel: tinygrad's PythonProgram (the executable
spec of comma's WMMA semantics, including generic_wmma_helper) vs our CUDA
mma cubin, on the SAME real runtime buffers."""
import os, sys, time, ctypes, struct, io, pickle
import numpy as np

os.environ.setdefault('TG_DEV', 'CUDA')
PYX_KERNEL = 'r_32_3_32_2_2_2_2_4_4_2_4_4'
sys.argv = ['x', os.environ.get('PKL', 'models/big_driving.pkl')]
src = open('run_pinned2.py', encoding='utf-8').read()
exec(compile(src.split('# --- smoke run')[0], 'run_pinned2.py', 'exec'))
import tinygrad.runtime.autogen.cuda as _cug
import tinygrad.runtime.ops_cuda as _oc

KN = PYX_KERNEL
assert KN in _LIN_BY_NAME, f'{KN} not stashed'
lin_uops, local_size = _LIN_BY_NAME[KN]
print(f'[pyx] kernel {KN}: {len(lin_uops)} uops, local={local_size}', flush=True)

# --- rewrite 'i' SPECIALs into g/l form that PythonProgram understands ---
from tinygrad.uop.ops import Ops, UOp, PatternMatcher, UPat
from tinygrad.dtype import dtypes
_LOC = [int(d) if isinstance(d, (int, float)) and d else 1 for d in local_size] + [1, 1, 1]

def _rule_ispec(u):
    if u.op is Ops.SPECIAL and isinstance(u.arg, str) and u.arg[0] == 'i':
        d = int(u.arg[-1]) if u.arg[-1].isdigit() else 0
        dt = u.dtype
        g = UOp(Ops.SPECIAL, dt, arg='g' + str(d))
        l = UOp(Ops.SPECIAL, dt, arg='l' + str(d))
        return g * UOp(Ops.CONST, dt, arg=_LOC[d]) + l
    return None

from tinygrad.uop.ops import GroupOp as _GOp

def _norm(s, dt):
    # the Python emulator asserts operand dtypes match; the PTX renderer
    # treats registers as untyped bits, so normalize BIT-PRESERVING for
    # int<->int (address math relies on it) and by value for floats
    from tinygrad.dtype import dtypes as _dt
    if dtypes.is_float(dt) and dtypes.is_float(s.dtype) and s.dtype != dt:
        return s.cast(dt)
    return s.bitcast(dt) if s.dtype != dt else s

def _rule_dt(u):
    if u.op in _GOp.ALU and u.op not in _GOp.Comparison:
        if any(s.dtype != u.dtype for s in u.src):
            return UOp(u.op, u.dtype, tuple(_norm(s, u.dtype) for s in u.src))
        return None
    if u.op in _GOp.Comparison and not all(s.dtype == u.src[0].dtype for s in u.src):
        dt = max((s.dtype for s in u.src), key=lambda d: d.priority)
        return UOp(u.op, u.dtype, tuple(_norm(s, dt) for s in u.src))
    return None

pm_i = PatternMatcher([(UPat(Ops.SPECIAL, name='u'), _rule_ispec),
                       (UPat(tuple(_GOp.ALU), name='u'), _rule_dt)])
uops_py = rewrite_uop_list(list(lin_uops), (pm_i,))
print(f'[pyx] specials in linear: '
      f'{sorted({u.arg for u in uops_py if u.op is Ops.SPECIAL})}', flush=True)

# --- hook CUDAProgram.__call__ to snapshot the target kernel's buffers ---
from tinygrad.device import TinyELF
from tinygrad.runtime.ops_python import PythonProgram

_ct = ctypes
_captured = {}
_orig_call = _oc.CUDAProgram.__call__

def _copyout(ptr, nb):
    host = bytearray(nb)
    _oc.check(_cug.cuMemcpyDtoH_v2(_ct.cast((_ct.c_char * nb).from_buffer(host), _ct.c_void_p), ptr, nb))
    return host

def _arg_span(ptr):
    # ptr may be an interior pointer: clamp the copy to its allocation
    _base = _cug.CUdeviceptr(); _sz = _ct.c_size_t()
    _oc.check(_cug.cuMemGetAddressRange_v2(_ct.byref(_base), _ct.byref(_sz), ptr))
    off = ptr.value - _base.value if hasattr(ptr, 'value') else int(ptr) - _base.value
    return max(0, _sz.value - off)

def _hook(self, *a, **kw):
    if self.name == KN and 'pre' not in _captured:
        sizes = [_arg_span(p) for p in a]
        _captured['pre'] = [_copyout(p, n) for p, n in zip(a, sizes)]
        _captured['prg'] = self   # keep the CUDAProgram: signature/smem for encode_args
        r = _cp_spy(self, *a, **kw)   # records + launches (preserves plan build)
        _oc.check(_cug.cuCtxSynchronize())
        _captured['post'] = [_copyout(p, n) for p, n in zip(a, sizes)]
        _captured['kw'] = dict(kw)
        print(f'[pyx] captured {KN}: args={len(a)} sizes={sizes}', flush=True)
        return r
    return _cp_spy(self, *a, **kw)

_oc.CUDAProgram.__call__ = _hook
out = run_frame(np.full((2, 6, img[2], img[3]), 111, np.uint8), np.zeros(524, np.float32))
assert 'pre' in _captured, 'target kernel never launched'
post_cuda = _captured['post']

kw = _captured['kw']
gs = kw.get('global_size', (1, 1, 1)); ls = kw.get('local_size', (1, 1, 1)); vv = kw.get('vals', ())
_prg = _captured['prg']

# --- run the FFMA and mma cubins on the same pre-state (before the slow
#     PythonProgram so a harness failure surfaces in seconds) ---
def run_alt(cubin):
    mod = _cug.CUmodule(); _oc.check(_cug.cuModuleLoadData(ctypes.byref(mod), open(cubin, "rb").read()))
    af = _cug.CUfunction(); _oc.check(_cug.cuModuleGetFunction(ctypes.byref(af), mod, KN.encode()))
    dvals = []
    for hst in _captured["pre"]:
        d = _cug.CUdeviceptr(); _oc.check(_cug.cuMemAlloc_v2(ctypes.byref(d), len(hst)))
        _oc.check(_cug.cuMemcpyHtoD_v2(d, _ct.cast((_ct.c_char * len(hst)).from_buffer(hst), _ct.c_void_p), len(hst)))
        dvals.append(d.value)
    # pack params exactly like tinygrad does: extra-format buffer via encode_args
    c_args, vargs = _oc.encode_args(dvals, vv, _prg.signature)
    print(f'[pyx] {os.path.basename(cubin)}: nargs={len(dvals)} vals={vv} '
          f'sig={_prg.signature!r} smem={_prg.smem} gs={gs} ls={ls}', flush=True)
    _oc.check(_cug.cuLaunchKernel(af, *gs, *ls, _prg.smem, None, None, vargs))
    _oc.check(_cug.cuCtxSynchronize())
    return [_copyout(d, len(h)) for d, h in zip(dvals, _captured["pre"])]

mma_post2 = run_alt("ptxas_cache/sm_75_451bf91e5f2c3cd4.cubin")   # harness self-check
# NOTE: sm_75_6b069abf is a PRE-fix render (mixed tid.x into the consume base
# -> misaligned 716 standalone); this is the fixed FFMA render
ffma_post = run_alt("ptxas_cache/sm_75_62197b916ec44426.cubin")
for j, (c, m) in enumerate(zip(post_cuda, mma_post2)):
    n = min(len(c), len(m)) // 4
    dc = np.frombuffer(c, np.float32)[:n]; dm = np.frombuffer(m, np.float32)[:n]
    print(f'[pyx] harness arg{j}: |mma_relaunch - real_launch| max={float(np.max(np.abs(dc-dm))) if n else 0:.3e}', flush=True)

py_bufs = [memoryview(bytearray(p)) for p in _captured['pre']]
print(f'[pyx] running PythonProgram gs={gs} ls={ls} ...', flush=True)
from tinygrad.helpers import Target as _Target
obj = TinyELF(name=KN, lib=pickle.dumps(uops_py), target=_Target('PY', 0), signature=())
pyprg = PythonProgram(None, obj)
print(f'[pyx] PythonProgram built with {len(pyprg.uops)} uops', flush=True)
t0 = time.perf_counter()
pyprg(*py_bufs, global_size=gs, local_size=ls, vals=vv)
print(f'[pyx] python exec: {(time.perf_counter()-t0)/60:.1f} min', flush=True)

print("=== per-param: py-spec vs cu-mma vs cu-ffma ===")
for j, (c, p) in enumerate(zip(post_cuda, py_bufs)):
    n = min(len(c), len(p)) // 4
    dc = np.frombuffer(c, np.float32)[:n]
    dp = np.frombuffer(p, np.float32)[:n]
    df = np.frombuffer(ffma_post[j], np.float32)[:n]
    dm = np.frombuffer(mma_post2[j], np.float32)[:n]
    dpm = float(np.max(np.abs(dc - dp))) if n else 0.0
    dmf = float(np.max(np.abs(dm - df))) if n else 0.0
    dpf = float(np.max(np.abs(dp - df))) if n else 0.0
    print(f"  arg{j}: n={n} py-vs-mma={dpm:.3e} ffma-vs-mma={dmf:.3e} py-vs-ffma={dpf:.3e} "
          f"nnz(py/cu/ff)={np.count_nonzero(dp)}/{np.count_nonzero(dc)}/{np.count_nonzero(df)}")
import sys as _s; _s.exit(0)

# --- compare every param: cuda post vs python post ---
print('\n=== per-param comparison (cuda-mma vs python-spec) ===')
for j, (c, p) in enumerate(zip(post_cuda, py_bufs)):
    n = min(len(c), len(p)) // 4
    dc = np.frombuffer(c, np.float32)[:n]
    dp = np.frombuffer(p, np.float32)[:n]
    fin = np.isfinite(dp).all()
    md = float(np.max(np.abs(dc - dp))) if n and fin else float('nan')
    rel = md / max(1e-9, float(np.max(np.abs(dc)))) if n else float('nan')
    print(f'  arg{j}: n={n} maxdiff={md:.3e} rel={rel:.2e} finite={fin} '
          f'nnz(cuda/py)={np.count_nonzero(dc)}/{np.count_nonzero(dp)}')
