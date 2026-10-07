"""Offline A/B bench for r_12_32_4_2_2_2_2_4_768 variants.
Loads r12_pre.npz (captured real inputs), launches a cubin on a fresh ctx,
times it, and compares outputs against the baseline cubin bit-for-bit.
Usage: python bench_alu.py <cubin> [baseline_cubin]
"""
import ctypes, time, sys, json
import numpy as np
from tinygrad.runtime.autogen import cuda as cu

def check(st):
    if st != 0: raise RuntimeError(cu.enum_cudaError_enum.get(st, st))

KN = b'r_12_32_4_2_2_2_2_4_768'
GS = (12, 1, 1)
LS = tuple(int(x) for x in (sys.argv[3].split(',') if len(sys.argv) > 3 else '32,4,1'.split(',')))
NP_ = 6

def run_cubin(cub_path, bufs, ls=None):
    mod = cu.CUmodule(); check(cu.cuModuleLoadData(ctypes.byref(mod), open(cub_path, 'rb').read()))
    f = cu.CUfunction(); check(cu.cuModuleGetFunction(ctypes.byref(f), mod, KN))
    vals = (ctypes.c_uint64 * NP_)(*[b.value for b in bufs])
    params = (ctypes.c_void_p * NP_)(*[ctypes.cast(ctypes.byref(vals, i * 8), ctypes.c_void_p) for i in range(NP_)])
    ls = ls or (32, 4, 1)
    ts = []
    for i in range(15):
        check(cu.cuCtxSetCurrent(ctx))
        t0 = time.perf_counter()
        check(cu.cuLaunchKernel(f, *GS, *ls, 0, None, params, None))
        check(cu.cuCtxSynchronize())
        ts.append(time.perf_counter() - t0)
    check(cu.cuModuleUnload(mod))
    return ts

def copyout(dptr, nb):
    host = bytearray(nb)
    check(cu.cuMemcpyDtoH_v2(ctypes.cast((ctypes.c_char * nb).from_buffer(host), ctypes.c_void_p), dptr, nb))
    return host

check(cu.cuInit(0))
dev = cu.CUdevice(); check(cu.cuDeviceGet(ctypes.byref(dev), 0))
ctx = cu.CUcontext(); check(cu.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev))
check(cu.cuCtxSetCurrent(ctx))

z = np.load('r12_pre.npz')
host = [z[f'arg{i}'].tobytes() for i in range(NP_)]
bufs = []
for h in host:
    p = cu.CUdeviceptr(); check(cu.cuMemAlloc_v2(ctypes.byref(p), len(h)))
    check(cu.cuMemcpyHtoD_v2(p, ctypes.cast((ctypes.c_char * len(h)).from_buffer(bytearray(h)), ctypes.c_void_p), len(h)))
    bufs.append(p)

base_cub = sys.argv[2] if len(sys.argv) > 2 else 'ptxas_cache/sm_75_a959386ad636c803.cubin'
ts0 = run_cubin(base_cub, bufs)
base_out = copyout(bufs[0], min(len(host[0]), 1 << 16))
ts = run_cubin(sys.argv[1], bufs, ls=LS)
out = copyout(bufs[0], min(len(host[0]), 1 << 16))

same = np.frombuffer(base_out, np.uint8) == np.frombuffer(out, np.uint8)
b32 = np.frombuffer(base_out, np.float32); v32 = np.frombuffer(out, np.float32)
nz = b32 != 0
maxabs = float(np.max(np.abs(b32[nz] - v32[nz]))) if nz.any() else 0.0
maxrel = float(np.max(np.abs(b32[nz] - v32[nz]) / np.maximum(1e-30, np.abs(b32[nz])))) if nz.any() else 0.0
print(f"baseline: med={sorted(ts0)[7]*1e3:.2f}ms min={min(ts0)*1e3:.2f}ms")
print(f"variant : med={sorted(ts)[7]*1e3:.2f}ms min={min(ts)*1e3:.2f}ms  "
      f"bit-identical={bool(same.all())} ndiff={int((~same).sum())} maxabs={maxabs:.3e} maxrel={maxrel:.3e}")
