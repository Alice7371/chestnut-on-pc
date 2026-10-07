"""Micro-bench one WMMA kernel standalone: r_6_2_32_4_2_2_2_4_4_384."""
import ctypes, time, os, sys
import numpy as np
from tinygrad.runtime.autogen import cuda as cu

def check(st):
    if st != 0: raise RuntimeError(cu.enum_cudaError_enum.get(st, st))

check(cu.cuInit(0))
dev = cu.CUdevice(); check(cu.cuDeviceGet(ctypes.byref(dev), 0))
ctx = cu.CUcontext(); check(cu.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev))
check(cu.cuCtxSetCurrent(ctx))

cub = open('ptxas_cache/sm_75_f78969747d652dca.cubin', 'rb').read()
mod = cu.CUmodule(); check(cu.cuModuleLoadData(ctypes.byref(mod), cub))
f = cu.CUfunction(); check(cu.cuModuleGetFunction(ctypes.byref(f), mod, b'r_6_2_32_4_2_2_2_4_4_384'))

NBUF = 12
bufs = []
for i in range(NBUF):
    check(cu.cuCtxSetCurrent(ctx))
    p = cu.CUdeviceptr(); check(cu.cuMemAlloc_v2(ctypes.byref(p), 1 << 28)); bufs.append(p)

# params: 12 u64 pointers -> array of pointers-to-pointer
vals = (ctypes.c_uint64 * NBUF)(*[b.value for b in bufs])
params = (ctypes.c_void_p * NBUF)(*[ctypes.cast(ctypes.byref(vals, i * 8), ctypes.c_void_p) for i in range(NBUF)])

t = []
for i in range(12):
    t0 = time.perf_counter()
    check(cu.cuCtxSetCurrent(ctx))
    check(cu.cuLaunchKernel(f, 2, 6, 1, 32, 4, 1, 0, None, params, None))
    check(cu.cuCtxSynchronize())
    t.append(time.perf_counter() - t0)
print('ms:', [f'{x*1e3:.1f}' for x in t])
