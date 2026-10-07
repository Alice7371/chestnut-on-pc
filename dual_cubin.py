"""Launch the SAME dumped inputs through the mma and ffma r_6_2 cubins, compare outputs."""
import ctypes, glob, numpy as np
from tinygrad.runtime.autogen import cuda as cu

def check(st):
    if st != 0: raise RuntimeError(st)

check(cu.cuInit(0))
dev = cu.CUdevice(); check(cu.cuDeviceGet(ctypes.byref(dev), 0))
ctx = cu.CUcontext(); check(cu.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev))
check(cu.cuCtxSetCurrent(ctx))

# inputs from the dump (capped at 4MB each; zeros beyond — identical for both runs)
inputs = []
for f in sorted(glob.glob('kd_84_*.bin')):
    tag = f.split('kd_84_')[1].rstrip('.bin')
    size = int(tag.split('_')[1])
    data = np.fromfile(f, dtype=np.uint8)
    buf = np.zeros(size, np.uint8)
    buf[:len(data)] = data
    inputs.append(buf)
print('input sizes:', [len(b) for b in inputs])

def run_cubin(cubin_path):
    mod = cu.CUmodule(); check(cu.cuModuleLoadData(ctypes.byref(mod), open(cubin_path, 'rb').read()))
    f = cu.CUfunction(); check(cu.cuModuleGetFunction(ctypes.byref(f), mod, b'r_6_2_32_4_2_2_2_4_4_384'))
    devptrs = []
    for b in inputs:
        p = cu.CUdeviceptr(); check(cu.cuMemAlloc_v2(ctypes.byref(p), len(b)))
        check(cu.cuMemcpyHtoD_v2(p, b.ctypes.data_as(ctypes.c_void_p), len(b)))
        devptrs.append(p)
    vals = (ctypes.c_uint64 * 6)(*[d.value for d in devptrs])
    params = (ctypes.c_void_p * 6)(*[ctypes.cast(ctypes.byref(vals, i*8), ctypes.c_void_p) for i in range(6)])
    check(cu.cuLaunchKernel(f, 2, 6, 1, 32, 4, 1, 0, None, params, None))
    check(cu.cuCtxSynchronize())
    outs = []
    for d, b in zip(devptrs, inputs):
        o = np.zeros(len(b), np.uint8)
        check(cu.cuMemcpyDtoH_v2(o.ctypes.data_as(ctypes.c_void_p), d, len(b)))
        outs.append(o)
    return outs

mma_out = run_cubin('ptxas_cache/sm_75_4feeeadb0dcbb514.cubin')
ffma_out = run_cubin('ptxas_cache/sm_75_91cabf802382aa34.ptx' if False else 'ptxas_cache/sm_75_91cabf802382aa34.cubin')

for i, (a, b) in enumerate(zip(mma_out, ffma_out)):
    va = np.frombuffer(a, np.uint8)
    vb = np.frombuffer(b, np.uint8)
    diff = np.count_nonzero(va != vb)
    print(f'buf{i}: differing bytes = {diff}/{len(va)}')
