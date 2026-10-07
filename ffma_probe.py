"""Probe the FFMA consumer sequence (B-column cache + 128 FMA) vs the RDNA4 model."""
import ctypes, numpy as np
from tinygrad.runtime.autogen import cuda as cu

def check(st):
    if st != 0: raise RuntimeError(st)

check(cu.cuInit(0))
dev = cu.CUdevice(); check(cu.cuDeviceGet(ctypes.byref(dev), 0))
ctx = cu.CUcontext(); check(cu.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev))
check(cu.cuCtxSetCurrent(ctx))

def kj(t, j):
    if j < 4: return j if t < 16 else j + 4
    return j + 4 if t < 16 else j + 8

def ja(k): return k - (0, 4, 4, 8)[k // 4]
def gk(k): return (k // 4) % 2

stage_pairs = []
for j in range(0, 8, 2):
    stage_pairs.append(f"""
  mov.u32 %r5, %r4;
  add.u32 %r5, %r5, {j};
  cvt.u16.u32 %h_0, %r5;
  mov.u32 %r6, %r5;
  add.u32 %r6, %r6, 1;
  cvt.u16.u32 %h_1, %r6;
  mov.b32 %t6, {{%h_0, %h_1}};
  st.shared.b32 [%rd2+{j*2}], %t6;
  st.shared.b32 [%rd2+{512 + j*2}], %t6;""")

# FFMA consumer, transcribed from run_pinned2 _wm_emit FFMA branch
cons = ["  bar.warp.sync 0xffffffff;"]
cons.append("  mov.u32 %r7, %tid.y;")
cons.append("  shl.b32 %r7, %r7, 11;")
cons.append("  mov.u32 %r8, %tid.x;")
cons.append("  shr.u32 %r9, %r8, 4;")
cons.append("  shl.b32 %r9, %r9, 7;")
cons.append("  add.u32 %r9, %r7, %r9;")
cons.append("  cvt.u64.u32 %rd4, %r9;")
cons.append("  add.s64 %rd4, %rd3, %rd4;")          # wqA (A rows)
cons.append("  mov.u32 %r8, %tid.x;")
cons.append("  and.b32 %r8, %r8, 15;")
cons.append("  shl.b32 %r8, %r8, 4;")
cons.append("  add.u32 %r8, %r7, %r8;")
cons.append("  add.u32 %r8, %r8, 512;")
cons.append("  cvt.u64.u32 %rd5, %r8;")
cons.append("  add.s64 %rd5, %rd3, %rd5;")          # wqB (B cols)
# B column cache: bf[k] for k=0..15 -> %f_0..15
for k in range(16):
    cons.append(f"  ld.shared.b16 %h_0, [%rd5+{gk(k)*256 + ja(k)*2}];")
    cons.append(f"  cvt.f32.f16 %f_{k}, %h_0;")
# outputs: 8 outputs e (row m_e=(l>>4)*8+e), col n=l&15
for e in range(8):
    for k in range(16):
        tgt = f"%d_{e}" if k == 0 else f"%d_{e}"
        addend = "%wz" if k == 0 else f"%d_{e}"
        cons.append(f"  ld.shared.b16 %h_0, [%rd4+{e*16 + gk(k)*256 + ja(k)*2}];")
        cons.append(f"  cvt.f32.f16 %f_16, %h_0;")
        cons.append(f"  fma.rn.f32 %d_{e}, %f_16, %f_{k}, {addend};")
consumer = "\n".join(cons)
outs = []
for e in range(8):
    outs.append(f"""  shr.u32 %r8, %r1, 4;
  mad.lo.u32 %r9, %r8, 128, 0;
  add.u32 %r9, %r9, {e*16};
  and.b32 %r6, %r1, 15;
  add.u32 %r9, %r9, %r6;
  shl.b32 %r9, %r9, 2;
  cvt.u64.u32 %rd2, %r9;
  add.s64 %rd2, %rd1, %rd2;
  st.global.f32 [%rd2], %d_{e};""")

PTX = ("""
.version 7.0
.target sm_75
.address_size 64
.visible .entry probe(.param .u64 data0) {
  .reg .b64 %rd<8>;
  .reg .b32 %r<10>;
  .reg .b16 %h_<2>;
  .reg .b32 %t6;
  .reg .f32 %d_<8>; .reg .f32 %f_<17>; .reg .f32 %wz;
  .shared .align 4 .b8 _wm[4096];
  ld.param.u64 %rd1, [data0];
  mov.u32 %r1, %tid.x;
  mov.u32 %r2, %tid.y;
  mad.lo.u32 %r3, %r2, 128, %r1;
  shl.b32 %r3, %r3, 4;
  cvt.u64.u32 %rd2, %r3;
  mov.u64 %rd3, _wm;
  add.s64 %rd2, %rd3, %rd2;
  mul.lo.u32 %r4, %r1, 8;
  add.u32 %r4, %r4, 1;
""" + "\n".join(stage_pairs) + consumer + "\n".join(outs) + """
  ret;
}
""").encode()

mod = cu.CUmodule(); check(cu.cuModuleLoadData(ctypes.byref(mod), PTX))
f = cu.CUfunction(); check(cu.cuModuleGetFunction(ctypes.byref(f), mod, b'probe'))
out = np.zeros(256, np.float32)
p = cu.CUdeviceptr(); check(cu.cuMemAlloc_v2(ctypes.byref(p), out.nbytes))
check(cu.cuMemcpyHtoD_v2(p, out.ctypes.data_as(ctypes.c_void_p), out.nbytes))
vals = (ctypes.c_uint64 * 1)(p.value)
params = (ctypes.c_void_p * 1)(ctypes.cast(ctypes.byref(vals), ctypes.c_void_p))
check(cu.cuLaunchKernel(f, 1, 1, 1, 32, 2, 1, 0, None, params, None))
check(cu.cuCtxSynchronize())
check(cu.cuMemcpyDtoH_v2(out.ctypes.data_as(ctypes.c_void_p), p, out.nbytes))

vbits = np.zeros((32, 8), dtype=np.uint16)
for lane in range(32):
    for j in range(8): vbits[lane, j] = lane*8 + j + 1
vf = vbits.view(np.float16).astype(np.float64)
A = np.zeros((16, 16)); Bm = np.zeros((16, 16))
for lane in range(32):
    for j in range(8):
        k = kj(lane, j)
        A[lane % 16, k] = vf[lane, j]
        Bm[k, lane % 16] = vf[lane, j]
Dref = A @ Bm
err = 0.0
worst = None
for lane in range(32):
    for e in range(8):
        m, n = (lane//16)*8 + e, lane % 16
        gpu = out[(lane//16)*128 + e*16 + lane % 16]
        ref = Dref[m, n]
        rel = abs(gpu - ref) / max(1.0, abs(ref))
        if rel > err:
            err = rel
            worst = (lane, e, m, n, gpu, ref)
print(f'max relative error: {err:.2e}')
if err > 1e-5:
    lane, e, m, n, gpu, ref = worst
    print(f'FFMA BUG: lane {lane} e {e} -> D[{m}][{n}]: gpu={gpu:.6g} ref={ref:.6g}')
else:
    print('FFMA CONSUMER CORRECT')
