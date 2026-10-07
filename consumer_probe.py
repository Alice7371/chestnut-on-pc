"""Run the EXACT WMMA consumer sequence (staging + 4x mma.m16n8k8 + D reshuffle)
with known staged fragment values, compare vs the RDNA4 model semantics."""
import ctypes, numpy as np
from tinygrad.runtime.autogen import cuda as cu

def check(st):
    if st != 0: raise RuntimeError(st)

check(cu.cuInit(0))
dev = cu.CUdevice(); check(cu.cuDeviceGet(ctypes.byref(dev), 0))
ctx = cu.CUcontext(); check(cu.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev))
check(cu.cuCtxSetCurrent(ctx))

def kj(t, j):
    """RDNA4 fragment map: thread t elem j -> k index
    t<16:  j=0..3 -> k=j;     j=4..7 -> k=j+4
    t>=16: j=0..3 -> k=j+4;   j=4..7 -> k=j+8"""
    if j < 4: return j if t < 16 else j + 4
    return j + 4 if t < 16 else j + 8

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

consumer = """
  bar.warp.sync 0xffffffff;
  shr.u32 %r7, %r1, 2;
  shl.b32 %r8, %r7, 4;
  and.b32 %r9, %r1, 3;
  shr.u32 %a_0, %r9, 1;
  and.b32 %a_1, %a_0, 1;
  shl.b32 %a_2, %a_0, 3;
  shl.b32 %a_3, %r9, 2;
  mad.lo.u32 %r5, %a_1, 248, %a_3;
  sub.u32 %r5, %r5, %a_2;
  add.u32 %r5, %r5, %r8;
  mad.lo.u32 %r5, %r2, 2048, %r5;
  cvt.u64.u32 %rd4, %r5;
  add.s64 %rd4, %rd3, %rd4;
  add.u32 %r5, %r5, 512;
  cvt.u64.u32 %rd5, %r5;
  add.s64 %rd5, %rd3, %rd5;
  ld.shared.b32 %a_0, [%rd4+0];
  ld.shared.b32 %a_1, [%rd4+128];
  ld.shared.b32 %a_2, [%rd4+8];
  ld.shared.b32 %a_3, [%rd4+136];
  ld.shared.b32 %q_0, [%rd5+0];
  ld.shared.b32 %q_1, [%rd5+128];
  ld.shared.b32 %q_2, [%rd5+8];
  ld.shared.b32 %q_3, [%rd5+136];
  mov.f32 %wz, 0f00000000;
  mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%d_0, %d_1, %d_2, %d_3}, {%a_0, %a_1}, {%q_0}, {%wz, %wz, %wz, %wz};
  mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%g_0, %g_1, %g_2, %g_3}, {%a_0, %a_1}, {%q_1}, {%wz, %wz, %wz, %wz};
  mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%d_0, %d_1, %d_2, %d_3}, {%a_2, %a_3}, {%q_2}, {%d_0, %d_1, %d_2, %d_3};
  mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%g_0, %g_1, %g_2, %g_3}, {%a_2, %a_3}, {%q_3}, {%g_0, %g_1, %g_2, %g_3};
  shr.u32 %r7, %r1, 2;
  shl.b32 %r8, %r7, 6;
  and.b32 %r9, %r1, 3;
  shl.b32 %r9, %r9, 3;
  add.u32 %r8, %r8, %r9;
  mad.lo.u32 %r8, %r2, 2048, %r8;
  add.u32 %r8, %r8, 1024;
  cvt.u64.u32 %rd6, %r8;
  add.s64 %rd6, %rd3, %rd6;
  st.shared.f32 [%rd6+0], %d_0;
  st.shared.f32 [%rd6+4], %d_1;
  st.shared.f32 [%rd6+512], %d_2;
  st.shared.f32 [%rd6+516], %d_3;
  st.shared.f32 [%rd6+32], %g_0;
  st.shared.f32 [%rd6+36], %g_1;
  st.shared.f32 [%rd6+544], %g_2;
  st.shared.f32 [%rd6+548], %g_3;
  bar.warp.sync 0xffffffff;
  shr.u32 %r7, %r1, 4;
  shl.b32 %r8, %r7, 9;
  and.b32 %r9, %r1, 15;
  shl.b32 %r9, %r9, 2;
  add.u32 %r8, %r8, %r9;
  mad.lo.u32 %r8, %r2, 2048, %r8;
  add.u32 %r8, %r8, 1024;
  cvt.u64.u32 %rd7, %r8;
  add.s64 %rd7, %rd3, %rd7;
"""
outs = []
for e in range(8):
    outs.append(f"""  ld.shared.f32 %d_{e}, [%rd7+{e*64}];
  mad.lo.u32 %r9, %r7, 128, 0;
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
  .reg .b32 %a_<4>; .reg .b32 %q_<4>; .reg .b32 %t6;
  .reg .f32 %d_<8>; .reg .f32 %g_<8>; .reg .f32 %wz;
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

# reference: staged value bits = lane*8+j+1 (per-wave lane = tid.x); thread lane
# elem j holds A[lane%16][kj], B[kj][lane%16]; both warps compute identical D
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
    print(f'CONSUMER BUG: lane {lane} e {e} -> D[{m}][{n}]: gpu={gpu:.6g} ref={ref:.6g}')
else:
    print('CONSUMER CORRECT')
