"""mma.m16n8k8 fragment layout probe: known A/B values -> compare vs numpy."""
import ctypes, numpy as np
from tinygrad.runtime.autogen import cuda as cu
def check(st):
    if st != 0: raise RuntimeError(st)
check(cu.cuInit(0))
dev = cu.CUdevice(); check(cu.cuDeviceGet(ctypes.byref(dev), 0))
ctx = cu.CUcontext(); check(cu.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev))
check(cu.cuCtxSetCurrent(ctx))
PTX = b"""
.version 7.0
.target sm_75
.address_size 64
.visible .entry probe(.param .u64 data0) {
  .reg .b64 %rd<4>;
  .reg .b32 %r<8>;
  .reg .b32 %a_<2>; .reg .b32 %b_<1>; .reg .f32 %d_<4>; .reg .b32 %r9<2>; .reg .pred %p_<2>; .reg .f32 %f5; 
  .reg .b16 %h_<4>;
  ld.param.u64 %rd1, [data0];
  mov.u32 %r1, %tid.x;
  shr.u32 %r2, %r1, 2;
  and.b32 %r3, %r1, 3;
  // A[r][k] = r*16+k : a0 = {A[r0][2c], A[r0][2c+1]}, a1 = row+8
  mad.lo.u32 %r4, %r2, 16, 0;
  shl.b32 %r5, %r3, 1;
  add.u32 %r4, %r4, %r5;
  add.u32 %r5, %r4, 1;
  cvt.u16.u32 %h_0, %r4;
  cvt.u16.u32 %h_1, %r5;
  mov.b32 %a_0, {%h_0, %h_1};
  add.u32 %r4, %r4, 128;
  add.u32 %r5, %r5, 128;
  cvt.u16.u32 %h_2, %r4;
  cvt.u16.u32 %h_3, %r5;
  mov.b32 %a_1, {%h_2, %h_3};
  // B[k][n] = n*16+k : b0 = {B[2c][r0], B[2c+1][r0]}
  mad.lo.u32 %r4, %r2, 16, 0;
  shl.b32 %r5, %r3, 1;
  add.u32 %r4, %r4, %r5;
  add.u32 %r5, %r4, 1;
  cvt.u16.u32 %h_0, %r4;
  cvt.u16.u32 %h_1, %r5;
  mov.b32 %b_0, {%h_0, %h_1};
  mov.f32 %d_0, 0f00000000;
  mov.f32 %d_1, 0f00000000;
  mov.f32 %d_2, 0f00000000;
  mov.f32 %d_3, 0f00000000;
  mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%d_0, %d_1, %d_2, %d_3}, {%a_0, %a_1}, {%b_0}, {%d_0, %d_1, %d_2, %d_3};
  // D store: lane l, en: row=(l>>2)+8*(en>>1), col=2*(l&3)+(en&1); f32 off = row*8+col
  mad.lo.u32 %r4, %r2, 8, 0;
  shl.b32 %r5, %r3, 1;
  add.u32 %r4, %r4, %r5;
  shl.b32 %r4, %r4, 2;
  cvt.u64.u32 %rd2, %r4;
  add.s64 %rd2, %rd2, %rd1;
  st.global.f32 [%rd2], %d_0;
  st.global.f32 [%rd2+4], %d_1;
  st.global.f32 [%rd2+32], %d_2;
  st.global.f32 [%rd2+36], %d_3;
  // debug: lane 0 stores a_0, b_0 raw + sentinel
  setp.eq.u32 %p_0, %r1, 0;
  @!%p_0 bra DONE;
  mov.b32 %r4, %a_0;
  cvt.rn.f32.u32 %f5, %r4;
  st.global.f32 [%rd2+4096], %f5;
  mov.b32 %r4, %b_0;
  cvt.rn.f32.u32 %f5, %r4;
  st.global.f32 [%rd2+4100], %f5;
  mov.f32 %f5, 0f42200000;
  st.global.f32 [%rd2+4104], %f5;
DONE:
  ret;
}
"""
mod = cu.CUmodule(); check(cu.cuModuleLoadData(ctypes.byref(mod), PTX))
f = cu.CUfunction(); check(cu.cuModuleGetFunction(ctypes.byref(f), mod, b'probe'))
out = np.zeros(2048, np.float32)
p = cu.CUdeviceptr(); check(cu.cuMemAlloc_v2(ctypes.byref(p), out.nbytes))
check(cu.cuMemcpyHtoD_v2(p, out.ctypes.data_as(ctypes.c_void_p), out.nbytes))
vals = (ctypes.c_uint64 * 1)(p.value)
params = (ctypes.c_void_p * 1)(ctypes.cast(ctypes.byref(vals), ctypes.c_void_p))
check(cu.cuLaunchKernel(f, 1, 1, 1, 32, 1, 1, 0, None, params, None))
check(cu.cuCtxSynchronize())
check(cu.cuMemcpyDtoH_v2(out.ctypes.data_as(ctypes.c_void_p), p, out.nbytes))

_abits = np.arange(256, dtype=np.uint16)
_af16 = _abits.view(np.float16).astype(np.float64)   # f16 value of bit-pattern i
A = np.empty((16,8)); B = np.empty((8,8))
for r in range(16):
    for k in range(8): A[r,k] = _af16[r*16+k]
for k in range(8):
    for n in range(8): B[k,n] = _af16[n*16+k]
D = A @ B
ok = True
for l in range(32):
    r0, c = l>>2, l&3
    for en, doff in ((0,0),(1,1),(2,8),(3,9)):
        row, col = r0 + 8*(en>>1), c*2 + (en&1)
        gpu = out[r0*8 + c*2 + doff]
        ref = D[row, col]
        if abs(gpu - ref) > 0.5:
            ok = False; print(f'  worst so far: lane {l} en {en}: gpu={gpu:.3e} ref={ref:.3e}')
print(f'debug: a0raw={out[1024]:.0f} b0raw={out[1025]:.0f} sentinel={out[1026]:.0f}')
if ok:
    print('D layout MATCHES: a0={A[l>>2][2(l&3)],+1}, a1=row+8, b0={B[2(l&3)][l>>2],+1}, D c_map=(en%2+(l%4)*2, l>>4... wait row=(l>>2)+8*(en>>1))')
else:
    print('D layout MISMATCH — dump:')
    print(out[:16])
