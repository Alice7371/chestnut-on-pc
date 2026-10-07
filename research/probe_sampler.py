"""Probe comma's exact sampler semantics on the CL device: what does
read_imagef with CLK_ADDRESS_CLAMP return for OOB coordinates?"""
import ctypes
import numpy as np
from tinygrad import Device as _D
from tinygrad.runtime import ops_cl as _ocl
from tinygrad.runtime.autogen import opencl as cl

dev = _D['CL']
q = dev.queue

src = '''
__kernel void probe(__global float* out, __read_only image2d_t img) {
  const sampler_t smp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
  float4 a = read_imagef(img, smp, (int2)(-1, 0));
  float4 b = read_imagef(img, smp, (int2)(2, 0));
  float4 c = read_imagef(img, smp, (int2)(0, -1));
  float4 d = read_imagef(img, smp, (int2)(0, 1));
  float4 e = read_imagef(img, smp, (int2)(0, 0));
  float4 f = read_imagef(img, smp, (int2)(1, 0));
  vstore4(a, 0, out); vstore4(b, 1, out); vstore4(c, 2, out);
  vstore4(d, 3, out); vstore4(e, 4, out); vstore4(f, 5, out);
}
'''
st = ctypes.c_int32()
prog = _ocl.checked(cl.clCreateProgramWithSource(dev.context, 1, _ocl.to_char_p_p([src.encode()]), None, st), st)
_ocl.check(cl.clBuildProgram(prog, 1, dev.cl_dev, None, _ocl.BP_CB(), None))
k = _ocl.checked(cl.clCreateKernel(prog, b'probe', st), st)

fmt = cl.cl_image_format(cl.CL_RGBA, cl.CL_FLOAT)
desc = cl.cl_image_desc(cl.CL_MEM_OBJECT_IMAGE2D, 2, 1)
img = _ocl.checked(cl.clCreateImage(dev.context, cl.CL_MEM_READ_WRITE, fmt, desc, None, st), st)
data = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float32)
origin = (cl.size_t * 3)(0, 0, 0)
region = (cl.size_t * 3)(2, 1, 1)
# proven path: buffer -> image copy (same as the shim in run_pinned2)
buf = _ocl.checked(cl.clCreateBuffer(dev.context, cl.CL_MEM_READ_WRITE | cl.CL_MEM_COPY_HOST_PTR, 32, _ocl.from_mv(memoryview(bytearray(data.tobytes()))), st), st)
_ocl.check(cl.clEnqueueCopyBufferToImage(q, buf, img, 0, origin, region, 0, None, None))
dev.synchronize()
chk = np.zeros(8, dtype=np.float32)
_ocl.check(cl.clEnqueueReadImage(q, img, True, origin, region, 0, 0, _ocl.from_mv(memoryview(chk)), 0, None, None))
print('image write/readback:', chk)

obuf = _ocl.checked(cl.clCreateBuffer(dev.context, cl.CL_MEM_READ_WRITE, 96, None, st), st)
_ocl.check(cl.clSetKernelArg(k, 0, ctypes.sizeof(obuf), ctypes.byref(obuf)))
_ocl.check(cl.clSetKernelArg(k, 1, ctypes.sizeof(img), ctypes.byref(img)))
_ocl.check(cl.clEnqueueNDRangeKernel(q, k, 1, None, (cl.size_t * 1)(1), None, 0, None, None))

out = np.zeros(24, dtype=np.float32)
_ocl.check(cl.clEnqueueReadBuffer(q, obuf, True, 0, 96, _ocl.from_mv(memoryview(out)), 0, None, None))

names = ['x=-1 (left OOB) ', 'x=2  (right OOB)', 'y=-1 (top OOB)  ', 'y=1  (bottom OOB)', '(0,0) in-range  ', '(1,0) in-range  ']
for i, nm in enumerate(names):
    print(nm, out[i*4:(i+1)*4])
