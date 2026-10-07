import io, struct, pickle, sys

# stub non-existent device runtimes so unpickling QCOM/AMD buffers doesn't touch real hardware
import sys as _sys, types as _types

class _Universal:
    def __call__(self, *a, **k): return _universal
    def __getattr__(self, name): return _universal
_universal = _Universal()

class DummyAllocator:
    def alloc(self, nbytes, options=None): return _universal
    def __getattr__(self, name): return _universal

class DummyDev:
    allocator = DummyAllocator()
    def __init__(self, device): self.device = device
    def finalize(self): pass
    def synchronize(self): pass

for _x in ('qcom', 'amd', 'cuda', 'gpu', 'hexagon', 'dsp'):
    _m = _types.ModuleType(f'tinygrad.runtime.ops_{_x}')
    setattr(_m, _x.upper() + 'Device', type(_x.upper() + 'Device', (DummyDev,), {}))
    _sys.modules[_m.__name__] = _m
import tinygrad.device as tdev  # noqa: E402
import tinygrad.engine.realize as _realize  # noqa: E402
_realize.run_linear = lambda *a, **k: None  # no-op all device copies during unpickle

class StubUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except Exception:
            dummy = type(name, (), {'__setstate__': lambda self, s: None})
            return dummy

def load_oob_from(data):
    n = struct.unpack('<q', data[:8])[0]
    stream = io.BytesIO(data[8:8+n])
    buf = io.BytesIO(data[8+n:])
    def buffers():
        while True:
            h = buf.read(8)
            if not h:
                return
            (nb,) = struct.unpack('<q', h)
            yield pickle.PickleBuffer(buf.read(nb))
    return StubUnpickler(stream, buffers=buffers()).load()

path = sys.argv[1]
data = open(path, 'rb').read()
n = struct.unpack('<q', data[:8])[0]
print(f'opcode stream: {n} bytes, total file: {len(data)} bytes')

obj = load_oob_from(data)
print('top-level type:', type(obj))
meta = None
if isinstance(obj, dict):
    for k, v in obj.items():
        print(f'  key: {k!r:24} type: {type(v).__name__}')
    meta = obj.get('metadata')
if meta is None:
    sys.exit(0)
print('\nmetadata keys:', list(meta.keys()))
print('\ninput_shapes:')
for k, v in meta['input_shapes'].items():
    print(f'  {k:20} {v}')
print('\noutput_slices:')
for k, v in meta['output_slices'].items():
    print(f'  {k:24} {v.start:>6}..{v.stop}')
for k in meta:
    if k not in ('input_shapes', 'output_slices'):
        v = meta[k]
        print(f'{k}: {str(v)[:200]}')
