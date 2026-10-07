"""Capture ground truth for the r_6_2_32_4_2_2_2_4_4_384 ALU kernel: launch dims,
pre-state buffers, linear uops. Dumps to ru62_meta.json + ru62_pre.npz +
ru62_uops.pkl so bench variants run offline without booting the engine."""
import os, sys, time, ctypes, struct, io, pickle, json
import numpy as np

os.environ.setdefault('TG_DEV', 'CUDA')
KN = 'r_6_2_32_4_2_2_2_4_4_384'
sys.argv = ['x', 'E:/models/openpilot/big_driving.pkl']
src = open('run_pinned2.py', encoding='utf-8').read()
exec(compile(src.split('# --- smoke run')[0], 'run_pinned2.py', 'exec'))
import tinygrad.runtime.autogen.cuda as _cug
import tinygrad.runtime.ops_cuda as _oc

assert KN in _LIN_BY_NAME, f'{KN} not stashed'
lin_uops, local_size = _LIN_BY_NAME[KN]
print(f'[ru62] {KN}: {len(lin_uops)} uops stashed, local={local_size}', flush=True)

_ct = ctypes
_captured = {}
_cp_call = _oc.CUDAProgram.__call__   # recording spy installed by run_pinned2

def _copyout(ptr, nb):
    host = bytearray(nb)
    _oc.check(_cug.cuMemcpyDtoH_v2(_ct.cast((_ct.c_char * nb).from_buffer(host), _ct.c_void_p), ptr, nb))
    return host

def _arg_span(ptr):
    _base = _cug.CUdeviceptr(); _sz = _ct.c_size_t()
    _oc.check(_cug.cuMemGetAddressRange_v2(_ct.byref(_base), _ct.byref(_sz), ptr))
    off = ptr.value - _base.value if hasattr(ptr, 'value') else int(ptr) - _base.value
    return max(0, _sz.value - off)

def _hook(self, *a, **kw):
    if self.name == KN:
        if 'pre' not in _captured:
            _captured['pre'] = [_copyout(p, _arg_span(p)) for p in a]
            _captured['kw'] = dict(kw)
            _captured['nargs'] = len(a)
        _captured.setdefault('all_kw', []).append(
            (list(kw.get('global_size', ())), list(kw.get('local_size', ()))))
        if len(_captured['all_kw']) == 1:
            r = _cp_call(self, *a, **kw)
            _oc.check(_cug.cuCtxSynchronize())
            print(f'[ru62] captured: args={len(a)} kw={ {k: v for k, v in kw.items() if k != "vals"} }', flush=True)
            return r
    return _cp_call(self, *a, **kw)

_oc.CUDAProgram.__call__ = _hook
run_frame(np.full((2, 6, img[2], img[3]), 111, np.uint8), np.zeros(524, np.float32))
assert 'pre' in _captured, 'target kernel never launched'

kw = _captured['kw']
meta = {
    'name': KN, 'nargs': _captured['nargs'],
    'global_size': list(kw.get('global_size', ())),
    'local_size': list(kw.get('local_size', ())),
    'vals': list(kw.get('vals', ())),
    'all_launches': _captured.get('all_kw'),
    'signature': [[s[0], s[1], s[2].name, list(s[3])] for s in _prg_sig] if (_prg_sig := None) else None,
    'spans': [len(h) for h in _captured['pre']],
}
# op histogram of the linear uops
from collections import Counter
hist = Counter(u.op.name for u in lin_uops)
meta['uop_hist'] = dict(hist)
meta['n_uops'] = len(lin_uops)
open('ru62_meta.json', 'w').write(json.dumps(meta, indent=1, default=str))
np.savez('ru62_pre.npz', **{f'arg{i}': np.frombuffer(h, np.uint8) for i, h in enumerate(_captured['pre'])})
pickle.dump(lin_uops, open('ru62_uops.pkl', 'wb'))
print('[ru62] dumps written: ru62_meta.json ru62_pre.npz ru62_uops.pkl', flush=True)
print(json.dumps(meta, indent=1, default=str), flush=True)
