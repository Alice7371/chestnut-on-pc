"""Per-kernel cross-check between the CL (comma-verbatim, correct) and CUDA
(re-rendered) pipelines: run one replay frame, hash every kernel arg buffer
after each launch, dump to xcheck_<dev>.json; --compare diffs the two."""
import sys, json, math, ctypes, hashlib, os
import numpy as np

ARGV = list(sys.argv)
MODE = 'xmag' if '--xmag' in ARGV else ('sens' if '--sens' in ARGV else ('compare' if '--compare' in ARGV else 'run'))
sys.argv = [a for a in sys.argv if not a.startswith('--')]
DEV = os.environ.get('TG_DEV', 'CL')

if MODE == 'run':
    src = open('run_pinned2.py', encoding='utf-8').read()
    exec(compile(src.split('# --- smoke run')[0], 'run_pinned2.py', 'exec'))

    import tinygrad.runtime.ops_cuda as _oca
    from tinygrad.runtime import ops_cl as _ocl
    from tinygrad.device import _to_np_dtype
    import tinygrad.uop.ops as _uops
    from tinygrad.uop.ops import Ops as _Ops
    from tinygrad import Device as _D

    REC = {'on': False, 'rows': []}
    from tinygrad.device import Buffer as _Buf
    _GBUF = []   # Buffer objects seen via get_buf since last launch
    _orig_get_buf = _Buf.get_buf
    def _gb(self, device):
        r = _orig_get_buf(self, device)
        if REC['on']: _GBUF.append(self)
        return r
    _Buf.get_buf = _gb
    # buffers live on the pickled 'QCOM:0' device string (aliased to the real
    # backend) — must read through THAT instance's queue/allocator
    _aliasdev = _D['QCOM:0']

    def _read_cuda(ptr, nbytes):
        mv = memoryview(bytearray(nbytes))
        _aliasdev.allocator._copyout(mv, getattr(ptr, 'value', ptr))
        return mv

    def _read_cl(buf, nbytes):
        h = ctypes.create_string_buffer(nbytes)
        _ocl.check(_ocl.cl.clEnqueueReadBuffer(_aliasdev.queue, buf, True, 0, nbytes, _ocl.from_mv(h), 0, None, None))
        return memoryview(h)

    def _read_buf_bytes(buf):
        mv = memoryview(bytearray(buf.nbytes))
        if DEV == 'CUDA':
            _aliasdev.allocator._copyout(mv, buf._buf)
        else:
            h = ctypes.create_string_buffer(buf.nbytes)
            _ocl.check(_ocl.cl.clEnqueueReadBuffer(_aliasdev.queue, buf._buf, True, 0, buf.nbytes, _ocl.from_mv(h), 0, None, None))
            mv = memoryview(h)
        return mv

    XDump = os.environ.get('XDump') == '1'
    XINJECT = os.environ.get('XINJECT') == '1'
    _dump = []
    _inj = None
    _launch_idx = [0]
    if XINJECT:
        import pickle as _pk
        _inj = _pk.load(open(os.environ.get('XDUMPF', 'xcheck_CL_predump.pkl'), 'rb'))

    def _record(prg, bufs, readback, when):
        if not REC['on']: return
        name = getattr(prg, 'name', getattr(prg, '_cl_name', '?'))
        tbufs = _GBUF[:]
        _GBUF.clear()
        hashes, absm, prev, tmeta, probs = [], [], [], [], []
        for (_, slot, dt, shape) in prg.signature:
            if slot >= len(bufs): continue
            tb = tbufs[slot] if slot < len(tbufs) else None
            if when == 'pre' and tb is not None:
                tmeta.append(f'{tb.nbytes}B@{tb.device}')
            nbytes = math.prod(shape) * dt.itemsize
            try:
                mv = readback(bufs[slot], nbytes)
            except Exception as e:
                if when == 'pre': print(f'[{name}] arg{slot} READBACK FAIL: {e}')
                hashes.append(f'ERR:{str(e)[:40]}'); absm.append(-1.0); prev.append([]); continue
            hashes.append(hashlib.md5(mv).hexdigest()[:12])
            arr = np.frombuffer(mv, dtype=_to_np_dtype(dt))
            absm.append(float(np.abs(np.nan_to_num(arr, posinf=1e30, neginf=-1e30)).max()) if arr.size else 0.0)
            prev.append([round(float(x), 4) for x in arr.flatten()[:4]])
            fa = np.abs(arr.astype(np.float64))
            _N = fa.size
            probe = [round(float(x), 7) for x in fa[::max(1, _N // 64)][:64]] if _N else []
            probs.append(probe)
        REC['rows'].append({'n': name, 'when': when, 'h': hashes, 'a': absm, 'p': prev, 't': tmeta, 'b': probs})
        if when == 'pre' and XDump and REC['on']:
            raws = []
            for (_, slot, dt, shape) in prg.signature:
                if slot >= len(bufs): raws.append(None); continue
                nbytes = math.prod(shape) * dt.itemsize
                try: raws.append(bytes(readback(bufs[slot], nbytes)))
                except Exception: raws.append(None)
            _dump.append(raws)
        if when == 'post' and REC['on']:
            _launch_idx[0] += 1
        if when == 'post' and os.environ.get('XSNAP') and _launch_idx[0] - 1 == int(os.environ['XSNAP']):
            import pickle as _pk
            raws = []
            for (_, slot, dt, shape) in prg.signature:
                if slot >= len(bufs): raws.append(None); continue
                nbytes = math.prod(shape) * dt.itemsize
                try: raws.append(bytes(readback(bufs[slot], nbytes)))
                except Exception: raws.append(None)
            _pk.dump(raws, open(f'xcheck_snap_{DEV}.pkl', 'wb'))
            print(f'snapped launch {_launch_idx[0]-1} ({name})')
        if when == 'pre' and XINJECT and REC['on']:
            idx = _launch_idx[0]
            if idx < len(_inj) and (not os.environ.get('XSNAP') or idx == int(os.environ['XSNAP'])):
                for (_, slot, dt, shape), clb in zip(prg.signature, _inj[idx]):
                    if clb is None or slot >= len(bufs): continue
                    try:
                        if DEV == 'CUDA':
                            _aliasdev.allocator._copyin(bufs[slot], memoryview(clb))
                        else:
                            _ocl.check(_ocl.cl.clEnqueueWriteBuffer(_aliasdev.queue, bufs[slot], True, 0, len(clb), _ocl.from_mv(memoryview(bytearray(clb))), 0, None, None))
                        chk = bytes(readback(bufs[slot], len(clb)))
                        if chk[:64] != clb[:64]:
                            print(f'[INJ-VERIFY FAIL] [{idx}] arg{slot}: wrote {clb[:16].hex()} got {chk[:16].hex()}')
                    except Exception as e: print(f'inject fail [{idx}] arg{slot}: {e}')

    _orig_cuda_call = _oca.CUDAProgram.__call__
    def _cuda_call(self, *a, **kw):
        _record(self, a, _read_cuda, 'pre')
        out = _orig_cuda_call(self, *a, **kw)
        _record(self, a, _read_cuda, 'post')
        return out
    _oca.CUDAProgram.__call__ = _cuda_call

    _orig_cl_call = _ocl.CLProgram.__call__
    def _cl_call_wrap(self, *a, **kw):
        _record(self, a, _read_cl, 'pre')
        out = _orig_cl_call(self, *a, **kw)
        _record(self, a, _read_cl, 'post')
        return out
    _ocl.CLProgram.__call__ = _cl_call_wrap

    # one warm (capture) + one recorded (replay) frame
    p0 = np.zeros(524, np.float32)
    out = run_frame(make_warped(128), p0)

    inputs_meta = {}
    for u, buf in list(_uops.buffers.items()):
        if u.op is not _Ops.BUFFER: continue
        try:
            mv = _read_buf_bytes(buf)
        except Exception:
            continue
        u8 = np.frombuffer(mv, dtype=np.uint8)
        ent = {'u8absmax': int(u8.max()) if u8.size else 0,
               'md5': hashlib.md5(mv).hexdigest()[:12]}
        if buf.nbytes % 4 == 0:
            arr = np.frombuffer(mv, dtype=np.float32)
            ent['f32absmax'] = round(float(np.abs(arr).max()), 3) if arr.size else 0
        inputs_meta[buf.nbytes] = ent
    json.dump(inputs_meta, open(f'xcheck_{DEV}_inputs.json', 'w'), indent=1)

    REC['on'] = True
    _val = int(os.environ.get('XVAL', '77'))
    out = run_frame(make_warped(_val), p0)   # DIFFERENT input: replay must see it
    REC['on'] = False
    if XDump:
        import pickle as _pk
        _pk.dump(_dump, open('xcheck_CL_predump.pkl', 'wb'))
        print(f'dumped {len(_dump)} pre-states')
    json.dump(REC['rows'], open(f'xcheck_{DEV}{os.environ.get('XTAG','')}.json', 'w'))
    print(f'{DEV}: recorded {len(REC["rows"])} kernel launches -> xcheck_{DEV}{os.environ.get('XTAG','')}.json')
    print('out absmax', round(float(np.abs(out).max()), 4), 'finite', bool(np.isfinite(out).all()))

elif MODE == 'xmag':
    def pairs2(fn):
        rows = json.load(open(fn))
        pres = [r for r in rows if r['when'] == 'pre']
        posts = [r for r in rows if r['when'] == 'post']
        return list(zip(pres, posts))
    import numpy as _np
    A = pairs2('xcheck_CL_77.json'); B = pairs2('xcheck_CUDA_77.json')
    for i, ((pra, poa), (prb, pob)) in enumerate(zip(A, B)):
        sa = _np.array([x for p in poa['b'] for x in p])
        sb = _np.array([x for p in pob['b'] for x in p])
        ma = float(_np.nanmax(sa)) if sa.size else -1
        mb = float(_np.nanmax(sb)) if sb.size else -1
        ratio = mb/ma if ma and ma > 0 and mb == mb else float('nan')
        flag = '  <<< DIVERGES' if (ma == ma and mb == mb and ma > 0 and (mb > ma*3 or mb < ma/3)) else ''
        print(f'[{i:3}] {pra["n"][:34]:34} cl={ma:.4g} cuda={mb:.4g} ratio={ratio:.3g}{flag}')
elif MODE == 'sens':
    d = ARGV[-1] if ARGV[-1] in ('CUDA','CL') else 'CUDA'
    def pairs(fn):
        rows = json.load(open(fn))
        pre = iter([r for r in rows if r['when']=='pre'])
        post = iter([r for r in rows if r['when']=='post'])
        pres = [r for r in rows if r['when'] == 'pre']
        posts = [r for r in rows if r['when'] == 'post']
        return list(zip(pres, posts))
    pa = pairs(f'xcheck_{d}_77.json')
    pb = pairs(f'xcheck_{d}_128.json')
    print(f'{d}: 77 vs 128 — {len(pa)} launches')
    import numpy as _np
    prev_d = None
    for i, ((pra, poa), (prb, pob)) in enumerate(zip(pa, pb)):
        if pra['n'] != prb['n']: print(f'[{i}] name mismatch'); break
        sa = _np.array([x for p in poa['b'] for x in p])
        sb = _np.array([x for p in pob['b'] for x in p])
        d = float(_np.nanmax(_np.abs(sa - sb))) if sa.size and sa.size == sb.size else -1.0
        m77 = float(_np.nanmax(sa)) if sa.size else -1.0
        m128 = float(_np.nanmax(sb)) if sb.size else -1.0
        bad = '' if (_np.isfinite(sa).all() and _np.isfinite(sb).all()) else '  <<< NON-FINITE ACTIVATION'
        if bad or (i % 1 == 0 and i >= 30):
            print(f'[{i:3}] {pra["n"][:34]:34} sens={d:.3e} mag77={m77:.4g} mag128={m128:.4g}{bad}')
        prev_d = d
    else:
        pass
else:
    def pairs(fn):
        rows = json.load(open(fn))
        pres = [r for r in rows if r['when'] == 'pre']
        posts = [r for r in rows if r['when'] == 'post']
        return list(zip(pres, posts))
    A = pairs('xcheck_CL.json'); B = pairs('xcheck_CUDA.json')
    print(f'CL launches={len(A)}  CUDA launches={len(B)}')
    for i, ((pra, poa), (prb, pob)) in enumerate(zip(A, B)):
        if pra['n'] != prb['n']:
            print(f'[{i}] NAME MISMATCH cl={pra["n"]} cuda={prb["n"]}'); break
        import numpy as _np
        sa = _np.array([x for p in poa['b'] for x in p])
        sb = _np.array([x for p in pob['b'] for x in p])
        d = float(_np.nanmax(_np.abs(sa - sb))) if sa.size and sa.size == sb.size else -1.0
        if i < 60 or d > 1e-2:
            print(f'[{i:3}] {pra["n"][:34]:34} |cl-cuda|={d:.3e}')
    else:
        print('done')
