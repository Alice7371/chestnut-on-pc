"""Warp-split transform for r_32_32_3_12288_32.

The kernel is a single-warp 12288-iteration serial reduce loop (step 1, bound
12288) -> ~6.8ms because 32 blocks x 1 warp leaves the whole GPU idle. This
transform spreads the loop across W warps per block (local_size y 1->W): each
warp handles k in {w, w+W, w+2W, ...}, dumps its chain partials to shared,
and warp 0 sums the partials before the unchanged phase-2 epilogue + stores.
Summation order changes (f32 reassociation, same noise class as the
already-qualified mma reassociation).

Usage: python alu_opt.py <src.ptx> <out.cubin> <W>
"""
import re, sys, subprocess, os

src_ptx, out_cub, W = sys.argv[1], sys.argv[2], int(sys.argv[3])
lines = open(src_ptx).read().split('\n')

# --- parse: def map reg -> line idx (first def wins) ---
defs = {}
for i, ln in enumerate(lines):
    m = re.match(r'\s*([\w.]+)\s+(%[\w.]+)\s*,', ln)
    if m and m.group(2) not in defs:
        defs[m.group(2)] = i

def op_of(i):
    return re.match(r'\s*([\w.]+)', lines[i]).group(1)

def srcs_of(i):
    body = lines[i].split('\t', 1)[1].rsplit(';', 1)[0]
    return re.findall(r'%[\w.]+', body)

# --- backward walk from the three stored values to the chain bases ---
# descend through arithmetic + conversions (incl f16 quantize path: the loop
# accumulators enter phase 2 via cvt.rn.f16.f32 then mul.f16); stop at loads,
# moves (loop phis) and anything else
DESCEND = {'add.f32', 'mul.f32', 'sub.f32', 'fma.rn.f32',
           'mul.f16', 'add.f16', 'cvt.f32.f16', 'cvt.rn.f16.f32'}

def bases_from(reg, seen):
    if reg in seen: return []
    seen.add(reg)
    i = defs.get(reg)
    if i is None: return [reg]
    if op_of(i) in DESCEND:
        out = []
        for s in srcs_of(i):
            if s.startswith('%const_'): continue
            out += bases_from(s, seen)
        return out
    return [reg]

stored = ['%alu_f32_190', '%alu_f32_191', '%alu_f32_189']
bases = []
for s in stored:
    for b in bases_from(s, set()):
        if b.startswith('%reg_f32') and b not in bases:
            bases.append(b)
print(f'[alu_opt] chain bases ({len(bases)}): {bases if len(bases) < 12 else "[reg_f32 x%d]" % len(bases)}')
NB = len(bases)

# --- locate anchors ---
i_init = next(i for i, l in enumerate(lines) if 'mov.u32' in l and '%ridx_s32_0, -1;' in l)
i_end = next(i for i, l in enumerate(lines) if l.strip() == 'END_ridx_s32_0:')
i_step = next(i for i, l in enumerate(lines) if 'add.s32' in l and '%ridx_s32_0, %ridx_s32_0, 1;' in l)
# insertion point: AFTER the loop-back branch (the partial-store + reduce must
# run once, only when the loop has fully exited), before the epilogue
i_loopback = next(i for i, l in enumerate(lines) if 'bra' in l and 'LOOP_ridx_s32_0' in l)
assert i_init < i_end < i_step < i_loopback

# --- shared region: W warps x 32 lanes x NB f32 ---
WS_BYTES = W * 32 * NB * 4
i_decl = next(i for i, l in enumerate(lines) if '.shared' in l)
# replace the dead _wm[4096] decl (no ld/st.shared reference it) with our
# partial buffer, so W=4's 48KB fits under the static shared limit
decl = f'\t.shared\t\t.align 4 .b8 _ws[{WS_BYTES}];'
newregs = ('\t.reg \t.u32 %ws_w;\n\t.reg \t.u32 %ws_off;\n\t.reg \t.u32 %ws_l;\n'
           '\t.reg \t.u64 %ws_p;\n\t.reg \t.u64 %ws_p2;\n'
           '\t.reg \t.f32 %ws_t;\n\t.reg \t.pred %ws_p0;')

store_part = [
    '\tmov.u32\t\t%ws_w, %tid.y;',
    '\tmov.u32\t\t%ws_off, %tid.x;',
    '\tmul.lo.u32\t%ws_l, %ws_off, ' + str(NB*4) + ';',
    '\tmad.lo.u32\t%ws_off, %ws_w, 32, %ws_off;',
    f'\tmul.lo.u32\t%ws_off, %ws_off, {NB*4};',
    '\tcvt.u64.u32\t%ws_p, %ws_off;',
    '\tmov.u64\t\t%ws_p2, _ws;',
    '\tadd.s64\t\t%ws_p, %ws_p2, %ws_p;',
]
store_part += [f'\tst.shared.f32\t[%ws_p+{j*4}], {b};' for j, b in enumerate(bases)]
store_part += [
    '\tbar.sync\t\t0;',
    '\tsetp.gt.s32\t%ws_p0, %ws_w, 0;',
    '\t@%ws_p0\tbra\t\tWS_DONE;',
    '\tcvt.u64.u32\t%ws_p, %ws_l;',
    '\tadd.s64\t\t%ws_p, %ws_p2, %ws_p;',
]
for w in range(1, W):
    for j, b in enumerate(bases):
        store_part.append(f'\tld.shared.f32\t%ws_t, [%ws_p+{w*32*NB*4 + j*4}];')
        store_part.append(f'\tadd.f32\t\t{b}, {b}, %ws_t;')

out = list(lines)
out[i_decl] = decl + '\n' + newregs
out[i_init] = (f'\tmov.u32\t\t%ridx_s32_0, %tid.y;\n'
               f'\tadd.s32\t\t%ridx_s32_0, %ridx_s32_0, -{W};')
out[i_step] = lines[i_step].replace(', 1;', f', {W};')
text = '\n'.join(out)
text = text.replace('.maxntid 32', f'.maxntid {32*W}')
text = text.replace(lines[i_loopback], lines[i_loopback] + '\n' + '\n'.join(store_part), 1)
text = text.replace('\tret;', '\tWS_DONE:\n\tret;', 1)
open(out_cub + '.ptx', 'w').write(text)

PTXAS = os.path.join(sys.prefix, 'Lib', 'site-packages', 'nvidia', 'cuda_nvcc', 'bin', 'ptxas.exe')
r = subprocess.run([PTXAS, '-arch=sm_75', '-v', '-o', out_cub, out_cub + '.ptx'], capture_output=True, text=True)
print(r.stderr.strip()[-500:] if r.returncode else '[alu_opt] compiled OK')
sys.exit(1 if r.returncode else 0)
