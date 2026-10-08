# Implementation notes

Distilled engineering notes from building this. See `PROJECT_STATE`-style
detail in the git history; this file keeps what will still matter in a year.

## The WMMA problem, precisely

* The big pickle's attention GEMMs are linearized to 192 `WMMA` uops with spec
  `((16,16,16), half, 'AMD', 32)` — RDNA4 fragmenting: 8 A + 8 B halves and
  8 C/D floats per thread. Layout authority: tinygrad `ops_python.py`'s
  `generic_wmma_helper` (the `len == 8` branch; rdna3 is the 16-element one).
* `mma.sync.m16n16k16.f32.f16.f16.f32` is rejected by ptxas for **sm_75, 80,
  86 and 89** — that shape only exists in the (deprecated, fragment-register)
  `wmma.sync` API, whose PTX forms ptxas also rejects. The only f16 mma on
  sm_75 is `m16n8k8`; `m16n8k16` needs sm_80. Hence: lowering, not emulation.
* The lowering (`_wm_emit` in `run_pinned2.py`) stages each lane's A/B
  fragments in shared memory per 1 KB warp slice, `bar.warp.sync`, then each
  lane consumes with 4 chained `mma.m16n8k8` (k two-step through the C
  accumulator, n in two halves), D reshuffled through shared.
* NV fragment layouts (address = slot + (l>>2)*16 + off(l&3); D row/col per
  lane) were verified on hardware with standalone PTX probes before being
  trusted — see `mma_probe.py` / `consumer_probe.py`.

## Numerical fidelity

`py_cross.py` is the definitive check: it runs the *same* captured runtime
buffers through tinygrad's Python emulator (`PythonProgram`, the executable
form of comma's semantics) and through both CUDA consumers:

```
py-spec  vs  cu-FFMA : bit-exact on all 4 buffers
py-spec  vs  cu-mma  : == (cu-FFMA vs cu-mma), i.e. pure reassociation noise
```

The remaining mma-vs-FFMA differences concentrate in near-zero outputs where
fp32 accumulation order flips the sign of a cancellation — quantified at 2.8%
of tiles on one kernel, 1/16384 values (1.5e-5 absolute) on another. Downstream
plans are unaffected (verified frame-identical on road footage).

## CUDA traps worth remembering

* `cuLaunchKernel` via tinygrad's autogen bindings: tinygrad itself passes the
  packed param buffer in the **`extra` slot** (11th) as a
  `CU_LAUNCH_PARAM_BUFFER_POINTER` 5-tuple (`encode_args`), with
  `kernelParams=None`. A hand-rolled kernelParams pointer-array works in a
  fresh context but reports INVALID_VALUE through the same bindings — reuse
  `encode_args` for standalone launches.
* WDDM launch tax is real: ~0.5 ms GPU-side per launch with a deep queue
  (GPU busy <1 ms, util 6%). CUDA graphs fix it; within a graph a launch is
  ~1.6 µs. Capture requires a stream-aware launcher that refreshes
  `kernelParams` per launch (same kernel launches many times with different
  buffers), and async H2D staging buffers must outlive the capture.
* tinygrad pools device buffers: several tensors are interior pointers of one
  `cudaMalloc`. `cuMemGetAddressRange` returns the *whole* allocation, so
  span-based pre/post dumps of neighboring tensors overlap each other —
  parameter comparisons must use the signature's visible element counts,
  never allocation spans.
* Timing inside a deep queue: host wall-clock and event-bracketed launches
  both lie. Use e0/launch/e1/synchronize inline per launch.
* `ptxas_cache` is content-hashed (sha1 of the PTX), but stale variants of the
  same kernel live forever. Before reusing a cached cubin, grep its `.ptx` for
  the expected signature (e.g. the pre-fix FFMA render still has the mixed
  `tid.x`-into-base addressing bug and faults MISALIGNED standalone).

## The serial-reduce case study (`alu_probe/alu_opt/bench_alu`)

`r_32_32_3_12288_32` looks innocuous but is a 12288-trip `RANGE` loop
(plain `bra`, easy to miss if you grep for `bra.uni`) executed by ONE warp —
32 threads on the whole GPU — at ~8 ms/launch. The fix (`_warp_split_ptx`)
splits the loop across W warps of one block (`local_size` y: 1→W), writes
per-warp partials to shared, and lets warp 0 finish the reduction. Details
that mattered:

* The 96 fp32 accumulators are ALL live (they enter phase 2 through an f16
  quantize + multiply), so the shared buffer is W×32×96×4 B — W=4 hits the
  48 KB static limit exactly, W=8 doesn't fit (chunked scheme would).
* The transform must splice AFTER the loop-back branch, not at the loop's
  END label (the label sits *inside* the do-while decrement structure).
* PTX forbids special registers (`%tid.x`) as operands of anything but
  `mov`; copy first.
* Structural finding left on the table: every lane computes all 96 output
  slots but stores only its own 3 — 31/32 of phase-1 arithmetic is redundant.
  Exploiting that means changing the data flow (uop level), not the PTX.

## The WMMA attention kernels: k-split, one win and three rejections (`_ksplit_ptx`)

The big WMMA kernels (`r_6_2` 29×1.2 ms, `r_2_24` 30×1.0 ms, `r_12_32_4`
2×1.7 ms, `r_48_32_4` 3×1.0 ms per frame) are ALL serial do-while loops —
K-reductions, not output tiles: the loop's stores happen only AFTER it
(loop-carried f32 mma fragments), so splitting requires a partial-sum
exchange. `_ksplit_ptx` implements it (tid.z pair scheme, chunked shared
exchange, `_wm` staging slices remapped to the linear warp id). Per-kernel,
A/B + frame-validated:

| kernel | regs | result |
| --- | --- | --- |
| `r_12_32_4` (138 regs) | 1.52→1.32 ms, output **bit-identical**, frame rows bit-identical | **shipped, default on** |
| `r_48_32_4` (160 regs) | +3% slower | rejected |
| `r_6_2` (255 regs) | flat | rejected |
| `r_2_24` (255 regs) | 16% slower | rejected |

The pattern is occupancy: every extra warp must fit the same register file.
At 255 regs/thread the block count per SM is already 1 and more warps only
add reduction traffic; at 138 regs there is headroom and the split pays.
The transform stays available (`WMMA_KSPLIT=2` default) but applies to an
explicit whitelist — the structural anchors alone misfire on lookalike
kernels (`r_2048_16_24` has the same skeleton but different staging and
won't compile when remapped).

War stories from this transform, for the next person:

* PTX predicates are their own register class — `@%ksz` with `%ksz.u32`
  fails; you need `setp` + a `.pred`. Inverting which side stores vs adds
  produces outputs that look *almost* right (the C term dominates this
  kernel) — validate against single-launch outputs, not plausibility.
* A bench harness whose baseline launch dims get mangled produces
  ILLEGAL_ADDRESS that looks exactly like a broken transform — check the
  harness before the kernel (cost: three debug rounds).
* When A/B-testing a transformed kernel, make sure the BASELINE cubin is
  the same consumer generation: comparing an mma build against a cached
  FFMA-era cubin shows maxabs ~1.7 (real consumer divergence) and masks
  the split's actual effect (bit-identical). Content-hashed caches hold
  every historical render; grep the PTX, don't trust filenames.
* `compute-sanitizer` may refuse the device ("not supported") on some
  driver/toolkit combos; the fallback is bisect builds.

## Known dead ends (don't retry)

* `mma.sync.m16n16k16` / `wmma.sync` PTX forms on any NVIDIA arch (above).
* Driver in-process JIT for the big PTX: hangs and eats ~15 GB; use offline
  `ptxas` (cached).
* Ceil'd grid dims: the pickle's grids are exact `g*l` products; ceil'ing adds
  threads that OOB-write images and poison the queue.
* Graph-level WHERE masking for image zero-border: whole-graph shape
  propagation defeats it three ways; do it at PTX emission (address clamp +
  validity predicate) instead.
* Re-linearizing from SINK: loses thread distribution, produces 40 MB /
  850 K-register kernels.
* **Halving the attention K-loops** (`WMMA_LOOPHALF`, in-tree but default
  off): single-kernel A/B shows odd iterations contribute only ~0.3% median
  — but after 90 downstream kernels the frame plans move by up to 2.0
  (planned speed 1.2 vs 0.5 m/s) — a qualitative behavior change. Rejected
  at the frame gate. Lesson: element-level "negligible" is not frame-level
  negligible; always run the frame rows comparison.
* **Grid-split-K partial planes** (`_gsplit_ptx`, default off): transform
  compiles and the two planes differ as expected, but the reduced output
  breaks — the epilogue's z-free additive terms are not all visible as
  simple `add` sites (phase-2 products fold C in multiplicatively). Needs a
  full z-dependence proof over the phase-2 DAG before the partial-sum
  scheme can be correct.
* Confound to remember: the engine feeds `img = big_img` (the same frame
  stacked twice), so attention-side content is duplicated by construction.
  Kernel-level A/Bs on duplicated inputs can show "each half ≈ full" for
  reasons that are properties of the INPUT, not the kernel.
