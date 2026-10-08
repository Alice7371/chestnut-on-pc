# chestnut-on-pc

Run comma.ai's **chestnut-class driving models** — including the 1B-param "big"
model — on a desktop **NVIDIA** GPU, no comma hardware or AMD eGPU required.
Ships a realtime viewer GUI: screen capture → model inference → live plan HUD.

The chestnut branch ships its models as tinygrad pickles whose attention GEMMs
are expressed as **AMD RDNA4 WMMA uops** (`((16,16,16), half, 'AMD', 32 threads)`).
This project decodes the pickle, recompiles every kernel for CUDA **offline**
(ptxas), and lowers those WMMA uops onto the only f16 tensor-core shape sm_75
accepts (`mma.sync.m16n8k8`) — plus a replay/CUDA-graph frame loop and a
warp-split rewrite of a pathological serial-reduce kernel.

## Results (RTX 2080, sm_75, Windows)

| model | stock tinygrad path | this repo |
| --- | --- | --- |
| driving (39M, OpenCL source pkl) | ~53 ms (CL) | **8.6 ms (~116 Hz)** CUDA |
| big_driving (1B, WMMA pkl) | does not run (no AMD HW) | 305 ms → **~84 ms (~12 Hz)** |

Speedup layers for the big model:
1. **Plan replay** — record the resolved 497-launch sequence once, replay it
   (stock CapturedJit re-walks the 48K-uop graph in Python every frame and does
   6 synchronous D2H round trips per frame).
2. **CUDA graph** — one `cuGraphLaunch` per frame; on WDDM each launch costs
   ~0.5 ms GPU-side, which dominated the frame (~250 ms of pure launch latency).
3. **Tensor-core WMMA consumer** — `mma.m16n8k8` ×4 per WMMA with k-chaining
   through the C accumulator (sm_75's only f16 mma shape; `m16n16k16` is illegal
   on every NVIDIA arch, and `wmma.sync` fragment forms are rejected by ptxas).
4. **Warp-split serial reduce** — `r_32_32_3_12288_32` ran a 12288-iteration
   loop on a *single warp* (6.8 ms); split across 4 warps with a shared-memory
   partial-sum reduce → 2.1 ms (`ALU_SPLIT=4`, default on).
5. **K-split for `r_12_32_4`** — the attention K-reduction loops resist naive
   splitting (see NOTES.md for the full case study); `r_12_32_4` has register
   headroom and takes 1.2× bit-identically (`WMMA_KSPLIT=2`, default on).

## Where the frame time goes (per-kernel case studies)

Profile of one big-model frame (~84 ms on a 2080), and what happened to
each hotspot:

| kernel | shape found | per-frame | action | result |
| --- | --- | --- | --- | --- |
| `r_6_2` (attention GEMM) | 384-trip serial K-loop, 12 blocks = 1 warp/SM, 255 regs | ~25 ms | k-split attempted | **flat** — register-file wall |
| `r_2_24` (attention) | same shape, 255 regs | ~18 ms | k-split attempted | **+16%** — rejected |
| `r_12_32_4` | same shape, 138 regs | ~3.3 ms | k-split, partial-sum exchange | **1.2×, bit-identical** |
| `r_32_32_3` | 12288-trip loop on ONE warp | 6.8 ms | warp-split, shared reduce | **3.15×** |
| `r_48_32_4` | K-loop, 160 regs | ~2.5 ms | k-split | +3% — rejected |

The lesson: a serial loop is only *cheaply* splittable if the register file
has room for the extra warps. The 255-register kernels carry their whole
output tile as live f32 mma fragments across the loop; moving that state
(shared / graph-level split-K with a scratch buffer + reduction pass) is the
remaining lever, and the next project.

## Architecture

```
pickle (comma tinygrad graph, 93 kernels)
  │  decode: fake QCOM/AMD device stubs → linear uop lists
  ▼
reprepare: per-kernel PTX rendering (patched tinygrad PTXRenderer)
  │  ├─ WMMA emission-time lowering: RDNA4 fragments → shared stage →
  │  │   mma.m16n8k8 ×4 (or FFMA reference via WMMA_FFMA=1)
  │  ├─ image2d zero-border: address clamp + validity predicate (PTX level)
  │  ├─ ALU_SPLIT: warp-split for single-warp serial reduce kernels
  │  └─ WMMA_KSPLIT: tid.z K-split + chunked shared partial exchange
  ▼
offline ptxas → cubin  (content-hashed cache; driver in-process JIT hangs)
  ▼
first frame: record the resolved plan (497 launches + copies)
  ▼
CUDA graph capture → one cuGraphLaunch per frame  (~1.6 µs/launch)
  ▼
run_frame(warped, packed) → plan/lead/lane_lines/hidden_state
```

## Measurement methodology

* Per-kernel times: `KSYNC=1` inserts an event pair around every launch
  (e0/launch/e1/synchronize inline — host wall-clock and event-bracketed
  timings both lie in a deep queue). KSYNC totals include sync overhead;
  kernel-level deltas are what matter.
* Single-kernel A/B: probes capture the real runtime buffers at a kernel's
  first launch (`*_probe.py`), then a fresh-context bench (`bench_*.py`)
  launches baseline and variant cubins on identical inputs.
* Frame-level: `video_check.py` writes per-frame plan rows; every shipped
  change must keep rows bit-identical (exact transforms) or within fp32
  reassociation noise (k-split transforms).
* Consumer generation matters: content-hashed caches keep every historical
  render. An A/B against a stale FFMA-era cubin shows ~1.7 max-abs fake
  divergence (real mma-vs-FFMA consumer difference) — verify the baseline's
  PTX before trusting a diff.

## Quick start

```powershell
# Python 3.11+; tinygrad must be comma's pinned tree from the same branch
# (pip tinygrad 0.14 has breaking IR changes — do not use)
git clone --branch release-chestnut --depth 1 https://github.com/commaai/openpilot
pip install numpy opencv-python dxcam pywin32
pip install -e openpilot/tinygrad_repo   # or set PYTHONPATH to it

# fetch the model pickles from the same branch (big = 1.8 GB)
#   openpilot/selfdrive/modeld/models/driving.pkl       (39M)
#   openpilot/selfdrive/modeld/models/big_driving.pkl   (1B)
# copy them into ./models/  (see models/README.md)

# offline validation on dashcam footage
python video_check.py models/big_driving.pkl my_drive.mp4 0 8

# realtime GUI (screen region L T W H, defaults to full screen)
python drive_gui.py
```

In the GUI: **START** begins capture; the HUD overlays the live plan
trajectory (top-down view, speed profile, planned steer/accel/brake readout).
Display only — the GUI writes to no output device.

## Correctness evidence

* **Cross-validation against the executable spec** (`py_cross.py`): the same
  real runtime buffers are run through tinygrad's `PythonProgram` (tinygrad's
  own WMMA semantics, incl. `generic_wmma_helper`) and through both CUDA
  consumers. The Python reference and the FFMA consumer agree **bit-exact on
  every argument**; the tensor-core consumer differs only by fp32
  reassociation noise (identical nonzero counts, ~1e-5 absolute on
  cancellation-prone outputs).
* **Hardware fragment probes** (`mma_probe.py`, `consumer_probe.py`,
  `ffma_probe.py`): the m16n8k8 fragment layouts were verified against the
  silicon, not just on paper.
* **Road footage**: comma2k19 and personal dashcam clips produce rising,
  centered, input-sensitive plans (pos@2s grows 12→29 m while driving;
  bright/dark A/B flips the plan).

## Diagnostics (env vars)

| var | effect |
| --- | --- |
| `TG_DEV=CUDA/CL` | engine backend (CUDA default) |
| `REPLAY=0` | disable plan replay (first frame path every frame) |
| `ALU_SPLIT=n` | warp count for the serial-reduce kernel (default 4, 0 = stock) |
| `WMMA_FFMA=1` | render WMMA as FFMA reference instead of tensor cores |
| `WMMA_SKIP=1` | WMMA passthrough D:=C isolation probe |
| `KSYNC=1` | per-launch CUDA-event timings |
| `KDUMP=1` | dump per-launch buffer pre-states |
| `CLDEBUG=1` / `CUDASYNC=1` | per-kernel fences to name a faulting kernel |

See `NOTES.md` for the full set of implementation notes, gotchas and the
"dead ends" list.

## Layout

```
run_pinned2.py      the engine: pkl decode, recompile, WMMA lowering, replay/graph
drive_gui.py        realtime capture -> model -> plan HUD (display only)
video_check.py      offline validation on video files
video_demo.py       offline validation on comma2k19 frame folders
modeld.py           reference interpreter derived from openpilot's modeld
parse_model_outputs.py, constants.py, helpers.py   plan/output decoding
py_cross.py         Python-emulator cross-validation harness
alu_probe.py, alu_opt.py, bench_alu.py   warp-split case study (capture/A-B/bench)
*_probe.py, dual_cubin.py, bench_one.py  hardware fragment probes
research/           cross-check harness + pickle metadata tools
models/             put the comma pickles here (not included)
```

## Limitations & future work

* The two biggest kernels (`r_6_2`, `r_2_24`, ~45 ms/frame combined) sit at
  the 255-register occupancy wall: their whole output tile lives as f32 mma
  fragments across the serial K-loop. In-block splitting cannot add resident
  blocks. The next lever is graph-level split-K — spread K over new grid
  blocks, write partials to a scratch buffer, and reduce with a synthesized
  second kernel (estimated 3-4× on these kernels, i.e. frame → ~45 ms).
* Windows/WDDM only (CUDA graph behavior is the frame-time foundation).
* sm_75 assumed for the mma shape; sm_80+ could use `m16n8k16` directly.
* The engine targets comma's pinned tinygrad tree; pip tinygrad 0.14 breaks
  the IR.

## Credits & license

Derived from and standing on:
* [comma.ai openpilot](https://github.com/commaai/openpilot) (MIT) — the model
  pickles, `modeld.py`/`compile_modeld.py` and the output decoding are ports or
  derivatives of openpilot code; the model weights remain comma's.
* [tinygrad](https://github.com/tinygrad/tinygrad) (MIT) — both runtime and the
  Python emulator used as the WMMA reference.

This code is MIT (see `LICENSE`). Research/education use; the plans are model
output, not driving advice.
