"""Real-footage demo: comma2k19 road frames -> preprocess -> CL engine ->
parsed driving outputs. This is the integration core the GUI will reuse.

usage: video_demo.py [segment_dir] [max_frames]
  default: $COMMA2K19_IMGS (comma2k19 scb1/imgs), 40 frames
"""
import os
import sys
import glob
import time
import numpy as np
import cv2

TG_DEV = os.environ.get('TG_DEV', 'CL')
os.environ['TG_DEV'] = TG_DEV
PKL = 'models/driving.pkl'
_argv = sys.argv[1:]
if _argv and _argv[0].endswith('.pkl'):
    PKL = _argv[0]; _argv = _argv[1:]
sys.argv = [sys.argv[0], PKL]

# --- engine (comma-LINEAR machinery from run_pinned2, up to the smoke run) ---
_src = open('run_pinned2.py', encoding='utf-8').read()
exec(compile(_src.split('# --- smoke run')[0], 'run_pinned2.py', 'exec'))
# provides: run_frame(), out_slices, ish, PACKED, queues

# --- preprocessing: RGB -> the model's 6-channel YUV layout ------------------
# model input warped = (2, 6, 128, 256) uint8; 6ch = Y00,Y10,Y01,Y11,U,V
# each channel 128x256 (half-res of the 512x256 road view)
def rgb_to_channels(rgb):
    """rgb: (H,W,3) uint8 (any size) -> (6,128,256) uint8"""
    r = cv2.resize(rgb, (512, 256), interpolation=cv2.INTER_AREA)
    yuv = cv2.cvtColor(r, cv2.COLOR_RGB2YUV)
    Y, U, V = yuv[:, :, 0], yuv[:, :, 1], yuv[:, :, 2]
    y00 = Y[0::2, 0::2]
    y10 = Y[1::2, 0::2]
    y01 = Y[0::2, 1::2]
    y11 = Y[1::2, 1::2]
    u = U[0::2, 0::2]
    v = V[0::2, 0::2]
    return np.stack([y00, y10, y01, y11, u, v]).astype(np.uint8)

def make_warped_from_rgb(rgb):
    ch = rgb_to_channels(rgb)
    return np.stack([ch, ch])  # img and big_img fed the same (games: no wide cam)

# --- output parsing -----------------------------------------------------------
# output 2574-dim; plan slice (1576,2566) = 5 hypotheses x 33 steps x 6 values
# (pos xyz + vel xyz per step)
def parse_plan(out):
    s = out_slices['plan']
    plan = out[s].reshape(5, 33, 6)
    lat = np.abs(plan[:, -1, 1])          # farthest-step lateral per hypothesis
    best = int(np.argmin(lat))
    return plan[best]                     # (33, 6): [x,y,z, vx,vy,vz] per step

# --- main ---------------------------------------------------------------------
seg = _argv[0] if len(_argv) > 0 else os.environ.get('COMMA2K19_IMGS', 'E:/datasets/comma2k19-ld/extracted/scb1/imgs')
max_frames = int(_argv[1]) if len(_argv) > 1 else 40
frames = sorted(glob.glob(os.path.join(seg, '*.png')), key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
frames = frames[:max_frames]
print(f'{len(frames)} frames from {seg}')

pk = np.zeros(524, np.float32)
times = []
vis_dump = []
for i, f in enumerate(frames):
    rgb = cv2.imread(f)              # BGR
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    warped = make_warped_from_rgb(rgb)
    t0 = time.perf_counter()
    out = run_frame(warped, pk)
    dt = (time.perf_counter() - t0) * 1e3
    times.append(dt)
    pk[12:] = out[out_slices['hidden_state']]   # recurrent feedback

    tr = parse_plan(out)
    hs = out[out_slices['hidden_state']]
    print(f'[{i:3}] {dt:6.1f}ms  pos@2s=({tr[8][0]:7.2f},{tr[8][1]:7.2f},{tr[8][2]:7.2f})'
          f'  x@30s={tr[-2][0]:8.2f}  y@30s={tr[-2][1]:7.2f}  |hidden|={np.abs(hs).mean():.4f}')
    if i % 10 == 0:
        vis = rgb.copy()
        h, w = vis.shape[:2]
        # project trajectory: x forward, y lateral (model frame: meters)
        for k in range(0, len(tr), 2):
            px = int(w/2 + tr[k][1] * w / 20)
            py = int(h * 0.85 - tr[k][0] * h / 150)
            cv2.circle(vis, (px, py), 4, (0, 255, 0), -1)
        vis_dump.append((f, vis))

print(f'\navg {np.mean(times[3:]):.1f}ms/frame ({1000/np.mean(times[3:]):.0f}Hz), '
      f'finite={np.isfinite(out).all()}')
for f, vis in vis_dump:
    cv2.imwrite('demo_' + os.path.basename(f), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
print('viz written: demo_*.png')
