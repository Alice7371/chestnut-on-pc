"""Offline verification on the user's own footage:
video -> preprocess -> CL/CUDA engine -> plan overlay mp4 + control summary.
usage: video_check.py <video> [start_sec] [duration_sec]
"""
import os
import sys
import time
import numpy as np
import cv2

os.environ.setdefault('TG_DEV', 'CUDA')
_CLI = sys.argv[1:]
PKLV = 'models/driving.pkl'
if _CLI and _CLI[0].endswith('.pkl'):
    PKLV = _CLI[0]; _CLI = _CLI[1:]
sys.argv = [sys.argv[0], PKLV]
print(f'[video_check] model={PKLV}')
_src = open('run_pinned2.py', encoding='utf-8').read()
exec(compile(_src.split('# --- smoke run')[0], 'run_pinned2.py', 'exec'))
from tinygrad import Tensor

VID = _CLI[0] if len(_CLI) > 0 else os.environ.get('VID', '')
START = float(_CLI[1]) if len(_CLI) > 1 else 60.0
DUR = float(_CLI[2]) if len(_CLI) > 2 else 60.0
STEP = int(_CLI[3]) if len(_CLI) > 3 else 1

def rgb_to_channels(rgb):
    r = cv2.resize(rgb, (512, 256), interpolation=cv2.INTER_AREA)
    yuv = cv2.cvtColor(r, cv2.COLOR_RGB2YUV)
    Y, U, V = yuv[:, :, 0], yuv[:, :, 1], yuv[:, :, 2]
    return np.stack([Y[0::2, 0::2], Y[1::2, 0::2], Y[0::2, 1::2], Y[1::2, 1::2],
                     U[0::2, 0::2], V[0::2, 0::2]]).astype(np.uint8)

def parse_plan(out):
    plan = out[out_slices['plan']].reshape(5, 33, 6)
    return plan[int(np.argmin(np.abs(plan[:, -1, 1])))]

cap = cv2.VideoCapture(VID)
cap.set(cv2.CAP_PROP_POS_MSEC, START * 1000)
fps = cap.get(cv2.CAP_PROP_FPS) or 21.0
n_target = int(DUR * fps)
W, H = 960, 544
OUT = 'verified_big.mp4' if 'big' in PKLV else 'verified.mp4'
vw = cv2.VideoWriter(OUT, cv2.VideoWriter_fourcc(*'mp4v'), fps / STEP, (W, H))

pk = np.zeros(524, np.float32)
pk[8:10] = (1.0, 0.0)
rows = []
i = 0
t0 = time.time()
while i < n_target:
    ok, bgr = cap.read()
    if not ok: break
    if i % STEP:
        i += 1
        continue
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    ch = rgb_to_channels(rgb)
    out = run_frame(np.stack([ch, ch]), pk)
    pk[12:] = out[out_slices['hidden_state']]
    tr = parse_plan(out)
    li = 8   # ~2 s
    v_now = float(np.linalg.norm(tr[max(1, li-1), 3:6]))
    v_next = float(np.linalg.norm(tr[li, 3:6]))
    steer = max(-1.0, min(1.0, float(tr[li, 1]) / 2.0))
    rows.append((i, v_now, tr[li, 1], v_next, steer))
    # overlay: ground-plane trajectory
    horizon = int(H * 0.42)
    for k in range(0, 33, 2):
        x = float(tr[k, 0])
        if x < 0.3: continue
        sc = 3.0 / (x + 3.0)
        px = int(W/2 + tr[k, 1] * W * 1.1 * sc)
        py = int(horizon + (H - horizon) * sc)
        cv2.circle(bgr, (px, py), 4, (0, 255, 0), -1)
    pts = [(int(W/2 + tr[k,1]*W*1.1*(3.0/(float(tr[k,0])+3.0))),
            int(horizon + (H-horizon)*(3.0/(float(tr[k,0])+3.0))))
           for k in range(0, 33, 2) if float(tr[k, 0]) >= 0.3]
    if len(pts) >= 2:
        for p, q in zip(pts, pts[1:]):
            cv2.line(bgr, p, q, (0, 255, 0), 2)
    cv2.putText(bgr, f'v@2s={v_next:5.1f}m/s  steer={steer:+.2f}', (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    vw.write(bgr)
    i += 1
vw.release()
cap.release()
rows = np.array(rows)
print(f'processed {len(rows)} frames in {time.time()-t0:.0f}s -> {OUT}')
print(f'planned speed@2s: mean={rows[:,3].mean():.1f} min={rows[:,3].min():.1f} max={rows[:,3].max():.1f} m/s')
print(f'steer: mean={rows[:,4].mean():+.3f} min={rows[:,4].min():+.3f} max={rows[:,4].max():+.3f}')
print(f'decel frames (v@2s < v@prev-step*0.9): {(rows[:,3] < rows[:,1]*0.9).sum()}')
print(f'stop frames (v@2s < 1): {(rows[:,3] < 1).sum()}')
np.save('video_check_rows.npy', rows)
