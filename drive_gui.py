"""chestnut viewer GUI — screen capture -> engine -> plan HUD.

Pipeline (~20Hz):
  dxcam region grab -> RGB -> YUV 6ch -> run_frame (hidden_state feedback)
  -> plan trajectory -> live HUD (video preview, top-down plan, speed
  profile, planned steer/accel/brake bars). Display only: nothing is
  sent to any output device.

usage: drive_gui.py [capture L T W H]   default: full primary screen
"""
import os
import sys
import time
import threading
import numpy as np
import cv2

os.environ.setdefault('TG_DEV', 'CUDA')   # CL = numerics reference (slower)

import tkinter as tk
from tkinter import ttk
import dxcam

# --- engine -------------------------------------------------------------------
_CLI = sys.argv[1:]                      # [L T W H] for capture region
sys.argv = [sys.argv[0], 'models/driving.pkl']   # run_pinned2 reads sys.argv[1] as PKL
_src = open('run_pinned2.py', encoding='utf-8').read()
exec(compile(_src.split('# --- smoke run')[0], 'run_pinned2.py', 'exec'))
from tinygrad import Tensor

# --- preprocessing (same as video_demo) ---------------------------------------
def rgb_to_channels(rgb):
    r = cv2.resize(rgb, (512, 256), interpolation=cv2.INTER_AREA)
    yuv = cv2.cvtColor(r, cv2.COLOR_RGB2YUV)
    Y, U, V = yuv[:, :, 0], yuv[:, :, 1], yuv[:, :, 2]
    return np.stack([Y[0::2, 0::2], Y[1::2, 0::2], Y[0::2, 1::2], Y[1::2, 1::2],
                     U[0::2, 0::2], V[0::2, 0::2]]).astype(np.uint8)

def parse_plan(out):
    plan = out[out_slices['plan']].reshape(5, 33, 6)
    best = int(np.argmin(np.abs(plan[:, -1, 1])))
    return plan[best]     # (33, 6): x,y,z, vx,vy,vz per step

# --- planned-control readout (display only) ------------------------------------
T_IDXS = [10.0 * (i / 32) ** 2 for i in range(33)]   # constants.py index_function

def control_from_plan(tr):
    """tr (33,6) -> (steer -1..1, accel 0..1, brake 0..1) as a plan readout."""
    li = min(range(33), key=lambda k: abs(T_IDXS[k] - 2.0))   # lookahead ~2 s
    lat = float(tr[li, 1])
    steer = max(-1.0, min(1.0, lat / 2.0))
    v_now = float(np.linalg.norm(tr[max(1, li - 1), 3:6]))
    v_next = float(np.linalg.norm(tr[li, 3:6]))
    decel = max(0.0, (v_now - v_next) - 0.2)
    accel = 0.35 if v_next > 2.0 else 0.0
    brake = max(0.0, min(1.0, decel * 0.8))
    return steer, accel, brake, lat

# --- capture + inference worker -------------------------------------------------
class Worker(threading.Thread):
    def __init__(self, app):
        super().__init__(daemon=True)
        self.app = app
        self.region = None      # None = full screen
        self.running = True
        self.cam = dxcam.create(output_idx=0, output_color='RGB')
        self.lat = []

    def run(self):
        pk = np.zeros(524, np.float32)
        pk[8:10] = (1.0, 0.0)   # traffic_convention: left-hand-drive flag, comma normal
        t_last = 0.0
        while self.running:
            if not self.app.capture_on:
                time.sleep(0.05)
                continue
            frame = self.cam.grab(region=self.region)
            if frame is None:
                time.sleep(0.005)
                continue
            now = time.perf_counter()
            fps = 1.0 / (now - t_last) if t_last else 0.0
            t_last = now
            ch = rgb_to_channels(frame)
            warped_np = np.stack([ch, ch])
            t0 = time.perf_counter()
            out = run_frame(warped_np, pk)
            inf_ms = (time.perf_counter() - t0) * 1e3
            pk[12:] = out[out_slices['hidden_state']]

            tr = parse_plan(out)
            steer, accel, brake, lat = control_from_plan(tr)
            act = out[out_slices['action']] if 'action' in out_slices else None
            lead_txt = ''
            try:
                lp = out[out_slices['lead_prob']]
                if lp[0] > 0.5:
                    ld = out[out_slices['lead']].reshape(2, 6, -1)
                    lead_txt = f'lead dx={ld[0,0,0]:7.1f}m vy={ld[0,0,4]:6.1f}'
            except Exception:
                pass
            self.lat.append(inf_ms)
            if self.app.rec_var.get():
                os.makedirs('rec', exist_ok=True)
                cv2.imwrite(f'rec/{int(time.time()*1000)}.png', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                with open('rec/plan.log', 'a') as f:
                    f.write(f'{time.time():.3f} ' + ' '.join(f'{v:.4g}' for v in tr.ravel()) + '\n')
            small = cv2.resize(frame, (512, 288))
            self.app.push_hud(small, tr, steer, accel, brake, fps, inf_ms,
                              len(self.lat) and sum(self.lat[-30:]) / min(len(self.lat), 30),
                              act, lead_txt)
            # pace to ~20 Hz
            spend = time.perf_counter() - now
            time.sleep(max(0.0, 0.05 - spend))

# --- GUI ------------------------------------------------------------------------
class App:
    def __init__(self, root, region):
        self.root = root
        root.title('chestnut viewer')
        root.attributes('-topmost', True)
        self.capture_on = False

        top = ttk.Frame(root, padding=4); top.pack(fill='x')
        self.start_btn = ttk.Button(top, text='START capture', command=self.toggle)
        self.start_btn.pack(side='left')
        self.rec_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(top, text='REC (dump frames)', variable=self.rec_var).pack(side='left', padx=6)
        self.status = ttk.Label(top, text='idle'); self.status.pack(side='left', padx=8)

        mid = ttk.Frame(root); mid.pack()
        self.canvas = tk.Canvas(mid, width=512, height=280, bg='#101418')
        self.canvas.pack(side='left')
        self.img_id = self.canvas.create_image(0, 0, anchor='nw')
        self.traj_id = None
        self.side = tk.Canvas(mid, width=380, height=280, bg='#0c1014', highlightthickness=0)
        self.side.pack(side='left')
        self.info = ttk.Label(root, text='', font=('Consolas', 9))
        self.info.pack(fill='x')

        self.region = region
        self.lock = threading.Lock()
        self.hud_data = None
        self.worker = Worker(self)
        self.worker.region = region
        self.worker.start()
        root.after(50, self._tick)
        root.protocol('WM_DELETE_WINDOW', self.close)

    def push_hud(self, *args):
        with self.lock:
            self.hud_data = args

    def _tick(self):
        with self.lock:
            data = self.hud_data
            self.hud_data = None
        if data is not None:
            self.update_hud(*data)
        self.root.after(50, self._tick)

    def toggle(self):
        self.capture_on = not self.capture_on
        self.start_btn.config(text='STOP' if self.capture_on else 'START capture')

    def update_hud(self, frame, tr, steer, accel, brake, fps, inf_ms, avg_ms, act=None, lead_txt=''):
        img = cv2.resize(frame, (512, 288))[0:280] if frame.shape[1] * 280 // frame.shape[0] >= 512 \
            else cv2.resize(frame, (frame.shape[1] * 280 // frame.shape[0], 280))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        _, buf = cv2.imencode('.ppm', img)
        self.tkimg = tk.PhotoImage(data=buf.tobytes(), format='ppm')
        self.canvas.itemconfig(self.img_id, image=self.tkimg)
        if self.traj_id: self.canvas.delete(self.traj_id)
        pts = []
        h, w = img.shape[:2]
        horizon = int(h * 0.42)
        x0 = 3.0      # ground-plane projection: py = horizon + (h-horizon)*x0/(x+x0)
        for k in range(0, 33, 2):
            x = float(tr[k, 0])
            if x < 0.3: continue
            scale = x0 / (x + x0)
            px = int(w / 2 + tr[k, 1] * w * 1.1 * scale)
            py = int(horizon + (h - horizon) * scale)
            pts += [px, py]
        if len(pts) >= 4:
            self.traj_id = self.canvas.create_line(*pts, fill='#00ff66', width=2)
        self.info.config(text=f'fps={fps:4.1f}  inf={inf_ms:5.1f}ms avg={avg_ms:5.1f}ms  '
                              f'plan steer={steer:+.2f}  accel={accel:.2f}  brake={brake:.2f}')
        self._draw_side(tr, steer, accel, brake, act, lead_txt)

    def _draw_side(self, tr, steer, accel, brake, act, lead_txt):
        c = self.side
        c.delete('all')
        W, H = 380, 280
        # --- top-down plan trajectory (forward up, lateral right) ---
        c.create_text(8, 6, anchor='nw', text='PLAN (top view)', fill='#9fb3c8', font=('Consolas', 9))
        ox, oy = W // 2, 128          # origin: car position
        sx, sy = 4.0, 3.0             # px per meter (x forward up, y lateral right)
        c.create_line(ox, 18, ox, oy, fill='#233043', width=1)   # forward axis
        pts = []
        for k in range(0, 33, 2):
            px = ox + tr[k, 1] * sx
            py = oy - tr[k, 0] * sy
            pts += [px, py]
            c.create_oval(px - 2, py - 2, px + 2, py + 2, fill='#00ff66', outline='')
        c.create_line(*pts, fill='#00ff66', width=1)
        c.create_rectangle(ox - 5, oy - 9, ox + 5, oy + 9, outline='#e6e6e6')
        c.create_text(ox + 8, 22, anchor='nw', text='10s', fill='#4a5a6a', font=('Consolas', 8))
        # --- planned speed profile (y 140..178) ---
        c.create_text(8, 140, anchor='nw', text='PLAN SPEED 0-10s (m/s)', fill='#9fb3c8', font=('Consolas', 9))
        vmax = 40.0
        for k in range(0, 33, 2):
            v = float(np.linalg.norm(tr[k, 3:6]))
            bx = 8 + k * (W - 24) // 32
            bh = int(min(v, vmax) / vmax * 22)
            c.create_rectangle(bx, 172 - bh, bx + 6, 172, fill='#39c2ff', outline='')
        # --- planned-control readout (y 186..246) ---
        def bar(y, label, val, lo, hi, color):
            c.create_text(8, y, anchor='nw', text=label, fill='#9fb3c8', font=('Consolas', 9))
            c.create_text(108, y, anchor='nw', text=f'{val:+.2f}' if lo < 0 else f'{val:.2f}',
                          fill='#e6e6e6', font=('Consolas', 9))
            frac = (val - lo) / (hi - lo)
            c.create_rectangle(160, y + 2, 344, y + 12, outline='#233043')
            bx = 160 + int(max(0.0, min(1.0, frac)) * 184)
            c.create_rectangle(160, y + 2, bx, y + 12, fill=color, outline='')
        bar(186, 'steer', steer, -1, 1, '#ffb347')
        bar(210, 'accel', accel, 0, 1, '#6ee06e')
        bar(234, 'brake', brake, 0, 1, '#ff6b6b')
        # --- status line (y 258) ---
        txt = "plan readout (no output)"
        if act is not None:
            txt += f"  action: curv={act[0]:+.4f} acc={act[1]:+.3f}"
        c.create_text(8, 258, anchor='nw', text=txt, fill='#e6e6e6', font=('Consolas', 9))

    def close(self):
        self.capture_on = False
        self.worker.running = False
        self.root.destroy()

def main():
    region = None
    if len(_CLI) == 4:
        l, t, w, h = map(int, _CLI)
        region = (l, t, l + w, t + h)
    root = tk.Tk()
    App(root, region)
    root.mainloop()

if __name__ == '__main__':
    main()
