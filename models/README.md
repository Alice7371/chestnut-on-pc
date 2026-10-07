# Models (not included)

Fetch the pickles from comma.ai's `release-chestnut` openpilot branch:

| file | size | notes |
| --- | --- | --- |
| `driving.pkl` | ~84 MB | 39M-param model, kernels carry OpenCL source |
| `big_driving.pkl` | ~1.8 GB | 1B-param model, kernels carry RDNA4 WMMA uops |

Both live at `selfdrive/modeld/models/` on that branch. Easiest:

```bash
git clone --branch release-chestnut --depth 1 https://github.com/commaai/openpilot
cp openpilot/selfdrive/modeld/models/*.pkl models/
```

(If raw GitHub access is slow from your network, the files are plain git
blobs served by `raw.githubusercontent.com`; a proxy such as
`socks5h://localhost:7897` works with `git -c http.proxy=...`.)

Name them `models/driving.pkl` and `models/big_driving.pkl`, or pass the path
explicitly (`video_check.py <pkl> <video> ...` / `PKL=... python py_cross.py`).
