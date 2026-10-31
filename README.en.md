# SeeThrough-GGUF-Win

**English** · [简体中文](README.md)

A **local Windows web UI** wrapped around [IceFog72/see-through-gguf](https://github.com/IceFog72/see-through-gguf) (aka **STM**: a pure Python + PyTorch runtime that turns a single character image into layered PSD / depth / ragdoll-split parts). Upstream code is vendored as a snapshot under `stm/`, so **one clone is enough — no extra repository to fetch**.

**Screenshots**: [Workbench (layering / stage)](docs/screenshots/workbench-en.png) · [Settings (environment)](docs/screenshots/settings-en.png)

---

## 1. Quick start

Clone the repository first (or just download the ZIP and unpack it):

```bat
git clone https://github.com/Mingri196/SeeThrough-GGUF-Win.git
cd SeeThrough-GGUF-Win
```

Then install and launch:

```bat
1-setup.bat      :: create .venv, install deps, verify patches, probe network, prefetch helper components, self-check
2-launch.bat     :: → http://localhost:7860
```

You only need **Python 3.12 (or uv)** — Git is not required. `1-setup.bat` will:

1. verify the vendored `stm/`;
2. create `.venv` and install PyTorch cu128 + dependencies (~8 GB);
3. verify the `use_lama` patch (idempotent, skipped if already applied);
4. probe `huggingface.co` vs `hf-mirror.com` — **mirror is normally picked in mainland China, official elsewhere** — then prefetch the helper components needed by the first run (~2.4 GB, cached in the project's `components/` folder); **main models are not downloaded**;
5. run a self-check and print the result.

Afterwards drop the two GGUF files into `models/` (see section 2); if "Model selection" can see them, you are good to go. To skip the component download, run `set STG_SKIP_MODELS=1` before `1-setup.bat`; to also download the main models, run `.venv\Scripts\python.exe backend\download_models.py`.

Manual equivalent (run from the project root):

```bat
uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe -r requirements-torch.txt
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
.venv\Scripts\python.exe backend\patches\apply_use_lama.py
.venv\Scripts\python.exe backend\netcheck.py
.venv\Scripts\python.exe backend\download_models.py --components-only
.venv\Scripts\python.exe backend\selfcheck.py
.venv\Scripts\python.exe backend\server.py
```

Without `uv`: `py -3.12 -m venv .venv` and then `.venv\Scripts\python -m pip install -r requirements.txt`.

**Requirements**: Windows 10/11 64-bit · NVIDIA driver with CUDA 12.8 · Python **3.12** · ~26 GB of disk (env ~8 GB + components ~2.4 GB + your own main models ~5.4 GB + headroom) · network during installation. Once the components are prefetched, inference needs no network.

The helper components needed by the first run (Marigold text encoder, anime segmentation, LaMa, RTMPose, anime-seg) are downloaded into the project's **`components/`** folder instead of scattering into `C://Users//<you>//.cache`: deleting the project folder uninstalls everything, and moving to another drive is a single folder drag. To use the system default location instead (one shared cache for several tools), set `components_dir` to `"user"` in `config.json`.

> **GPU architecture note**: GTX 10-series is Pascal (sm_61). torch 2.9+ cu128 wheels are compiled only up to sm_75 and above — they install fine and `torch.cuda.is_available()` returns true, but the first real kernel aborts with `no kernel image is available for execution on the device`. That is why `requirements-torch.txt` pins **torch 2.8.0+cu128** (compiled for sm_61/70/75/80/86/90/100/120) — the last release that still supports GTX 10-series. 16/20/30/40-series cards can upgrade on their own.

### 1.1 Hand it to an AI

If you would rather not read commands: give **[docs/ai-setup.md](docs/ai-setup.md)** to an AI (drag the file in, or say "deploy this project following `docs/ai-setup.md`"). It will walk you through:

- **Health check**: Python / uv, disk (~26 GB), GPU and driver — and tell you what to install if something is missing;
- **Clone**: `git clone` (ZIP if Git is missing) and verify `stm/`, `backend/`, `frontend/` are complete;
- **Environment**: run the 5 steps of `1-setup.bat` with explanations, plus the equivalent commands for non-interactive use;
- **Models**: `download_models.py` or drop the GGUF files into `models/` yourself;
- **Launch and verify**: start the service → hit 4 endpoints → smoke-test one image at 512×512 / 6 steps;
- **How to use it**: the four-step workflow, how to pick parameters, what lives in `outputs/jobs/<id>/`;
- **Structure**: what every directory / script does, what you may change and what you should not;
- **Report**: a fixed report template, including a clear statement of where it got stuck and what it needs from you.

That document also lists the AI's red lines (never touch the system Python, never edit `stm/` sources, never install drivers / VS for you, never commit to git) and a troubleshooting table. Good for "I just want to use it, not tinker".

> Note: `docs/ai-setup.md` and `docs/triton-windows-ai.md` are currently written in Chinese only (this README and the web UI are the bilingual parts).

---

## 2. Model download

**You provide the main models yourself** — `1-setup.bat` only prefetches helper components. The download script picks a source automatically:

1. probe the model API of both sources in parallel;
2. when both are reachable the official endpoint gets a 600 ms bias (overseas users normally go official);
3. if official is unreachable or clearly slower, `hf-mirror.com` is used (common in mainland China);
4. the choice is printed in the install log, the server startup log, and the "Download source" row on the model details panel.

| Model | Repository | Default path |
| ----------------- | -------------------------------------------- | ------------------------------ |
| LayerDiff3D (layer decomposition) | `icefog72/seethroughv0.0.2_layerdiff3d_gguf` | `models/layerdiff3d-Q5_1.gguf` |
| Marigold3D (depth) | `icefog72/seethroughv0.0.1_marigold_gguf` | `models/marigold3d-Q8_0.gguf` |

Put both GGUF files under `models/` (subdirectories such as `models/layerdiff3d/` and `models/marigold3d/` are fine) — "Model selection" scans recursively and lists them by full relative path. Or let the script do it:

```bat
:: both main models + helper components
.venv\Scripts\python.exe backend\download_models.py --with-components

:: main models only
.venv\Scripts\python.exe backend\download_models.py

:: only check reachability and remote sizes, download nothing
.venv\Scripts\python.exe backend\download_models.py --check-only

:: force a source
.venv\Scripts\python.exe backend\download_models.py --endpoint hf-mirror
.venv\Scripts\python.exe backend\download_models.py --endpoint huggingface
```

`config.json`'s `hf_endpoint` defaults to `"auto"`; you may hard-code `https://hf-mirror.com` or `https://huggingface.co`, which STM also uses when it pulls helper components at runtime. If your file names differ, edit `layerdiff_gguf` / `depth_gguf` — the download script writes to those paths.

> **TLS note**: some security suites (Kaspersky and friends) perform HTTPS interception, which makes Python raise `CERTIFICATE_VERIFY_FAILED`. At install and server start the app merges certifi + Windows root certificates into `outputs/.cache/ca-bundle.pem` for requests / huggingface_hub; if that still fails the download script falls back to `curl` (system certificates, resumable).

---

## 3. UI and workflow

| Step | What it does |
| ----------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **01 Image** | Drag & drop or pick a file — **PNG / JPG / WEBP / BMP**. Extension, size and decodability are validated immediately; a thumbnail, file name, size and pixel dimensions are shown. You can remove or replace the image at any time; replacing clears the previous result and job traces |
| **02 Parameters** | Width/height (synced, multiple of 64), inference steps, seed mode, **output parts** (24 tags grouped by body / head, all selected by default), `tblr_split` / `use_lama` / `layer_fix` / `save_to_psd` switches, and advanced items (quantized streaming, depth steps, depth resolution, foreground mask resolution, attention backend, ragdoll split, pose confidence, joint overlap, keep cut images & manifest) |
| **03 Model selection** | Recursively scans `models/` (including subdirectories) and lets you pick the "layer model" and the "depth model" separately; the dropdown shows relative paths (e.g. `marigold3d/marigold3d-Q5_1.gguf`), so quantisation variants of the same name never get confused; switching variants needs no `config.json` edit. "Rescan" refreshes the list. "Keep model loaded" (on by default) keeps models in VRAM and saves ~40–75 s of loading per run; "Group offload" (off by default) shuttles weights between CPU and GPU per group — more VRAM headroom, slower |
| **04 Run** | Start / cancel; below it the resolution, steps, actual seed and elapsed time of this run are always visible |
| **Stage** | Top right "Re-post-process" (popover: left/right split, ragdoll split, confidence, joint overlap, PSD, cut images — post-processing only, no re-sampling) + real output thumbnails (click for full size) + PSD row ("Open folder" opens that job's `out/`) + manifest row (`cuts/manifest.json`, `pose_keypoints.json` and other JSON are downloadable) + a **pose confidence** strip (after ragdoll splitting, joints below the threshold are listed automatically); the persistent log console sits below |
| **Settings** | The top bar "Settings" opens `/settings` in a new tab. A fixed sidebar with three pages (fully bilingual, **follows the workbench CN/EN switch live**): **Environment** (Python / torch / CUDA / GPU & VRAM / driver / OS / CPU / RAM / download source with handshake latency / paths / component cache path and size / model list / patches & switches / disk, plus "Copy diagnostics" in the top right), **Startup & port** (listen address and local-only/LAN verdict, configured port, **change address / change port** written back to config.json — auto-backed up as `config.json.bak`, effective after restart — PID and uptime, config.json path and last-modified time), **Maintenance** (service status, integrity checks for .venv / STM code / both main models / three patches, job count and disk usage, open outputs / models / project root, plus "Open command prompt" in the top right, which launches `3-env.bat` in a cmd window with `.venv` activated — local machine only). Everything is read-only apart from the two port/address fields |

**Draggable layout**: the vertical splitter between the parameter rail and the stage can be dragged sideways; the horizontal splitter between the stage and the log can be dragged up and down. Double-click resets, arrow keys work too (hold Shift to move faster). Sizes are stored in `localStorage`.

- The CN/EN switch applies instantly and is remembered; all statuses, errors, field labels and log levels are bilingual.
- Parameter errors are shown inline under the field, and the backend validates defensively and returns field-level error codes.
- Invalid numeric input disables the button outright — no request is sent.

---

## 4. Parameters and API contract

`POST /api/run` (`multipart/form-data`: `image` + `params` JSON)

| Field | Type | Range / default | Notes |
| ----------------------- | --------------------- | ----------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `width` / `height` | int | 256–2048, multiple of 64, default 1024 | The engine infers on a square, so both must be equal; the backend returns `must_equal_width` otherwise |
| `inference_steps` | int | 1–150, default 30 | LayerDiff sampling steps |
| `inference_steps_depth` | int | -1 or 1–100, default **4** | Marigold depth steps; -1 = model built-in default (also 4) |
| `resolution_depth` | `"auto"` / -1 / int | default **`"auto"`** | Depth inference size: `auto` takes the smaller of the VRAM ceiling (768 on 6 GB) and the main resolution; `-1` follows the main resolution; a positive integer (multiple of 64) forces a size |
| `mask_resolution` | `"auto"` / int | default **`"auto"`** | Foreground segmentation resolution, **only used when the image has no alpha channel** (segmentation is skipped otherwise). `auto` clamps the long edge between 640 and the VRAM ceiling; a positive integer (256–4096) forces it — larger means finer edges and more VRAM |
| `attention_backend` | `native` / `xformers` | default **`native`** | Attention implementation; two buttons in the advanced panel. `native` = PyTorch SDPA, zero dependencies; `xformers` must be installed yourself: `uv pip install --python .venv\Scripts\python.exe --no-deps xformers==0.0.32.post1` (pairs with torch 2.8.0; restart before the button becomes selectable) |
| `seed_mode` | str | `fixed` / `random` | `random` is generated server-side and guaranteed to differ from the previous run |
| `seed` | int | 0–2147483647, default 42 | Only meaningful in `fixed` mode |
| `tblr_split` | bool | default `true` | Split parts into left/right pieces |
| `use_lama` | bool | default `true` | Turning it off requires the STM patch (`--no-use_lama`) |
| `save_psd` | bool | default `true` | Write a layered PSD |
| `ragdoll_split` | bool | default `false` | RTMPose ragdoll split |
| `ragdoll_confidence` | float | 0.05–0.95, default 0.3 | Lower it for heavily occluded characters |
| `ragdoll_overlap` | float | 0–0.5, default 0.2 | Overlap ratio at joints (computed on the shorter bone). Larger means less tearing at joints and more duplicated pixels; smaller means smaller layers |
| `parts` | str[] \| null | default `null` (all 24) | Generate only the listed layers; aliases `hair` → front+back hair, `eyes` → irides/eyebrow/eyewhite/eyelash. Only affects what is written out: body parts are always sampled as a whole, while the head group is skipped entirely when nothing from it is selected |
| `save_cut_images` | bool | default **`true`** | Also write `cuts/` (per-layer RGBA crops) and `cuts/manifest.json` (pivots, bones, parents, joint overlap) |
| `group_offload` | bool | default `false` | Shuttle weights between CPU and GPU per group — more VRAM headroom, slower; overlaps with streaming |
| `layer_fix` | bool | default **`true`** | Layer fixes (post-processing patch, no repainting): ① layers hugging the bottom edge of the canvas (typically footwear) are no longer killed by the "empty layer" check; ② eyes are stacked in a fixed eyewhite → irides → eyelash order so the iris is not hidden |
| `quantized` | bool | default **`true`** | GGUF quantisation + streaming; turning it off puts all weights in VRAM |
| `layerdiff_model` | str | relative path under `models/`, optional | Layer model; falls back to `layerdiff_gguf` in `config.json` |
| `keep_model` | bool | default **`true`** | Keep the model in VRAM after a job so the next run skips loading; off means reload every run and release memory immediately |
| `depth_model` | str | relative path under `models/`, optional | Depth model; falls back to `depth_gguf` in `config.json` |

Returns `{ok, job_id, seed_used, params}`. In random mode `seed_used` is written back to the UI immediately.

Other endpoints:

| Method | Path | Notes |
| ------ | ------------------------------- | ---------------------------------------------------------------------------------- |
| GET | `/api/config` | Defaults, constraint ranges, supported extensions, upload limit |
| GET | `/api/check_models` | Default GGUFs, STM repository, LaMa patch and network state |
| GET | `/api/models` | Recursive list of `.gguf` under `models/` (with role inference and defaults) |
| GET | `/api/stream/<job_id>` | SSE: `event: log` (`{ts, level, msg}`) + `event: done` (status, outputs, elapsed) |
| GET | `/api/status/<job_id>` | Polling fallback, includes the last 200 log lines |
| POST | `/api/jobs/<job_id>/cancel` | Kill the process tree, set status to `canceled` |
| GET | `/api/jobs/<job_id>/pose` | Read `pose_keypoints.json`, return the weakest joints and the ragdoll threshold |
| GET | `/api/env` | Read-only environment snapshot for the settings page: runtime, hardware, config, process, config file, integrity checks (does not import torch, uses no VRAM) |
| POST | `/api/terminal` | Open a command prompt with `.venv` activated on the local machine (runs `3-env.bat`); local requests only |
| POST | `/api/config-file` | Write `host` / `port` back into `config.json` (auto-backup `config.json.bak`); only the file changes, listening sockets stay — effective after restart; local requests only |
| POST | `/api/open-folder` | Open a project folder in Explorer; `target` is `root` / `outputs` / `models` / `stm`; local requests only |
| GET | `/settings` | Settings page (environment / startup & port / maintenance, read-only) |
| POST | `/api/jobs/<job_id>/reassemble` | Re-post-process: redo PSD / split / ragdoll for a finished `out/<name>/` without re-sampling |
| GET | `/api/jobs/<job_id>/reassemble` | Query re-post-process status and log tail |
| POST | `/api/jobs/<job_id>/reveal` | Open the job's `out/` folder in Explorer |
| DELETE | `/api/jobs/<job_id>` | Delete the job directory (input image and outputs) |
| GET | `/api/file/<job_id>/<rel>` | Output file; `?download=1` forces an attachment download |

Job metadata lives in `outputs/jobs/<id>/meta.json`, so download links keep working after a server restart. Only one job runs at a time (parallel jobs on a single card just fight over VRAM); a second submission returns `409 busy`.

---

## 5. Two things worth knowing

### 5.1 `--quantized` must be passed explicitly

STM's `common/utils/inference_utils.py` contains:

```python
if not quantized:
    stream = False
```

In other words, without `--quantized` the GGUF is fully dequantised on load and **streaming is silently disabled** — exactly how you blow up 6 GB of VRAM. This project always passes it and exposes a switch in the advanced panel.

### 5.2 The `use_lama` patch

STM's post-processing always inpaints with LaMa (`cluster_inpaint_part(..., inpaint='lama')`) and never exposes a CLI switch. The vendored `stm/` is already patched; `backend/patches/apply_use_lama.py` is kept so the same change can be re-applied to a newer upstream release (idempotent, skips files already patched). The patch adds `--use_lama / --no-use_lama`:

- on by default, matching native behaviour;
- when unpatched, the UI locks the switch to "on", flags it in model details, and the backend refuses "turn LaMa off" (`patch_missing`) instead of pretending it worked;
- the switch only affects the hair branch of `further_extr` (the only inpaint call on this path).

Only two upstream files are touched: `common/utils/inference_utils.py` (new `use_lama` parameter, hair branch picks `lama` / `cv2`) and `inference/scripts/inference_psd.py` (new `--use_lama / --no-use_lama`, passed down). Line-by-line list in [MODIFICATIONS.md](MODIFICATIONS.md).

### 5.3 Layer fixes (`layer_fix`)

Upstream post-processing has two defects that make outputs look like something is missing. This project patches both; the "Layer fixes" switch in 02 turns them off per job (`STG_LAYER_FIX=0` restores upstream behaviour exactly):

| Symptom | Cause | Fix |
| ----------- | ----------------------------------------------------------------------------------------------- | ---------------------------------- |
| A whole footwear layer disappears, feet show through | `load_part()` checks for an empty layer by first cropping the bottom/right 10% band, then counting alpha pixels; layers hugging the bottom edge (footwear bounding boxes often end exactly at row 768) count as 0 → the layer is dropped | Count the whole mask instead |
| One eye turns white, no pupil | `dump_parts_psd()` stacks layers by `depth_median`, and the three eye layers are extremely close (measured: eyewhite 0.341 / irides 0.349) — noise puts the eyewhite above the iris | Group the three layers and order them eyewhite → irides → eyelash, keeping their overall position |

It **only reorders / keeps existing pixels, it never repaints**: re-running post-processing on the same 768 output took the layer count from 22 to 23 (footwear back) and blank pixels versus the source image from 4728 down to 991. Patch script `backend/patches/apply_layer_fix.py` (idempotent, run automatically by `1-setup.bat`).

---

## 6. States and errors

| Scenario | Behaviour |
| ----------- | ----------------------------------------------------- |
| Models missing | Pre-checked at startup; a red bar lists what is missing, buttons are disabled, no request is sent |
| Wrong format / corrupt file | Rejected at upload with the reason (extension, size, decode failure listed separately) |
| Invalid parameters | Inline hint in the UI + backend 400 with field-level errors; inference never starts |
| Inference process failed | Status label "failed", UI stays usable, the full traceback is kept in the log, `error_detail` carries the exit code |
| PSD export failed | Status "done · with warnings", distinct from an inference failure; layer PNGs are still downloadable |
| Cancel | Kill the process tree, status "canceled", temp files stay in the job directory and can be removed by hand |
| Service unreachable | Prompts you to check 2-launch.bat; no white screen, no stuck loading state |

---

## 7. Troubleshooting

| Symptom | Cause and fix |
| ------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `no kernel image is available for execution on the device` | The PyTorch wheel has no kernels for your GPU architecture. Install **torch 2.8.0+cu128** as pinned by `requirements-torch.txt`; `backend/selfcheck.py` prints `torch.cuda.get_arch_list()` |
| The log shows `Compiler: cl is not found` | STM compiles the dequantisation kernel with `torch.compile`, which needs MSVC on Windows. `config.json` already sets `compile_dequant: false` (i.e. `SEETHROUGH_COMPILE_DEQUANT=0`), so usually nothing to do |
| Model download is slow or fails | Run `backend\download_models.py --check-only` to see both sources; force one with `--endpoint hf-mirror` / `--endpoint huggingface`; `config.json`'s `hf_endpoint` defaults to `auto` and is also used for runtime component downloads |
| First inference hangs while downloading helper components | Hugging Face is still required; the server probes at startup and sets `HF_ENDPOINT` (mirror in mainland China) |
| Download stalls at a few tens of MB with `xet` / `cas-bridge.xethub.hf.co` timeouts | HF's xet transport connects to overseas nodes directly and often times out in mainland China | The mirror path now sets `HF_HUB_DISABLE_XET=1` automatically (falls back to plain HTTP); older copies can `set HF_HUB_DISABLE_XET=1` manually and retry |
| `CERTIFICATE_VERIFY_FAILED: self-signed certificate in certificate chain` | Security software / corporate proxy is intercepting HTTPS. The app merges system roots into `outputs/.cache/ca-bundle.pem`; the download script also has a curl fallback (system certificates, resumable) |
| Out of VRAM / swapping to death | Make sure "GGUF quantised + streaming" is on (default) and drop the resolution to 720 with ~20 steps |
| VRAM still occupied after a job | "Keep model loaded" (in 03, on by default) holds the model. Turn it off and run once more, or cancel — cancelling also stops the persistent worker |
| Logs stop scrolling while the job runs | Some security suites buffer SSE; the page polls `/api/status` every 1.5 s as a fallback, which is normal. During sampling you should see `进度 12/30 · 40% · 37.2s/it` — only a completely frozen line means a real hang |
| Minutes per step, GPU at 100% but only 40–50 W | The resolution pushed you into the `streamed` tier: weights cross PCIe layer by layer and the GPU waits for data most of the time. Dropping to 768 returns to `resident` — on a 6 GB card **1024 ≈ 300 s/step → 768 ≈ 37 s/step** |
| Want the `torch.compile` dequantisation speed-up (only meaningful for CC ≥ 7.0) | Double-click `4-install_triton.bat`: it inspects the environment → derives the matching Triton version → measures mirrors → installs → smoke-tests with a **real dequantisation kernel**, and only then sets `compile_dequant: true`. Running it again lets you keep / uninstall / reinstall. Cards below CC 7.0 are told clearly they do not qualify and the flag is never enabled. Commands and troubleshooting: [docs/triton-windows.en.md](docs/triton-windows.en.md); if you would rather not do it yourself, hand [docs/triton-windows-ai.md](docs/triton-windows-ai.md) to an AI |
| Port already in use | Change `port` in `config.json`, or stop whatever holds 7860 |

### Optional: install xformers (attention backend)

By default the app uses PyTorch's own SDPA (`attention_backend: native`) and installs nothing extra. To enable the `xformers` button in the advanced panel, three steps:

```bat
:: 1) confirm torch is 2.8.0+cu128
.venv\Scripts\python.exe -c "import torch;print(torch.__version__, torch.version.cuda)"

:: 2) install the matching version with uv (this venv has no pip; python -m pip fails with No module named pip)
uv pip install --python .venv\Scripts\python.exe --no-deps xformers==0.0.32.post1

:: 3) smoke test: not just an import, run a real attention kernel
.venv\Scripts\python.exe -c "import torch,xformers,xformers.ops as xo;q=torch.randn(1,64,8,32,device='cuda',dtype=torch.float16);print(xformers.__version__, xo.memory_efficient_attention(q,q,q).shape)"
```

Version pairing: **torch 2.8.x → `xformers 0.0.32.post1`**, torch 2.10+ → `0.0.35`. For other versions check that release's `requires-dist` on PyPI (it pins `torch==x.y.z`) or run `uv pip index versions xformers`. `--no-deps` is mandatory — otherwise dependency resolution may swap torch for a different build.

Restart `2-launch.bat` afterwards; only then does the `xformers` button in 02 → advanced → attention backend stop being greyed out. The default stays `native` until you click it. Uninstall with `uv pip uninstall --python .venv\Scripts\python.exe xformers`.

Measured on a GTX 1060 / sm_61 (B=1, H=8, D=64, fp16): a single attention call drops from 22.1 ms to 17.8 ms at N=4096 and from 78.0 to 70.4 ms at N=8192, with identical peak memory. End-to-end gains (UNet + VAE + layered sampling) will be smaller than that ratio — run one 768 job before and after if you want a real number.

---

## 8. Project layout

```
SeeThrough-GGUF-Win/
├── 1-setup.bat / 2-launch.bat  # rebuild env / launch
├── 3-env.bat                   # cmd prompt with .venv activated (settings button calls it too)
├── 4-install_triton.bat        # optional: install/uninstall triton-windows (speed up dequant compile)
├── requirements.txt           # STM dependencies + Flask (PyPI)
├── requirements-torch.txt     # PyTorch cu128 (dedicated index, architecture notes inside)
├── config.json                # port, paths, defaults, constraints
├── LICENSE                    # Apache-2.0 (whole repository)
├── NOTICE                     # copyright and upstream attribution
├── MODIFICATIONS.md           # line-by-line declaration of upstream changes (Apache-2.0 §4b)
├── THIRD_PARTY_NOTICES.md     # third-party components and model licences
├── .gitignore                 # excludes .venv / models / outputs / caches
├── README.md / README.en.md    # Chinese (Simplified) / English
├── backend/
│   ├── server.py              # Flask: upload / validation / subprocess / SSE / download / cancel
│   ├── selfcheck.py           # environment self-check
│   ├── worker.py              # persistent inference process (reuses loaded models)
│   ├── components_paths.py    # component cache paths (HF_HOME / TORCH_HOME → components/)
│   ├── install_triton.py      # Triton installer (version mapping / mirror probe / smoke test / config sync)
│   ├── netcheck.py            # download source probe (official / mirror)
│   ├── download_models.py     # download both GGUFs using the probed source
│   ├── tls.py                 # merge system root certificates (security-suite interception)
│   ├── test_pipeline.py       # backend pipeline integration test
│   ├── requirements.txt
│   └── patches/apply_use_lama.py / apply_layer_fix.py
├── frontend/                  # index.html / styles.css / app.js (vanilla JS + GSAP)
├── docs/
│   ├── screenshots/           # UI screenshots (workbench / settings, zh + en)
│   ├── ai-setup.md            # zero-to-running brief: hand the whole file to an AI
│   ├── triton-windows.md      # triton-windows install / manual install / uninstall / troubleshooting
│   ├── triton-windows.en.md   # the same guide in English
│   └── triton-windows-ai.md   # same thing as an AI brief
├── stm/                       # upstream snapshot (Apache-2.0, vendored and patched)
├── models/                    # GGUF (downloaded by you, not in the repo)
├── components/                # first-run helper components (HF / torch.hub, not in the repo; created by 1-setup.bat)
├── outputs/jobs/<id>/         # per-job input, outputs, meta.json (not in the repo)
└── licenses/
    └── README.md              # licence notes and obligations
```

---

## 9. Credits and licence

Repository: <https://github.com/Mingri196/SeeThrough-GGUF-Win>

This project builds on [IceFog72/see-through-gguf](https://github.com/IceFog72/see-through-gguf) (Apache-2.0, baseline commit `02cba07`), which in turn builds on [shitagaki-lab/see-through](https://github.com/shitagaki-lab/see-through) and the paper *See-through: Layer Decomposition for Live2D Character Animation* (<https://arxiv.org/abs/2602.03749>). Those attributions are kept as required; the changes made to upstream code are listed in [MODIFICATIONS.md](MODIFICATIONS.md).

| Component | Licence | Text |
| ----------------------------------- | -------------------------------------------------- | --------------------------------------- |
| This project's code (backend / frontend / scripts / docs) | Apache-2.0 | [LICENSE](LICENSE) + [NOTICE](NOTICE) |
| see-through-gguf (STM) code (vendored `stm/`) | Apache-2.0 | [LICENSE](LICENSE) (original `stm/LICENSE` kept) |
| GSAP 3.12.5 (frontend/vendor) | GreenSock Standard License | copyright header kept in the file |
| GGUF weights and runtime components | see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) | not distributed with this repository |

**About model weights**: this repository contains code and scripts only, no weights. As of the 2026-10-29 check, both GGUF repositories (`icefog72/seethroughv0.0.2_layerdiff3d_gguf`, `icefog72/seethroughv0.0.1_marigold_gguf`) **declare no licence**; their source models fall under `openrail++` (CreativeML OpenRAIL++-M, with usage restrictions). Before sharing generated images or redistributing weights, follow those restrictions and confirm the terms with the model authors.
