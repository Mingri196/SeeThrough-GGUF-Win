# triton-windows installation guide

**English** · [简体中文](triton-windows.md)

Triton install / uninstall instructions for this project (SeeThrough-GGUF-Win): **scripted install**, **manual install** and **uninstall / rollback** all live here. If you would rather not type commands, hand [triton-windows-ai.md](triton-windows-ai.md) to an AI (see section 6).

Files involved:

| File | Purpose |
|---|---|
| `4-install_triton.bat` | Double-click entry point (three ASCII lines that locate `.venv` and pass arguments through) |
| `backend/install_triton.py` | The real wizard: probe → derive version → pick a mirror → install → smoke test → sync config |
| `config.json` → `compile_dequant` | The switch. The wizard only flips it to `true` **after the smoke test passes** |
| `backend/server.py` | Translates that switch into `SEETHROUGH_COMPILE_DEQUANT=1/0` when spawning the inference subprocess |
| `stm/common/utils/ggml_runtime.py` | Where upstream compiles the Q4_K / Q6_K dequantisation kernels with `torch.compile` |

---

## 1. Is it worth it?

With `compile_dequant` enabled, STM compiles two dequantisation kernels (Q4_K, Q6_K) using `torch.compile`. On Windows that path needs **triton-windows** as the CUDA backend for `torch.compile` — official OpenAI Triton ships no Windows wheels, so the community build `triton-windows` is the only option.

But there are two hard gates and one expectation to set before installing:

| Item | Notes |
|---|---|
| **GPU gate** | Requires compute capability **≥ 7.0** (Turing or newer). STM's dequantisation kernel uses the `evict_last` cache hint, and ptxas refuses to compile it for sm < 70. On older cards the install succeeds and `import` works, but the first compile fails with `GPUTooOldForTriton` / `evict_last ... requires sm_70` |
| **Toolchain gate** | `torch.compile` (inductor) on Windows also needs **MSVC `cl.exe`** (Visual Studio or Build Tools with "Desktop development with C++"). Without it you get `Compiler: cl is not found` |
| **Expected gain** | The dequantisation kernel itself gets a big **single-point** speed-up (measured 5.88× on a 6 GB card in this project, bit-exact numerically), but a full image only saves **10–13%** — the bottleneck is UNet sampling and dequantisation is a small slice. The higher the resolution and the larger the streaming share, the more it matters |

So: **if CC ≥ 7.0 and you want to squeeze the streamed tier**, install it; otherwise skip it. The project defaults to `compile_dequant: false` and works fine without it.

---

## 2. Scripted install (recommended)

Prerequisite: `1-setup.bat` has been run and `.venv` exists in the project root. Without `.venv` the script refuses outright — it **only touches the project's own virtual environment, never the system Python**.

```bat
:: from the project root
4-install_triton.bat            :: interactive, asks step by step
4-install_triton.bat --yes      :: all defaults (auto mirror, no questions)
```

### 2.1 Interactive flow

1. **Print the environment**: `.venv` path, torch version, GPU compute capability, installed triton version.
2. **If already installed, ask first**: `1) keep and exit  2) uninstall  3) reinstall` (Enter = keep).
3. **Derive the version**: look up the torch minor version and print something like `torch 2.8.0+cu128 → installing triton-windows==3.4.*`. If no mapping exists it warns and asks whether to gamble on the latest.
4. **GPU gate**: with CC < 7.0 it warns that "no GPU kernel can be generated, the speed-up will not apply" and requires a second `y`; the same applies when no NVIDIA GPU is found.
5. **Mirror speed test**: probes three mirrors in parallel, prints latency and real throughput; Enter = pick the fastest:

   ```
   == select download source ==
     1) Tsinghua   320 ms ·  6.2 MiB/s
     2) Aliyun     280 ms ·  4.1 MiB/s
     3) PyPI       2100 ms ·  0.3 MiB/s
   ```
6. **Install** → re-check version → **smoke test**: not a meaningless sin/cos, but `import`ing STM's real `dequantize_blocks_Q4_K`, compiling it once with `torch.compile` and comparing against eager (`atol/rtol = 1e-3`). Only then is `compile_dequant` set to `true` in `config.json`.

### 2.2 Arguments

| Argument | Purpose |
|---|---|
| `--yes` | Accept every default, auto-pick a mirror; good for batch / unattended runs |
| `--uninstall` | Uninstall directly (also resets `compile_dequant` to `false`) |
| `--reinstall` | Skip the "already installed" prompt and reinstall |
| `--channel 1\|2\|3` | Force a mirror: `1` Tsinghua / `2` Aliyun / `3` PyPI; skips the speed test |

Examples:

```bat
4-install_triton.bat --channel 2            :: when the speed test stalls on your network
4-install_triton.bat --reinstall --yes      :: non-interactive reinstall
4-install_triton.bat --uninstall            :: non-interactive uninstall
```

> The script **does not offer a GitHub source**: `triton-windows` GitHub Releases are unreachable from many mainland China networks; the three PyPI mirrors are more reliable.

### 2.3 Reading the result

| Final output | Meaning | Exit code |
|---|---|---|
| `[done] compile_dequant enabled (config.json → true)` | Everything passed; restart the service | 0 |
| `[installed, not enabled] compile_dequant stays false` | Installed but the smoke test failed (usually CC < 7.0, expected); functionality unaffected | 0 |
| `[refused] no .venv in the project root` | Run `1-setup.bat` first | 1 |
| `[failed] install command returned non-zero` / `still not detected after install` | Network or version problem; read the pip output above | 1 |

Afterwards you **must restart the service** (close the `2-launch.bat` window and start it again) — `config.json` is read at startup and the environment variable is injected when the inference subprocess is created. A warm worker kept alive by the "keep model loaded" option will not pick it up either; turn that off in 03 or cancel the job before restarting.

---

## 3. Manual install

Use this if you do not want the script or need to install into a different environment. **One rule matters: the triton version must match torch exactly.**

### 3.1 Version mapping

| torch minor | install `triton-windows` |
|---|---|
| 2.4 / 2.5 | `3.1.*` |
| 2.6 | `3.2.*` |
| 2.7 | `3.3.*` |
| 2.8 | `3.4.*` ← `requirements-torch.txt` pins torch 2.8.0+cu128, so use this row |
| 2.9 | `3.5.*` |
| 2.10 | `3.6.*` |

### 3.2 Commands

```bat
:: 1) check the current torch (must be 2.8.x)
.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.get_device_capability(0))"

:: 2) install. With uv use this (note: --python goes after the subcommand)
uv pip install --python .venv\Scripts\python.exe "triton-windows==3.4.*" --index-url https://pypi.tuna.tsinghua.edu.cn/simple

:: without uv, fall back to pip
.venv\Scripts\python.exe -m pip install "triton-windows==3.4.*" -i https://pypi.tuna.tsinghua.edu.cn/simple

:: 3) confirm the import works (no undefined symbol means the ABI matches)
.venv\Scripts\python.exe -c "import triton, torch; print('triton', triton.__version__, '| torch', torch.__version__)"
```

Three equivalent mirrors — pick the fast one:

```text
https://pypi.tuna.tsinghua.edu.cn/simple      Tsinghua
https://mirrors.aliyun.com/pypi/simple        Aliyun
https://pypi.org/simple                       PyPI (official)
```

### 3.3 Smoke test before enabling

**Do not flip `compile_dequant` to true right after installing.** Save this as `outputs/smoke_triton.py` in the project root first (`outputs/` is not version-controlled, so it keeps the repo clean):

```python
import json, os, sys
root = os.getcwd()
stm = os.path.join(root, "stm")
for entry in (os.path.join(stm, "common"), stm):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import torch, gguf
from utils import ggml_runtime as G

qtype = gguf.GGMLQuantizationType.Q4_K
block_size, type_size = gguf.GGML_QUANT_SIZES[qtype]
blocks = torch.randint(0, 256, (128, type_size), dtype=torch.uint8, device="cuda")
blocks[:, 0:2] = torch.tensor([0x00, 0x38], dtype=torch.uint8, device="cuda")
blocks[:, 2:4] = torch.tensor([0x00, 0x34], dtype=torch.uint8, device="cuda")

eager = G.dequantize_blocks_Q4_K(blocks, block_size, type_size, torch.float16)
out = torch.compile(G.dequantize_blocks_Q4_K, dynamic=True)(blocks, block_size, type_size, torch.float16)
torch.cuda.synchronize()
print("allclose:", torch.allclose(out, eager, atol=1e-3, rtol=1e-3))
```

```bat
.venv\Scripts\python.exe outputs\smoke_triton.py
```

Only continue once it prints `allclose: True`. If it errors, match the message against section 5.

### 3.4 Enable

Edit `config.json`:

```json
"compile_dequant": true
```

Restart the service. To test without touching the file you could set `SEETHROUGH_COMPILE_DEQUANT=1` before launch, but the service overrides it when `config.json` says `false` — so edit the file.

---

## 4. Uninstall and disable

### 4.1 Scripted uninstall (recommended)

```bat
4-install_triton.bat --uninstall
```

Or double-click and choose `2) uninstall`. It runs `uv pip uninstall triton-windows` and **automatically resets `compile_dequant` to `false`**.

### 4.2 Manual uninstall

```bat
uv pip uninstall --python .venv\Scripts\python.exe triton-windows
:: without uv
.venv\Scripts\python.exe -m pip uninstall -y triton-windows
```

Then you **must** set `compile_dequant` back to `false` in `config.json`.

> Order matters: uninstalling the package but leaving the flag on means the next startup still injects `SEETHROUGH_COMPILE_DEQUANT=1`, and `ggml_runtime.py` compiles kernels at import time — a missing backend kills the whole inference run.

### 4.3 Optional: clear caches

`torch.compile` artefacts do not live in `.venv` and may survive an uninstall:

| Location | Contents |
|---|---|
| `%USERPROFILE%\.triton\cache` | Triton kernel compile cache |
| `%TEMP%\torchinductor_*` | inductor-generated C++ / intermediate files |

Deleting them or not does not affect correctness; the next compile after a version change regenerates them. Only remove the contents of those directories, never the whole tree recursively.

---

## 5. FAQ

| Symptom | Cause | Fix |
|---|---|---|
| `import triton` fails with `undefined symbol: ...` / `DLL load failed` | triton and torch versions do not match (ABI mismatch) | Reinstall the matching version from the table in 3.1; do not mix halves installed by `pip` and `uv` |
| Compile fails with `Compiler: cl is not found` / `fatal error C1083` | MSVC missing. inductor needs `cl.exe` on Windows | Install Visual Studio or Build Tools with "Desktop development with C++", then reopen the terminal |
| `GPUTooOldForTriton`, `evict_last ... requires sm_70 or higher`, `ptxas fatal` | GPU CC < 7.0; the `evict_last` cache hint in the kernel is unsupported | Expected. Uninstall and set `compile_dequant` back to `false`; this project keeps it off by default on such cards |
| Installed but images are not faster | Most of the runtime is UNet sampling; dequantisation is a small slice, even smaller at low resolution | Normal. Measured 10–13% end-to-end; do not expect double |
| The wizard shows every mirror as `unavailable` | All sources unreachable (proxy / firewall) | Use `--channel` to force one, or first check `python -c "import urllib.request; urllib.request.urlopen('https://pypi.org/simple', timeout=10)"` |
| `--yes` finishes with "not enabled" | Smoke test failed (gate or MSVC) | Look at the `reason` printed by the smoke test and compare with the first three rows above |
| Still no speed-up after restarting the service | An old persistent worker is alive (the env var is injected at process start) | Turn off "keep model loaded" in 03 or cancel the job, then restart `2-launch.bat` |

Self-test hooks (for debugging the script's branches only — do not set them in normal use):

```bat
set STG_TRITON_TEST_CC=8,6        :: fake GPU compute capability
set STG_TRITON_TEST_TORCH=2.8.0   :: fake torch version
```

---

## 6. Let an AI do it → [triton-windows-ai.md](triton-windows-ai.md)

If you do not want to read commands or copy prompts: **hand [docs/triton-windows-ai.md](triton-windows-ai.md) to an AI** (drag the file in, or say "follow `docs/triton-windows-ai.md`") and add what you want — "install", "uninstall", "I installed it but nothing got faster" — it will diagnose, decide whether installing is possible, pick the right version, and only then act.

That document is a brief written for the AI and covers:

- three execution paths (install / uninstall / investigate) with decision tables (missing `.venv`, CC < 7.0, version mismatch, CUDA unavailable);
- completion criteria (which wizard line counts as success) and the actions the user must take (restart the service);
- red lines: never touch the system Python, never edit `stm/` sources, never fake the probes, never enable `compile_dequant` without a passing smoke test, never install VS, never commit to git, never guess paths;
- a fixed report template.

Your job is only to judge whether its conclusions make sense; command details remain governed by sections 2–4 of this document.

> Note: `triton-windows-ai.md` is written in Chinese.

---

## 7. Appendix: what the wizard does internally

- **Entry**: `4-install_triton.bat` checks for `.venv\Scripts\python.exe`; if missing it prints `[ERROR] .venv not found` and `pause`s; otherwise it passes all arguments to `backend/install_triton.py`.
- **Probe**: one subprocess collects torch version, CUDA availability, compute capability and installed triton version.
- **Install command**: `uv pip install --python <venv python> <spec> --index-url <mirror>` when `uv` exists, otherwise `python -m pip install ...` (`uv`'s `--python` must follow the subcommand).
- **Mirror speed test**: fetch `/simple/triton-windows/` from three mirrors in parallel, find a matching wheel with `win_amd64` in the index, download 1 MiB with `Range: bytes=0-1048575` to measure real throughput, take the fastest.
- **Smoke test**: `torch.compile(G.dequantize_blocks_Q4_K, dynamic=True)` compared against eager; locates the project root via `os.getcwd()` (a `python -c` has no `__file__`).
- **Write config**: only the `compile_dequant` key changes; indentation and everything else are preserved.
- **Never touches the system**: only the interpreter inside `.venv` is used; no registry writes, no PATH changes, no global packages.
