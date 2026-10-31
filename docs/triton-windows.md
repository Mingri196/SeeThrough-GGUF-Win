# triton-windows 安装指南

[English](triton-windows.en.md) · **简体中文**

面向本项目（SeeThrough-GGUF-Win）的 Triton 安装 / 卸载说明：**脚本安装**、**手动安装**、**卸载回滚**都在这里；不想自己敲命令的话，把 [triton-windows-ai.md](triton-windows-ai.md) 丢给 AI 让它代劳（见第 6 节）。

涉及的文件：

| 文件 | 作用 |
|---|---|
| `4-install_triton.bat` | 双击入口（纯 ASCII 三行壳，只负责找 `.venv` 并把参数透传） |
| `backend/install_triton.py` | 真正的向导：探测 → 版本推导 → 选源 → 安装 → 冒烟 → 同步配置 |
| `config.json` → `compile_dequant` | 开关。向导只在**冒烟通过**时才把它改成 `true` |
| `backend/server.py` | 起推理子进程时把该开关翻译成环境变量 `SEETHROUGH_COMPILE_DEQUANT=1/0` |
| `stm/common/utils/ggml_runtime.py` | 上游用 `torch.compile` 编译 Q4_K / Q6_K 反量化核的地方 |

---

## 1. 先说清楚值不值

`compile_dequant` 打开后，STM 会用 `torch.compile` 编译两个反量化 kernel（Q4_K、Q6_K）。这条路在 Windows 上需要 **triton-windows** 作为 `torch.compile` 的 CUDA 后端——OpenAI 官方 Triton 不发 Windows 轮子，只能用社区构建的 `triton-windows`。

但有两条硬门槛和一个期望值，先看再装：

| 项目 | 说明 |
|---|---|
| **显卡门槛** | 需要 compute capability **≥ 7.0**（Turing 及更新）。STM 的反量化 kernel 用了 `evict_last` 缓存提示，ptxas 在 sm < 70 上直接拒绝编译。老卡装了也只能 `import` 成功，一编译就报 `GPUTooOldForTriton` / `evict_last ... requires sm_70` |
| **编译门槛** | Windows 上 `torch.compile`（inductor）还要 **MSVC 的 `cl.exe`**（Visual Studio 或 Build Tools 勾选「使用 C++ 的桌面开发」）。缺了会报 `Compiler: cl is not found` |
| **期望收益** | 反量化 kernel **单点**加速很明显（本项目在 6 GB 显存设备上测到 5.88×、数值 bit-exact），但整轮出图只省 **10–13%**——因为瓶颈在 UNet 采样，反量化只占一小段。分辨率越高、流式加载占比越大，收益才越明显 |

所以：**CC ≥ 7.0 且想压榨流式档耗时**，值得装；否则直接跳过，项目默认 `compile_dequant: false`，不装完全不影响出图。

---

## 2. 脚本安装（推荐）

前提：已经跑过 `1-setup.bat`，项目根目录下有 `.venv`。没有 `.venv` 时脚本会直接拒绝——它**只动项目自己的虚拟环境，不碰系统 Python**。

```bat
:: 在项目根目录
4-install_triton.bat            :: 交互式，逐步问
4-install_triton.bat --yes      :: 全程默认值（自动选源，不问任何问题）
```

### 2.1 交互流程

1. **打印环境**：`.venv` 路径、torch 版本、显卡 compute capability、已装的 triton 版本。
2. **已安装时先问**：`1) 保留并退出  2) 卸载  3) 重装`（直接回车 = 保留）。
3. **推导版本**：按 torch 次版本查表，打印类似 `torch 2.8.0+cu128 → 安装 triton-windows==3.4.*`。查不到对应关系时会警告并询问是否冒险装最新版。
4. **显卡门槛**：CC < 7.0 时提示「装上也无法生成 GPU kernel，加速不会生效」，需输 `y` 二次确认；无 NVIDIA GPU 同样二次确认。
5. **测速选源**：并行测三个源，打印延迟与实际下载速度，直接回车 = 自动选最快的：

   ```
   == 选择下载源 ==
     1) 清华镜像     320 ms ·  6.2 MiB/s
     2) 阿里镜像     280 ms ·  4.1 MiB/s
     3) PyPI 官方   2100 ms ·  0.3 MiB/s
   ```
6. **安装** → 复查版本 → **冒烟测试**：不是跑 sin/cos 糊弄，而是 `import` STM 真实的 `dequantize_blocks_Q4_K`，`torch.compile` 编译一次并与 eager 结果比对（`atol/rtol = 1e-3`）。通过才把 `config.json` 的 `compile_dequant` 置 `true`。

### 2.2 参数

| 参数 | 作用 |
|---|---|
| `--yes` | 全部用默认值，选源自动，适合批处理 / 无人值守 |
| `--uninstall` | 直接卸载（同时把 `compile_dequant` 改回 `false`） |
| `--reinstall` | 跳过「已安装」三选，直接卸载后重装 |
| `--channel 1\|2\|3` | 指定源：`1` 清华 / `2` 阿里 / `3` PyPI 官方，跳过测速提问 |

示例：

```bat
4-install_triton.bat --channel 2            :: 国内网络卡在测速上时直接指定阿里源
4-install_triton.bat --reinstall --yes      :: 无交互重装
4-install_triton.bat --uninstall            :: 无交互卸载
```

> 脚本**不提供 GitHub 源**：`triton-windows` 的 GitHub Releases 在国内多数网络不可达，走 PyPI 三源更稳。

### 2.3 结果怎么读

| 结尾输出 | 含义 | 退出码 |
|---|---|---|
| `[完成] 已启用 compile_dequant（config.json → true）` | 全部通过，重启服务生效 | 0 |
| `[已安装，但未启用] config.json 的 compile_dequant 保持 false` | 装上了但冒烟没过（多为 CC < 7.0，属预期），功能不受影响 | 0 |
| `[拒绝] 项目根目录下没有 .venv` | 先跑 `1-setup.bat` | 1 |
| `[失败] 安装命令返回非零` / `安装后仍然检测不到` | 网络或版本问题，看上方打印的 pip 输出 | 1 |

装完必须**重启服务**（关掉 `2-launch.bat` 的窗口再开）——`config.json` 在服务启动时读取，环境变量在起推理子进程时才注入。页面「模型常驻」还留着的旧 worker 也不会热更新，建议先在 03 里关掉常驻或取消任务再重启。

---

## 3. 手动安装

不想跑脚本、或者要装到别的环境时，按下面来。**核心只有一条：triton 版本必须和 torch 精确配对。**

### 3.1 版本对应表

| torch 次版本 | 安装 `triton-windows` |
|---|---|
| 2.4 / 2.5 | `3.1.*` |
| 2.6 | `3.2.*` |
| 2.7 | `3.3.*` |
| 2.8 | `3.4.*` ← 本项目 `requirements-torch.txt` 固定 torch 2.8.0+cu128，用这档 |
| 2.9 | `3.5.*` |
| 2.10 | `3.6.*` |

### 3.2 命令

```bat
:: 1) 看当前 torch（必须是 2.8.x）
.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.get_device_capability(0))"

:: 2) 安装。有 uv 用这个（注意 --python 在子命令之后）
uv pip install --python .venv\Scripts\python.exe "triton-windows==3.4.*" --index-url https://pypi.tuna.tsinghua.edu.cn/simple

:: 没有 uv 时退回 pip
.venv\Scripts\python.exe -m pip install "triton-windows==3.4.*" -i https://pypi.tuna.tsinghua.edu.cn/simple

:: 3) 确认能 import（不报 undefined symbol 就说明 ABI 对上了）
.venv\Scripts\python.exe -c "import triton, torch; print('triton', triton.__version__, '| torch', torch.__version__)"
```

国内三个等价源，挑快的：

```text
https://pypi.tuna.tsinghua.edu.cn/simple      清华
https://mirrors.aliyun.com/pypi/simple        阿里
https://pypi.org/simple                       PyPI 官方
```

### 3.3 手动启用前先冒烟

**不要装完就把 `compile_dequant` 改 true**。先在项目根目录存一个 `outputs/smoke_triton.py`（`outputs/` 不入库，放这儿不脏仓库）：

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

打出 `allclose: True` 才继续。报错就把报错原文对照第 5 节。

### 3.4 启用

编辑 `config.json`：

```json
"compile_dequant": true
```

重启服务即可。想临时验证不写文件，也可以在启动前设环境变量 `SEETHROUGH_COMPILE_DEQUANT=1`（但 `config.json` 里是 `false` 时服务会覆盖它，所以还是改配置）。

---

## 4. 卸载与关闭

### 4.1 脚本卸载（推荐）

```bat
4-install_triton.bat --uninstall
```

或者双击后选 `2) 卸载`。它会 `uv pip uninstall triton-windows`，并**自动把 `compile_dequant` 改回 `false`**。

### 4.2 手动卸载

```bat
uv pip uninstall --python .venv\Scripts\python.exe triton-windows
:: 没有 uv 时
.venv\Scripts\python.exe -m pip uninstall -y triton-windows
```

然后**必须**把 `config.json` 的 `compile_dequant` 改回 `false`。

> 顺序很重要：只卸包不改配置，下次启动仍会注入 `SEETHROUGH_COMPILE_DEQUANT=1`，`ggml_runtime.py` 在 import 阶段就要编译 kernel，缺后端会让整个推理直接失败。

### 4.3 可选：清缓存

`torch.compile` 的产物不进 `.venv`，卸载后可能还留着：

| 位置 | 内容 |
|---|---|
| `%USERPROFILE%\.triton\cache` | Triton kernel 编译缓存 |
| `%TEMP%\torchinductor_*` | inductor 生成的 C++ / 中间产物 |

删不删都不影响正确性，只是换版本后首次编译会重新生成。清理时只动这两个目录内部，别整目录递归删。

---

## 5. 常见问题

| 现象 | 原因 | 处理 |
|---|---|---|
| `import triton` 报 `undefined symbol: ...` / `DLL load failed` | triton 与 torch 版本不配对（ABI 不符） | 按 3.1 的表重装对应版本；不要混用 `pip` 与 `uv` 装出来的半套 |
| 编译时报 `Compiler: cl is not found` / `fatal error C1083` | 缺 MSVC。Windows 上 inductor 需要 `cl.exe` | 装 Visual Studio 或 Build Tools，勾「使用 C++ 的桌面开发」，装完重开终端 |
| `GPUTooOldForTriton`、`evict_last ... requires sm_70 or higher`、`ptxas fatal` | 显卡 CC < 7.0，kernel 里的 `evict_last` 缓存提示不被支持 | 预期结果。卸载并把 `compile_dequant` 设回 `false`，本项目在 CC < 7.0 的卡上本来就默认关 |
| 装好了但出图没变快 | 整轮耗时主要花在 UNet 采样，反量化只占一小段；分辨率低时占比更小 | 属正常。实测整轮 10–13%，别指望翻倍 |
| 向导选源阶段一片 `不可用` | 全部源都不通（代理 / 防火墙） | `--channel` 指定某个源再试，或先确认 `python -c "import urllib.request; urllib.request.urlopen('https://pypi.org/simple', timeout=10)"` |
| `--yes` 跑完显示「未启用」 | 冒烟没过（门槛或 MSVC） | 看上方冒烟打印的 `reason`，对照本表前三条 |
| 服务重启后仍然没加速 | 旧常驻 worker 还在（环境变量是起进程时注入的） | 03 里关掉「模型常驻」或取消任务，再重启 `2-launch.bat` |

自测钩子（仅调试脚本分支用，正常使用不要设）：

```bat
set STG_TRITON_TEST_CC=8,6        :: 伪造显卡算力
set STG_TRITON_TEST_TORCH=2.8.0   :: 伪造 torch 版本
```

---

## 6. 让 AI 帮忙操作 → [triton-windows-ai.md](triton-windows-ai.md)

不想看命令、也不想复制提示词：**把 [docs/triton-windows-ai.md](triton-windows-ai.md) 直接交给 AI**（拖文件、或说「按 `docs/triton-windows-ai.md` 操作」），再补一句你要做什么——「装」「卸」「我装了但没变快」——它会自己诊断、判定能不能装、该装哪个版本，然后才动手。

那份文档是写给 AI 的任务书，包含：

- 三条执行路径（安装 / 卸载 / 效果排查）与判定表（缺 `.venv`、CC < 7.0、版本错配、CUDA 不可用分别怎么办）；
- 完成判据（向导的哪句输出算成功）、必须由用户自己做的动作（重启服务）；
- 红线清单：不动系统 Python、不改 `stm/` 源码、不伪造探测、冒烟未过不得开 `compile_dequant`、不装 VS、不提交 git、不猜路径；
- 统一汇报模板。

你只要判断它报的结论是否合理即可；命令细节仍然以本文档第 2–4 节为准。

---

## 7. 附：向导内部行为速查

- **入口**：`4-install_triton.bat` 先检查 `.venv\Scripts\python.exe`，缺了直接 `[ERROR] .venv not found` 并 `pause`；有则透传所有参数给 `backend/install_triton.py`。
- **探测**：一次子进程拿齐 torch 版本、CUDA 可用性、compute capability、已装 triton 版本。
- **安装命令**：有 `uv` 时 `uv pip install --python <venv python> <spec> --index-url <源>`，否则 `python -m pip install ...`（`uv` 的 `--python` 必须在子命令之后）。
- **选源测速**：并行拉三个源的 `/simple/triton-windows/`，从索引里找带 `win_amd64` 的匹配轮子，`Range: bytes=0-1048575` 下 1 MiB 实测速度，取最快。
- **冒烟**：`torch.compile(G.dequantize_blocks_Q4_K, dynamic=True)` 后与 eager 比对；用 `os.getcwd()` 定位项目根（`python -c` 没有 `__file__`）。
- **写配置**：只改 `compile_dequant` 一个键，保留原有缩进与其余内容。
- **不碰系统**：全程只用 `.venv` 内的解释器，不写注册表、不改 PATH、不装全局包。
