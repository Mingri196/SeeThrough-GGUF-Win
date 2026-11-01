# 从零部署：交给 AI 做（任务书）

> **这份文档是给 AI 读的。** 用户只要把它丢给你（或说「按 `docs/ai-setup.md` 把项目部署起来」），你就应该能从一台没装过的 Windows 机器开始，完成：**克隆 → 装环境 → 准备模型 → 启动 → 验收 → 汇报**，然后把使用方法和项目结构讲清楚。
>
> 用户不需要会命令行，但你每一步都要把「正在做什么、为什么、失败了怎么办」说人话。

---

## 1. 给你的角色与边界

**目标**：在 Windows 10/11 x64 上把 SeeThrough-GGUF-Win 跑起来，用户能在浏览器里打开 `http://127.0.0.1:7860` 并出一张分层图。

**这个项目是什么**：把上游 [IceFog72/see-through-gguf](https://github.com/IceFog72/see-through-gguf)（简称 STM，纯 Python + PyTorch 的「单张角色图 → 分层 PSD / 深度图 / 骨骼切割」运行时）包装成 Windows 本地网页工具。上游代码已作为快照内置在 `stm/`，**不需要另外克隆上游**。本地 GGUF 推理，不联网出图（只有首次预取辅助组件要联网）。

**你必须遵守**：

- 只在这个项目目录里干活；不写系统级配置（PATH、注册表、全局 pip 包、系统 Python）。
- 不改 `stm/` 下的上游代码（除了跑它自带的补丁脚本），不改 `frontend/`、`backend/` 的源码去「绕过」报错——先诊断清楚再说。
- 不替用户装显卡驱动、CUDA Toolkit、Visual Studio。
- 不 `git commit` / `git push` / 建分支，除非用户明确要求。
- 不删用户数据：`outputs/jobs/` 下的历史任务、`models/` 下的权重都不要动。
- **路径不写死**：本文出现的 `<项目目录>` 指你实际 clone 的位置，由你（或用户）确定；不确定就先报当前目录再动手。

**先问清楚三件事**（用户没说的就问，别猜）：

1. 仓库地址（`https://github.com/Mingri196/SeeThrough-GGUF-Win.git`；用户给了别的地址就以用户的为准）；
2. 装到哪个目录（磁盘要留 **约 26 GB**：环境约 8 GB + 辅助组件约 2.4 GB + 主模型约 5.4 GB + 余量）；
3. 是否已经有 Python 3.12 / uv。

---

## 2. 第 0 步：环境体检

在项目目录确认之前，先摸清这台机器：

```bat
:: 系统与磁盘（wmic 在部分 Win11 版本已移除，用 PowerShell 那条兜底）
ver
wmic logicaldisk get caption,freespace,size
powershell -NoProfile -Command "Get-PSDrive -PSProvider FileSystem | Select-Object Name,@{n='FreeGB';e={[math]::Round($_.Free/1GB,1)}}"

:: Python 与 uv
py -0p
uv --version

:: 显卡与驱动（nvidia-smi 能打出 CUDA Version 就够了）
nvidia-smi
```

判定：

| 结果 | 处理 |
|---|---|
| 没有 Python 3.12，也没有 uv | 让用户装 Python 3.12（python.org，安装时勾 Add to PATH）或 uv，**不要自己去下载安装包静默安装** |
| 磁盘剩余 < 26 GB | 停止，先让用户清出空间 |
| `nvidia-smi` 不存在或报错 | 停止。本项目需要 NVIDIA 显卡 + 支持 CUDA 12.8 的驱动；集显/AMD 跑不了 |
| 显存 ≤ 4 GB | 提醒：能跑但只能低分辨率（512 左右），出图会很慢 |
| 没有 Git | 用 `git` 克隆不行就走第 3 步的 ZIP 方式 |

---

## 3. 第 1 步：克隆仓库

```bat
:: 选一个干净的父目录，不要在已有 .venv 的旧目录上覆盖
cd /d <父目录>
git clone https://github.com/Mingri196/SeeThrough-GGUF-Win.git SeeThrough-GGUF-Win
cd /d <父目录>\SeeThrough-GGUF-Win
```

没有 Git 时：让用户在浏览器打开仓库页 → Code → Download ZIP → 解压，然后 `cd` 进解压出来的目录（目录名可能是 `SeeThrough-GGUF-Win-main`，名字无所谓）。

**克隆完先验证关键文件在**（缺了说明没下全或被裁剪）：

```bat
if not exist stm\inference\scripts\inference_psd.py echo MISSING stm
if not exist backend\server.py echo MISSING backend
if not exist frontend\index.html echo MISSING frontend
if not exist requirements-torch.txt echo MISSING torch reqs
if not exist 1-setup.bat echo MISSING setup
dir /b
```

期望的顶层内容：`1-setup.bat`、`2-launch.bat`、`3-env.bat`、`4-install_triton.bat`、`requirements.txt`、`requirements-torch.txt`、`config.json`、`backend\`、`frontend\`、`stm\`、`docs\`、`models\`（空）、`outputs\`。

**要知道的三件事**：

- 仓库**不含** `.venv`（不可移植）、`models/*.gguf`（权重不入库）、`components/`（首次推理的辅助组件）、`outputs/`（运行产物）——这几样由本机生成；
- 顶层还有 `LICENSE` / `NOTICE` / `MODIFICATIONS.md` / `THIRD_PARTY_NOTICES.md`，是上游 Apache-2.0 的署名义务，**不要删**；
- 如果目录里已经有 `.venv`，说明是别人装过的副本，先确认要不要复用，不要盲目覆盖。

---

## 4. 第 2 步：安装环境

### 4.1 首选：跑 `1-setup.bat`

双击，或在 cmd 里：

```bat
cd /d <项目目录>
1-setup.bat
```

它会依次做五件事，你按顺序核对输出：

| 步骤 | 做什么 | 成功的标志 | 失败怎么办 |
|---|---|---|---|
| `[1/5]` | 校验内置 `stm/` | `bundled STM code found` | 报 `stm\ not found` → 克隆不全，重下 |
| `[2/5]` | 建 `.venv`（优先 `uv venv --python 3.12`，否则 `py -3.12 -m venv`） | `creating .venv ...` 后无报错，或 `reusing .venv (Python 3.12.x)` | 报 failed → 没装 Python 3.12 / uv |
| `[3/5]` | 先装 `requirements-torch.txt`（cu128 专用索引），再装 `requirements.txt` | 两段 pip 输出都以 `Successfully installed ...` 结束 | 网络问题看第 11 节；**不要换 torch 版本**，见下方注意 |
| `[4/5]` | 跑 `backend/patches/apply_use_lama.py` 校验 `use_lama` 补丁（幂等） | 打印已打过 / 已应用 | 报错说明 `stm/` 被改坏了，重新 clone |
| `[5/5]` | 探测下载源（`netcheck.py`）→ 预取辅助组件（`download_models.py --components-only`）→ 自检（`selfcheck.py`） | 组件进度跑完；自检打印 `[OK] torch ... CUDA ... 显卡名 显存 GB` | 组件失败只是 `[WARN]`，首次推理会重试；自检失败看具体哪项 `[缺失]` |

预计耗时：依赖下载约 8 GB（视网络 10–40 分钟），组件约 2.4 GB（落项目内 `components/`）。

**注意**：`requirements-torch.txt` 钉死 **torch 2.8.0+cu128**。torch 2.9+ 的 cu128 轮子不再包含 sm_61（Pascal），老卡会报 `no kernel image is available`。**不要自作主张升级 torch**；16/20/30/40 系用户想升级也要先问。

### 4.2 备选：手工等价命令（`1-setup.bat` 无法交互时）

`1-setup.bat` 结尾有 `pause`，如果你的执行环境不能交互，就逐步跑等价命令：

```bat
uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe -r requirements-torch.txt
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
.venv\Scripts\python.exe backend\patches\apply_use_lama.py
.venv\Scripts\python.exe backend\netcheck.py
.venv\Scripts\python.exe backend\download_models.py --components-only
.venv\Scripts\python.exe backend\selfcheck.py
```

没有 uv 时把 `uv venv --python 3.12 .venv` 换成 `py -3.12 -m venv .venv`，并把 `uv pip install --python <exe>` 换成 `<exe> -m pip install`。

想跳过组件下载（网络太差、以后再补）：先 `set STG_SKIP_MODELS=1` 再跑 `1-setup.bat`，或干脆不跑 `--components-only`。组件缺失时首次推理会自动再试，但会卡在下载上。

---

## 5. 第 3 步：准备主模型（必须做）

`1-setup.bat` **不下载主模型**，这是刻意的（权重大、许可独立）。两个 GGUF 必须就位：

| 角色 | 上游仓库 | `config.json` 里的默认路径 |
|---|---|---|
| 图层分解（LayerDiff3D） | `icefog72/seethroughv0.0.2_layerdiff3d_gguf` | `models/layerdiff3d-Q5_1.gguf` |
| 深度（Marigold3D） | `icefog72/seethroughv0.0.1_marigold_gguf` | `models/marigold3d-Q8_0.gguf` |

两条路，二选一：

**A. 让脚本下（推荐）**

```bat
:: 只下主模型
.venv\Scripts\python.exe backend\download_models.py

:: 主模型 + 辅助组件
.venv\Scripts\python.exe backend\download_models.py --with-components

:: 只看通不通、远端多大，不下载（验收网络用）
.venv\Scripts\python.exe backend\download_models.py --check-only

:: 强制指定源：国内常见镜像 / 官方
.venv\Scripts\python.exe backend\download_models.py --endpoint hf-mirror
.venv\Scripts\python.exe backend\download_models.py --endpoint huggingface
```

**B. 用户自己下载后放进 `models/`**

可以是子目录（如 `models\layerdiff3d\xxx-Q4_K_M.gguf`），页面会递归扫描。**文件名不必等于默认值**——只要在页面 03 里选中即可，不要为了「对上默认值」去改文件名或改 `config.json`。

**验收**：`models\` 下至少两个 `.gguf`，且分别能对上「图层」「深度」两种角色（页面会按文件名推断，Q5_1/Q4_K_M/Q8_0 等量化档都行）。

---

## 6. 第 4 步：启动与验收

### 6.1 启动

**给人用**：双击 `2-launch.bat`（它会自动开浏览器并前台跑服务，关窗口即停）。

**给你（AI）做自动化验收**：不要用 `2-launch.bat`（它会 `start` 浏览器并 `pause`）。直接后台起服务：

```bat
.venv\Scripts\python.exe backend\server.py
```

然后轮询直到就绪：

```bat
curl -s http://127.0.0.1:7860/api/config
```

`/api/config` 返回 JSON 就说明服务起来了。端口在 `config.json` 的 `port`（默认 7860），只监听 `127.0.0.1`。

### 6.2 验收清单（按顺序，全过才算部署完成）

| # | 检查 | 命令 / 动作 | 期望 |
|---|---|---|---|
| 1 | 配置接口 | `curl -s http://127.0.0.1:7860/api/config` | 返回默认值、约束范围、支持的扩展名 |
| 2 | 模型与补丁状态 | `curl -s http://127.0.0.1:7860/api/check_models` | `layerdiff` / `depth` 都存在；`patch` 显示已打；网络状态正常 |
| 3 | 模型列表 | `curl -s http://127.0.0.1:7860/api/models` | 至少两项，路径形如 `layerdiff3d\xxx.gguf` |
| 4 | 页面能开 | 浏览器 `http://127.0.0.1:7860` | 01–04 四步界面，顶部无红条（有红条就是模型缺失） |
| 5 | 真跑一张 | 见 6.3 | 状态「完成」，成像台出现缩略图 + PSD 下载 |

### 6.3 冒烟出一张图（别用大图）

6 GB 显存设备上 1024 会掉进 `streamed` 档（约 300 s/步，30 步要几小时）。冒烟就用小图：

- 分辨率 **512**（最小 256，必须是 64 的倍数）；
- 步数 **4–6**；
- 深度步数保持默认 **4**；
- 深度分辨率 `auto`；
- 保持「GGUF 量化 + 流式加载」开（默认）、「模型常驻」开（默认）。

命令行跑等价（也可让用户在页面上点）：

```bat
curl -s -X POST http://127.0.0.1:7860/api/run ^
  -F "image=@<一张人物图.png>" ^
  -F "params={\"width\":512,\"height\":512,\"inference_steps\":6,\"seed_mode\":\"fixed\",\"seed\":42,\"quantized\":true,\"keep_model\":true}"
```

返回 `{ok, job_id, seed_used, params}`，然后：

```bat
curl -s http://127.0.0.1:7860/api/status/<job_id>
```

状态从 `running` 变 `done`（或 `failed` / `canceled`）。失败时看 `logs` 末尾的 traceback。产物在 `outputs\jobs\<job_id>\`。

> 只允许一个任务同时跑（单卡并行只会互相抢显存），重复提交返回 `409 busy`。

---

## 7. 怎么用（讲给用户听）

页面四步，从上到下走：

| 步骤 | 用户要做什么 |
|---|---|
| **01 图像** | 拖拽或点选一张人物图（PNG / JPG / WEBP / BMP）。会即时校验扩展名、体积、可解码性，显示缩略图与像素尺寸；换图会清空上一轮结果 |
| **02 参数** | 宽/高（同步，64 的倍数）、推理步数、种子（固定/随机）、**输出图层**（24 个 tag，身体组 13 + 头部组 11，默认全选；默认收起，点「展开」才显示勾选项）、四个开关：部件左右分割 `tblr_split`、LaMa 修补 `use_lama`、图层修补 `layer_fix`、导出 PSD `save_psd`；高级项里有：GGUF 量化流式、深度步数、深度分辨率、前景分割分辨率、注意力后端（native / xformers）、骨骼切割、骨骼置信度、关节重叠、保留切图与骨架清单 |
| **03 模型选择** | 分别选「图层模型」「深度模型」（下拉显示 `models/` 下的相对路径，同名的不同量化档不会混）；「模型常驻」默认开，下一轮省 40–75 秒加载；「分组卸载」默认关（更省显存、更慢）；模型换了点「重新扫描」 |
| **04 运行** | 点开始；下方常驻日志控制台（时间 + 级别 + 内容），以及本次分辨率、步数、实际种子、耗时 |
| **成像台** | 真实输出缩略图（点开看原图）+ PSD 行（「打开文件夹」直接打开该任务的 `out/`）+ 清单条（`cuts/manifest.json`、`pose_keypoints.json` 可直接下载）+ 关节置信度条；右上角「后处理重跑」（只重跑后处理，不重采样） |
| **设置页** | 顶栏「设置」打开 `/settings`（中英双语，跟随工作台 CN/EN 实时切换）：环境信息（含组件缓存路径与体积、复制诊断）、启动与端口（改 host / port 写回 `config.json`，重启生效）、维护与诊断（完整性检查、打开 outputs / models / 项目根、打开命令提示符） |

其它交互：左栏与成像台之间的竖条可左右拖，成像台与日志之间的横条可上下拖，双击复位（宽度记在 `localStorage`）；右上角中/EN 切换即时生效。

**参数怎么选（6 GB 显存的实用档）**：

| 分辨率 | 档位 | 实测步速 | 30 步大致耗时 |
|---|---|---|---|
| 720–768 | `resident`（权重全在显存） | 约 37 s/步 | 约 40 分钟 |
| 896–1024 | `streamed`（逐层 PCIe 搬） | 约 300 s/步 | 4–5 小时 |
| 1280+ | `low-vram` | 更慢，不建议 | — |

建议 **768 + 20~24 步**；要 1024 以上请留足 4 小时。显存更宽裕的卡可以往上走。

**输出**：`outputs\jobs\<job_id>\` 下有图层 PNG（通常 26 层左右）、`cuts\` 逐层切片与 `cuts\manifest.json`（轴心/骨骼/父子）、深度图、以及 `*.psd`；`meta.json` 记录参数与产物清单，所以服务重启后下载链接仍然有效。

---

## 8. 项目结构

```
SeeThrough-GGUF-Win/
├── 1-setup.bat                  # 一键装环境（5 步，见 §4.1）
├── 2-launch.bat                 # 启动服务并开浏览器 → http://localhost:7860
├── 3-env.bat                    # 打开已激活 .venv 的命令提示符（设置页按钮也调它）
├── 4-install_triton.bat         # 可选：装/卸 triton-windows（见 docs/triton-windows.md）
├── requirements-torch.txt     # torch 2.8.0+cu128（专用索引，含架构说明）
├── requirements.txt           # STM 依赖 + Flask（PyPI）
├── config.json                # 端口、路径、默认参数、约束范围
├── LICENSE / NOTICE           # Apache-2.0（整个仓库）
├── MODIFICATIONS.md           # 对上游代码的逐行修改声明（Apache-2.0 §4b）
├── THIRD_PARTY_NOTICES.md     # 第三方组件与模型许可清单
├── backend/
│   ├── server.py              # Flask：上传 / 校验 / 子进程 / SSE / 下载 / 取消
│   ├── worker.py              # 常驻推理进程（复用已加载模型，JSON stdin/stdout 协议）
│   ├── components_paths.py    # 组件缓存路径（HF_HOME / TORCH_HOME → components/）
│   ├── selfcheck.py           # 环境自检（依赖 / CUDA / 架构 / 补丁，不查模型）
│   ├── netcheck.py            # 下载源探测（官方 vs 镜像，官方 600ms 偏置）
│   ├── download_models.py     # 下载两个 GGUF 与辅助组件（失败回退 curl 续传）
│   ├── install_triton.py      # Triton 向导（版本推导 / 测速选源 / 冒烟 / 同步配置）
│   ├── tls.py                 # 合并系统根证书（应对杀软 HTTPS 中间人）
│   ├── test_pipeline.py       # 后端链路集成测试
│   ├── requirements.txt       # 后端单独依赖（一般不用管）
│   └── patches/                # apply_use_lama.py / apply_layer_fix.py（幂等，setup 自动跑）
├── frontend/
│   ├── index.html / styles.css / app.js   # 原生 JS + GSAP，无构建步骤
│   └── vendor/                # GSAP 3.12.5（本地，不走 CDN）
├── docs/
│   ├── ai-setup.md            # 本文（部署任务书）
│   ├── triton-windows.md      # Triton 安装 / 手动安装 / 卸载 / 排错
│   ├── triton-windows.en.md   # 上一份的英文版
│   ├── triton-windows-ai.md   # Triton 操作任务书（给 AI）
│   └── screenshots/           # 界面截图（workbench / settings，中英各一套）
├── stm/                       # 上游代码快照（Apache-2.0，已打 use_lama 与图层修补补丁）
├── models/                    # GGUF 权重（不入库，自己放）
├── components/                # 首次推理的辅助组件（HF / torch.hub，不入库，自动创建）
└── outputs/jobs/<id>/         # 每轮任务的输入、输出、meta.json（不入库）
```

**你可以改的**：`config.json`（端口、路径、默认值）。**不建议改的**：`stm/`、`frontend/`、`backend/` 的逻辑——报错先查，不要就地打补丁。

运行时还会产生：`outputs/.cache/ca-bundle.pem`（合并后的根证书）、`outputs/.cache/` 下其它下载缓存；首次推理要用的辅助组件缓存在**项目内 `components/`**（`1-setup.bat` 与服务都会把 `HF_HOME` / `TORCH_HOME` 指过去），不入库，删掉会自动重下。

---

## 9. 运行时行为（排查时用得上）

- **常驻 worker**：`keep_model=true`（默认）时，`backend/worker.py` 作为常驻子进程保留已加载的模型，下一轮免加载（实测 448 分辨率下首轮 224 s → 次轮 172 s，到首步 50.9 s → 13.5 s）。关掉它则每轮冷启动并立即释放显存。
- **串行**：同一时间只允许一个任务；重复提交 409。
- **进度**：`/api/stream/<job_id>`（SSE，带 seq）+ 每 1.5 s 轮询 `/api/status/<job_id>` 兜底（安全软件会缓冲 SSE）。日志形如 `进度 12/30 · 40% · 37.2s/it`。
- **刷新接回**：`/api/current` 拿正在跑的任务，`/api/last` 回看 24 小时内上次结果。
- **取消**：`POST /api/jobs/<id>/cancel` 终止进程树，状态置 `canceled`，临时文件留在该任务目录。
- **删除**：`DELETE /api/jobs/<id>` 删整个任务目录（输入图 + 输出）。
- **重启服务**后 `outputs/jobs/` 里的下载链接依然有效（靠 `meta.json`）。

---

## 10. 接口速查

| 方法 | 路径 | 用途 |
|---|---|---|
| GET | `/api/config` | 默认值、约束、支持的扩展名、上传上限 |
| GET | `/api/check_models` | 默认 GGUF、STM 仓库、LaMa 补丁、网络状态 |
| GET | `/api/models` | 递归扫描 `models/` 的 `.gguf` 列表 |
| POST | `/api/run` | 提交任务（`multipart`：`image` + `params` JSON） |
| GET | `/api/stream/<job_id>` | SSE：`event: log` + `event: done` |
| GET | `/api/status/<job_id>` | 轮询兜底，含末 200 行日志 |
| GET | `/api/current` / `/api/last` | 当前任务 / 上次结果 |
| POST | `/api/jobs/<job_id>/cancel` | 取消 |
| DELETE | `/api/jobs/<job_id>` | 删除任务目录 |
| GET | `/api/file/<job_id>/<rel>` | 取输出文件，`?download=1` 强制下载 |
| GET | `/api/jobs/<job_id>/pose` | 读 `pose_keypoints.json`，返回置信度最低的关节与骨骼门限 |
| POST | `/api/jobs/<job_id>/reassemble` | 后处理重跑（只重做 PSD / 分割 / 骨骼层） |
| POST | `/api/jobs/<job_id>/reveal` | 打开该任务的 `out/` 目录 |
| GET | `/api/env` | 设置页用的只读环境快照（含组件缓存位置与体积） |
| GET | `/settings` | 设置页（环境信息 / 启动与端口 / 维护与诊断） |
| POST | `/api/terminal` | 本机弹出已激活 `.venv` 的命令提示符（仅本机） |
| POST | `/api/config-file` | 把 `host` / `port` 写回 `config.json`（备份为 `.bak`，重启生效；仅本机） |
| POST | `/api/open-folder` | 打开项目内目录：`root` / `outputs` / `models` / `stm`（仅本机） |

`params` 关键字段：`width` / `height`（256–2048，64 倍数，必须相等）、`inference_steps`（1–150，默认 30）、`inference_steps_depth`（默认 4）、`resolution_depth`（`"auto"` / -1 / 64 倍数）、`mask_resolution`（`"auto"` / 256–4096）、`attention_backend`（`native` / `xformers`）、`seed_mode`（`fixed` / `random`）、`seed`、`tblr_split`、`use_lama`、`save_psd`、`ragdoll_split`、`ragdoll_confidence`、`ragdoll_overlap`、`parts`（数组，`null` = 全量）、`save_cut_images`、`layer_fix`、`group_offload`、`quantized`（**默认必须 true**）、`keep_model`、`layerdiff_model` / `depth_model`（`models/` 下相对路径）。

> `quantized` 不是可选项：STM 里 `if not quantized: stream = False`，不传它会关掉流式加载、整份权重挤进显存——6 GB 卡就是这样爆的。

---

## 11. 常见故障

| 现象 | 原因 | 处理 |
|---|---|---|
| `no kernel image is available for execution on the device` | PyTorch 轮子不含本机架构（Pascal = sm_61，torch 2.9+ 的 cu128 已不含） | 重装 `requirements-torch.txt`（torch 2.8.0+cu128）；`backend\selfcheck.py` 会打印 `编译架构` 列表，对不上就会警告 |
| 首次推理卡在下载 | 辅助组件没预取全，仍在访问 Hugging Face | 服务启动会探测并写入 `HF_ENDPOINT`；国内可 `config.json` 设 `hf_endpoint` 为 `https://hf-mirror.com` |
| `CERTIFICATE_VERIFY_FAILED: self-signed certificate` | 安全软件 / 企业代理做 HTTPS 中间人 | 程序会自动合并系统根证书到 `outputs/.cache/ca-bundle.pem`；下载脚本还有 `curl` 回退 |
| 显存不足 / 被系统内存拖死 | 关了量化流式，或分辨率过高 | 确认高级项「GGUF 量化 + 流式加载」为开；降到 720 / 步数 20 |
| 一步几分钟、GPU 100% 但功耗低 | 掉进 `streamed` 档，权重每层经 PCIe 搬运 | 降到 768 切回 `resident` |
| 任务结束后显存仍被占用 | 「模型常驻」保留着模型 | 03 里关掉常驻再跑一轮；取消任务也会结束常驻进程 |
| 日志长时间不刷新但任务在跑 | 安全软件缓冲 SSE | 页面每 1.5 s 轮询兜底，属正常；卡住不动才是真卡住 |
| `Compiler: cl is not found` | `torch.compile` 反量化要 MSVC，而 Windows 默认没有 | 项目默认 `compile_dequant: false`，一般无需处理；要试加速看 [triton-windows.md](triton-windows.md) |
| 端口被占用 | 7860 被别的程序占了 | 改 `config.json` 的 `port` |
| 页面顶部红条 + 按钮禁用 | 模型缺失 | 按 §5 放好两个 GGUF，或点「重新扫描」 |

---

## 12. 汇报模板

部署完按这个结构回话，别贴整段安装日志：

```text
【环境】Windows 11 · Python 3.12.x（uv / 官网）· <显卡名> <显存>GB · 驱动 CUDA 12.x · 磁盘剩余 xx GB
【位置】<项目目录>（git clone / ZIP）
【安装】1-setup.bat 五步：1 通过 · 2 通过 · 3 通过（torch 2.8.0+cu128）· 4 补丁已打 · 5 组件 xx/xx + 自检通过
【模型】layerdiff3d-xxx.gguf（x.x GB）· marigold3d-xxx.gguf（x.x GB）
【验收】/api/config ✔ · /api/check_models ✔（补丁已打）· /api/models 2 项 ✔ · 页面可开 ✔ · 512×512 / 6 步冒烟 完成，耗时 xx 秒，输出 N 层 + PSD
【用法】双击 2-launch.bat → http://localhost:7860 → 01 选图 / 02 参数（建议 768 + 20~24 步）/ 03 选模型 + 模型常驻 / 04 运行
【遗留】例如「组件未预取全，首次推理会再下」「显存 6 GB，1024 会很慢」
```

**中途停下也要汇报**：卡在哪一步、原始报错、你判断的原因、需要用户做什么（装 Python / 清磁盘 / 提供模型文件）。不要静默失败。

---

## 13. 相关文档

- 安装 / 参数 / 接口 / 架构说明的完整版：仓库根目录 `README.md`
- Triton 加速（可选，只有 CC ≥ 7.0 的卡有意义）：[triton-windows.md](triton-windows.md)
- 让 AI 代装 Triton：[triton-windows-ai.md](triton-windows-ai.md)
- 对上游代码改了什么：[MODIFICATIONS.md](../MODIFICATIONS.md)
