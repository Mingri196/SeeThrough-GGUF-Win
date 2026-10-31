# SeeThrough-GGUF-Win

[English](README.en.md) · **简体中文**

把 [IceFog72/see-through-gguf](https://github.com/IceFog72/see-through-gguf)（简称 **STM**：纯 Python + PyTorch 的「单张角色图 → 分层 PSD / 深度 / 骨骼切割」运行时）包装成 **Windows 本地网页工具**。上游代码已作为快照内置在 `stm/`，**一次 clone 即可安装，不需要额外克隆仓库**。

---

## 1. 快速开始

先克隆仓库（没有 Git 也可以直接下载 ZIP 解压）：

```bat
git clone https://github.com/Mingri196/SeeThrough-GGUF-Win.git
cd SeeThrough-GGUF-Win
```

然后双击安装、启动：

```bat
1-setup.bat      :: 建 .venv、装依赖、校验补丁、探测网络并预取辅助组件、自检
2-launch.bat     :: → http://localhost:7860
```

只需要 **Python 3.12（或 uv）**，不需要 Git。`1-setup.bat` 会自动：

1. 校验内置的 `stm/`；
2. 创建 `.venv` 并安装 PyTorch cu128 + 依赖（约 8 GB）；
3. 校验 `use_lama` 补丁（幂等，已打过会跳过）；
4. 探测 `huggingface.co` 与 `hf-mirror.com`，**国内通常自动选镜像、海外选官方**，预取首次推理要用的辅助组件（约 2.4 GB，缓存到项目内 `components/`）；**不下载主模型**；
5. 自检并打印结果。

装完还需要把两个 GGUF 放进 `models/`（见第 2 节），页面「模型选择」里能选到就能跑。想跳过组件下载：先 `set STG_SKIP_MODELS=1` 再运行 `1-setup.bat`；想连主模型一起下：`.venv\Scripts\python.exe backend\download_models.py`。

手工等价命令（在项目根目录）：

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

没有 `uv` 时用 `py -3.12 -m venv .venv` + `.venv\Scripts\python -m pip install -r requirements.txt`。

**环境要求**：Windows 10/11 64 位 · NVIDIA 驱动支持 CUDA 12.8 · Python **3.12** · 磁盘约 26 GB（环境约 8 GB + 组件约 2.4 GB + 自备主模型约 5.4 GB + 余量）· 安装时联网。辅助组件预取完成后，推理阶段不再需要联网。

首次推理要用的辅助组件（Marigold 文本编码器、动漫分割、LaMa、RTMPose、anime-seg）会下载到**项目内 `components/`**，不再散落在 `C://Users//<你>//.cache`：删掉项目目录就等于卸载干净，换盘也只是拖一个文件夹。想改用系统默认位置（多工具共享一份缓存），把 `config.json` 的 `components_dir` 改成 `"user"`。

> **GPU 架构注意**：GTX 10 系是 Pascal（sm_61）。torch 2.9 起的 cu128 轮子只编译到 sm_75 及以上，装上能 `torch.cuda.is_available()` 为真，但一跑算子就报 `no kernel image is available for execution on the device`。因此 `requirements-torch.txt` 固定在 **torch 2.8.0+cu128**（编译架构含 sm_61/70/75/80/86/90/100/120），这也是最后支持 GTX 10 系的版本。16/20/30/40 系可以自行升到更高版本。

### 1.1 交给 AI 从零部署

不想看命令行：把 **[docs/ai-setup.md](docs/ai-setup.md)** 整个交给 AI（拖文件，或说「按 `docs/ai-setup.md` 把这个项目部署起来」），它就会带着你走完：

- **体检**：Python / uv、磁盘（约 26 GB）、显卡与驱动，缺什么先说要装什么；
- **克隆**：`git clone`（没有 Git 走 ZIP），并核对 `stm/`、`backend/`、`frontend/` 是否完整；
- **装环境**：跑 `1-setup.bat` 的 5 步并逐步解释，附不能交互时的等价命令；
- **备模型**：`download_models.py` 或手动放入 `models/`；
- **启动验收**：起服务 → 打 4 个接口 → 用 512×512 / 6 步冒烟出一张图；
- **讲用法**：四步操作流、参数怎么选、`outputs/jobs/<id>/` 里有什么；
- **讲结构**：每个目录 / 脚本干什么，哪些能改哪些别碰；
- **报结果**：统一汇报模板，卡住也会说清卡在哪、要你做什么。

那份文档还写明了 AI 的红线（不动系统 Python、不改 `stm/` 源码、不替你装驱动 / VS、不提交 git），以及常见故障对照表。适合「只想用，不想折腾」的情况。

---

## 2. 模型下载

**主模型需要自己准备**，`1-setup.bat` 只预取辅助组件。下载脚本会按下面的规则自动选源：

1. 并行探测两个源的模型 API；
2. 都可达时官方有 600 ms 优先偏置（海外用户通常走官方）；
3. 官方不可达或明显更慢时走 `hf-mirror.com`（国内常见）；
4. 结果会打印在安装日志、服务启动日志，以及页面「模型详情」里的「下载源」一行。

| 模型                | 仓库                                           | 默认路径                           |
| ----------------- | -------------------------------------------- | ------------------------------ |
| LayerDiff3D（图层分解） | `icefog72/seethroughv0.0.2_layerdiff3d_gguf` | `models/layerdiff3d-Q5_1.gguf` |
| Marigold3D（深度）    | `icefog72/seethroughv0.0.1_marigold_gguf`    | `models/marigold3d-Q8_0.gguf`  |

把下载好的两个 GGUF 放进 `models/`（可以建子目录，例如 `models/layerdiff3d/`、`models/marigold3d/`），页面「模型选择」会递归扫描并按完整路径列出。或者直接让脚本代下：

```bat
:: 下载两个主模型 + 辅助组件
.venv\Scripts\python.exe backend\download_models.py --with-components

:: 只下主模型
.venv\Scripts\python.exe backend\download_models.py

:: 只看两边通不通、远端多大，不下载
.venv\Scripts\python.exe backend\download_models.py --check-only

:: 强制指定下载源
.venv\Scripts\python.exe backend\download_models.py --endpoint hf-mirror
.venv\Scripts\python.exe backend\download_models.py --endpoint huggingface
```

`config.json` 的 `hf_endpoint` 默认 `"auto"`；也可写死 `https://hf-mirror.com` 或 `https://huggingface.co`，运行期 STM 拉取辅助组件时会用它。文件名不同就改 `layerdiff_gguf` / `depth_gguf`，下载脚本会按这两个路径落盘。

> **TLS 说明**：部分安全软件（Kaspersky 等）会做 HTTPS 中间人，导致 Python 报 `CERTIFICATE_VERIFY_FAILED`。安装与服务启动时会自动把「certifi + Windows 系统根证书」合并成 `outputs/.cache/ca-bundle.pem` 供 requests / huggingface_hub 使用；若仍失败，下载脚本会回退到 `curl`（走系统证书、支持断点续传）。

---

## 3. 界面与操作流

| 步骤          | 内容                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| ----------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **01 图像**   | 拖拽或点选图片，支持 **PNG / JPG / WEBP / BMP**。立即校验扩展名、体积、可解码性，显示缩略图、文件名、体积与像素尺寸；可随时移除或换图，换图会清空上一张的结果与任务痕迹                                                                                                                                                                                                                                                                                                                                                                |
| **02 参数**   | 宽/高（同步，64 的倍数）、推理步数、种子模式、**输出图层**（24 个 tag 按身体组/头部组勾选，默认全选）、`tblr_split` / `use_lama` / `layer_fix` / `save_to_psd` 开关、高级项（量化流式、深度步数、深度分辨率、前景分割分辨率、注意力后端、骨骼切割、骨骼置信度、关节重叠、保留切图与骨架清单）                                                                                                                                                                                                                                                                              |
| **03 模型选择** | 递归扫描 `models/`（含子目录），分别选「图层模型」和「深度模型」；下拉框显示相对路径（如 `marigold3d/marigold3d-Q5_1.gguf`），同名的不同量化档不会混淆；换档位不用改 `config.json`。可点「重新扫描」刷新。「模型常驻」（默认开）让模型留在显存里，下一轮省掉约 40–75 秒加载；「分组卸载」（默认关）按组在 CPU / GPU 间搬权重，更省显存但更慢                                                                                                                                                                                                                                                     |
| **04 运行**   | 开始 / 取消；下方固定显示本次分辨率、步数、实际种子、耗时                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| **成像台**     | 右上角「后处理重跑」（浮层：左右分割 / 骨骼切割 / 置信度 / 关节重叠 / PSD / 切图，只重跑后处理不重采样）+ 真实输出缩略图（点击看原图）+ PSD 行（「打开文件夹」直接打开该任务的 `out/`）+ 清单条（`cuts/manifest.json`、`pose_keypoints.json` 等 JSON 可直接下载）+ **关节置信度**条（骨骼切割后自动列出低于门限的关节）；下方是常驻日志控制台                                                                                                                                                                                                                                              |
| **设置页**     | 顶栏「设置」打开新标签页 `/settings`，左侧固定侧栏三页（整页中英双语，跟随工作台的 CN/EN **实时切换**）：**环境信息**（Python / torch / CUDA / 显卡与显存 / 驱动 / 系统 / CPU / 内存 / 下载源与握手耗时 / 路径 / **组件缓存位置与体积** / 模型清单 / 补丁与开关 / 磁盘，右上角「复制诊断」）、**启动与端口**（监听地址与本机/局域网判定、配置端口、**更改地址 / 更改端口**并写回 config.json（自动备份 `config.json.bak`，重启后生效）、PID 与运行时长、config.json 路径与最后修改时间）、**维护与诊断**（服务状态、.venv / STM 代码 / 两个主模型 / 三个补丁的完整性检查、任务数与占用，打开 outputs / models / 项目根，以及右上角「打开命令提示符」——调 `3-env.bat` 弹一个已激活 `.venv` 的 cmd 窗口，仅本机可用）。全部只读，不改配置 |

**可拖拽布局**：左侧参数栏与成像台之间的竖分隔条可左右拖动，成像台与日志之间的横分隔条可上下拖动；双击复位，也支持键盘方向键（Shift 加速）。宽度/高度会记在 `localStorage`。

- 语言切换（CN/EN）即时生效并记忆；所有状态、错误、字段文案、日志级别均有双语。
- 参数错误在字段下方就地提示，同时后端做防御性校验并返回字段级错误码。
- 数值输入非法时按钮直接禁用，不会发出请求。

---

## 4. 参数与接口契约

`POST /api/run`（`multipart/form-data`：`image` + `params` JSON）

| 字段                      | 类型                    | 范围 / 默认                 | 说明                                                                                                                                                                                  |
| ----------------------- | --------------------- | ----------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `width` / `height`      | int                   | 256–2048，64 的倍数，默认 1024 | 引擎按正方形推理，两者必须相等；后端校验 `must_equal_width`                                                                                                                                             |
| `inference_steps`       | int                   | 1–150，默认 30             | LayerDiff 采样步数                                                                                                                                                                      |
| `inference_steps_depth` | int                   | -1 或 1–100，默认 **4**     | Marigold 深度步数；-1 = 用模型内置默认（也是 4）                                                                                                                                                    |
| `resolution_depth`      | `"auto"` / -1 / int   | 默认 **`"auto"`**         | 深度推理尺寸：`auto` 按显存上限（6 GB 卡 768）与主分辨率取小；`-1` 跟随主分辨率；正整数自定义（64 的倍数）                                                                                                                   |
| `mask_resolution`       | `"auto"` / int        | 默认 **`"auto"`**         | 前景分割分辨率，**只对没有透明通道的图生效**（有 alpha 时直接跳过分割）。`auto` 取图片长边并夹在 640–显存上限之间；正整数（256–4096）强制指定，调大边缘更细、更吃显存                                                                                  |
| `attention_backend`     | `native` / `xformers` | 默认 **`native`**         | 注意力实现，界面高级项里是两颗按钮。`native` = PyTorch 自带 SDPA，零依赖；`xformers` 需自行安装：`uv pip install --python .venv\Scripts\python.exe --no-deps xformers==0.0.32.post1`（对应 torch 2.8.0；装完重启，按钮才会解除置灰） |
| `seed_mode`             | str                   | `fixed` / `random`      | `random` 由后端生成并保证与上次不同                                                                                                                                                              |
| `seed`                  | int                   | 0–2147483647，默认 42      | 仅 `fixed` 模式生效                                                                                                                                                                      |
| `tblr_split`            | bool                  | 默认 `true`               | 部件左右分割                                                                                                                                                                              |
| `use_lama`              | bool                  | 默认 `true`               | 关闭需 STM 补丁支持（`--no-use_lama`）                                                                                                                                                       |
| `save_psd`              | bool                  | 默认 `true`               | 生成分层 PSD                                                                                                                                                                            |
| `ragdoll_split`         | bool                  | 默认 `false`              | RTMPose 骨骼分层                                                                                                                                                                        |
| `ragdoll_confidence`    | float                 | 0.05–0.95，默认 0.3        | 遮挡严重时调低                                                                                                                                                                             |
| `ragdoll_overlap`       | float                 | 0–0.5，默认 0.2            | 骨骼图层在关节处的重叠比例（按较短的那根骨骼算）。调大关节处更不易撕裂、重复像素更多；调小更省图层体积                                                                                                                                 |
| `parts`                 | str[] \| null         | 默认 `null`（全量 24 层）      | 只生成指定图层；别名 `hair` → 前后发、`eyes` → 虹膜/眉/眼白/睫毛。只影响落盘，身体组始终整组采样；头部组一个都不选则整趟跳过                                                                                                           |
| `save_cut_images`       | bool                  | 默认 **`true`**           | 额外输出 `cuts/` 逐层 RGBA 切片与 `cuts/manifest.json`（轴心、骨骼、父子关系、关节重叠）                                                                                                                      |
| `group_offload`         | bool                  | 默认 `false`              | 按组在 CPU / GPU 间搬权重，更省显存但更慢；与流式加载作用重叠                                                                                                                                                |
| `layer_fix`             | bool                  | 默认 **`true`**           | 图层修补（后处理补丁，不重画像素）：① 贴画布下沿的层（典型是鞋子）不再被「空层判定」误杀；② 眼睛按 眼白→虹膜→睫毛 固定叠放，避免眼白发白                                                                                                            |
| `quantized`             | bool                  | 默认 **`true`**           | 打开 GGUF 量化 + 流式加载；关闭后整份权重进显存                                                                                                                                                        |
| `layerdiff_model`       | str                   | `models/` 下的相对路径，可选     | 图层模型；缺省时用 `config.json` 的 `layerdiff_gguf`                                                                                                                                          |
| `keep_model`            | bool                  | 默认 **`true`**           | 模型常驻：任务结束后保留模型在显存，下一轮免加载；关闭则每轮重新加载并立即释放显存                                                                                                                                           |
| `depth_model`           | str                   | `models/` 下的相对路径，可选     | 深度模型；缺省时用 `config.json` 的 `depth_gguf`                                                                                                                                              |

返回：`{ok, job_id, seed_used, params}`。随机模式下 `seed_used` 会立即回填到界面。

其它接口：

| 方法     | 路径                              | 说明                                                                                 |
| ------ | ------------------------------- | ---------------------------------------------------------------------------------- |
| GET    | `/api/config`                   | 默认值、约束范围、支持扩展名、上传上限                                                                |
| GET    | `/api/check_models`             | 默认 GGUF、STM 仓库、LaMa 补丁与网络状态                                                        |
| GET    | `/api/models`                   | 递归扫描 `models/` 的 `.gguf` 列表（含角色推断与默认项）                                             |
| GET    | `/api/stream/<job_id>`          | SSE：`event: log`（`{ts, level, msg}`）+ `event: done`（状态、输出、耗时）                      |
| GET    | `/api/status/<job_id>`          | 轮询兜底，含末 200 行日志                                                                    |
| POST   | `/api/jobs/<job_id>/cancel`     | 终止进程树，状态置 `canceled`                                                               |
| GET    | `/api/jobs/<job_id>/pose`       | 读 `pose_keypoints.json`，返回置信度最低的几个关节与骨骼门限                                          |
| GET    | `/api/env`                      | 设置页用的只读环境快照：运行时、硬件、配置、进程、配置文件与完整性检查（不导入 torch，不占显存）                                |
| POST   | `/api/terminal`                 | 在本机弹出已激活 `.venv` 的命令提示符（内部执行 `3-env.bat`）；仅接受本机调用                                  |
| POST   | `/api/config-file`              | 把 `host` / `port` 写回 `config.json`（自动备份为 `config.json.bak`），只写文件不改监听，重启后生效；仅接受本机调用 |
| POST   | `/api/open-folder`              | 在资源管理器打开项目内目录，`target` 取 `root` / `outputs` / `models` / `stm`；仅接受本机调用             |
| GET    | `/settings`                     | 设置页（环境信息 / 启动与端口 / 维护与诊断，只读）                                                       |
| POST   | `/api/jobs/<job_id>/reassemble` | 后处理重跑：只对已完成的 `out/<name>/` 重做 PSD / 分割 / 骨骼层，不重跑采样                                 |
| GET    | `/api/jobs/<job_id>/reassemble` | 查询后处理重跑状态与日志尾巴                                                                     |
| POST   | `/api/jobs/<job_id>/reveal`     | 在资源管理器里打开该任务的 `out/` 目录                                                            |
| DELETE | `/api/jobs/<job_id>`            | 删除该任务目录（输入图与输出）                                                                    |
| GET    | `/api/file/<job_id>/<rel>`      | 输出文件；`?download=1` 强制附件下载                                                          |

任务元数据写入 `outputs/jobs/<id>/meta.json`，服务重启后下载链接依然有效。同一时间只允许一个任务（单卡并行只会互相抢显存），重复提交返回 `409 busy`。

---

## 5. 两个关键实现细节

### 5.1 `--quantized` 必须显式传

STM 的 `common/utils/inference_utils.py` 里有：

```python
if not quantized:
    stream = False
```

也就是说，不传 `--quantized` 时 GGUF 会被整体反量化后加载，**流式加载被静默关闭**——6 GB 卡上正是这样爆显存的。本项目默认传入该参数，并在界面高级项里暴露开关。

### 5.2 `use_lama` 补丁

STM 的后处理默认总是用 LaMa 修补（`cluster_inpaint_part(..., inpaint='lama')`），但没有暴露命令行开关。仓库内置的 `stm/` 已打好补丁；`backend/patches/apply_use_lama.py` 保留下来，用于把这套改动重新施加到更新后的上游版本（幂等，已打过会直接跳过）。补丁新增 `--use_lama / --no-use_lama`：

- 默认开启，与原生行为一致；
- 未打补丁时界面把该开关锁为「开启」并在模型详情里标注未打补丁，后端也会拒绝「关闭 LaMa」的请求（`patch_missing`），不会假装生效；
- 该开关只影响 `further_extr` 的 hair 分支（本推理链路唯一的一处修补调用）。

改动只涉及两个上游文件：`common/utils/inference_utils.py`（新增 `use_lama` 形参、hair 分支按开关选择 `lama` / `cv2`）与 `inference/scripts/inference_psd.py`（新增 `--use_lama / --no-use_lama`、向下传递）。逐行清单见 [MODIFICATIONS.md](MODIFICATIONS.md)。

### 5.3 图层修补（`layer_fix`）

上游后处理有两个会让产物「看起来缺东西」的缺陷，本项目打补丁修掉，界面 02 里可用「图层修补」开关按任务关闭（`STG_LAYER_FIX=0` 完全回退成上游行为）：

| 症状          | 原因                                                                                              | 修法                                 |
| ----------- | ----------------------------------------------------------------------------------------------- | ---------------------------------- |
| 鞋子整层消失、脚部露底 | `load_part()` 判空层时先裁掉底部/右侧 10% 条带再数 alpha 像素；贴着画布下沿的层（鞋子 bbox 常正好到 768 底边）被算成 0 → 整层丢弃          | 改为统计整张 mask                        |
| 一只眼睛发白、没有瞳孔 | `dump_parts_psd()` 按 `depth_median` 堆层，眼睛三层深度极接近（实测 eyewhite 0.341 / irides 0.349），噪声会让眼白压在虹膜之上 | 把三层收拢后按 眼白 → 虹膜 → 睫毛 固定顺序排列，整体位置不变 |

它**只重排/保留已有像素，不重画**：实测同一张 768 输出重跑后处理，层数 22 → 23（footwear 回归），相对原图的空白像素 4728 → 991。补丁脚本 `backend/patches/apply_layer_fix.py`（幂等，`1-setup.bat` 会自动跑）。

---

## 6. 状态与异常

| 场景          | 表现                                                    |
| ----------- | ----------------------------------------------------- |
| 模型缺失        | 启动时预检；顶部红条列出缺失项，按钮禁用，不发起请求                            |
| 格式错误 / 损坏文件 | 上传阶段拦截并说明原因（扩展名、体积、解码失败分列）                            |
| 参数非法        | 前端就地提示 + 后端 400 字段级错误；不会进入推理                          |
| 推理进程失败      | 状态标签「失败」，界面恢复可操作，日志保留完整 traceback，`error_detail` 附退出码 |
| PSD 导出失败    | 状态「完成 · 有告警」，区别于推理失败；图层 PNG 仍可下载                      |
| 取消          | 终止进程树，状态「已取消」，临时文件保留在该任务目录，可手动删除                      |
| 服务不可达       | 提示检查 2-launch.bat，不会白屏或残留 loading 状态                  |

---

## 7. 排障

| 现象                                                                        | 原因与处理                                                                                                                                                                                                                                                                                                             |
| ------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `no kernel image is available for execution on the device`                | PyTorch 轮子不包含你的显卡架构。按 `requirements-torch.txt` 装 **torch 2.8.0+cu128**；跑 `backend/selfcheck.py` 可看到 `torch.cuda.get_arch_list()`                                                                                                                                                                                  |
| 推理日志里出现 `Compiler: cl is not found`                                       | STM 默认用 `torch.compile` 编译反量化核，Windows 需要 MSVC。`config.json` 的 `compile_dequant` 已默认 `false`（设 `SEETHROUGH_COMPILE_DEQUANT=0`），一般无需处理                                                                                                                                                                             |
| 模型下载慢或失败                                                                  | `backend\download_models.py --check-only` 先看两个源的状态；用 `--endpoint hf-mirror` / `--endpoint huggingface` 强制指定；`config.json` 的 `hf_endpoint` 默认 `auto`，运行期组件下载也用它                                                                                                                                                    |
| 首次推理卡在下载辅助组件                                                              | 仍需访问 Hugging Face；服务启动时会探测并写入 `HF_ENDPOINT`，国内走镜像                                                                                                                                                                                                                                                                 |
| 下载卡在几十 MB 好几不动、日志里有 `xet` / `cas-bridge.xethub.hf.co` 超时 | HF 的 xet 传输会直连海外节点，国内常读超时 | 现在走镜像时会自动设置 `HF_HUB_DISABLE_XET=1`（退回普通 HTTP）；旧版本可手动 `set HF_HUB_DISABLE_XET=1` 后重跑 |
| `CERTIFICATE_VERIFY_FAILED: self-signed certificate in certificate chain` | 安全软件/企业代理做了 HTTPS 中间人。程序会自动合并系统根证书到 `outputs/.cache/ca-bundle.pem`；下载脚本还有 curl 回退（走系统证书、支持续传）                                                                                                                                                                                                                     |
| 显存不足 / 被系统内存拖死                                                            | 确认高级项里「GGUF 量化 + 流式加载」为开（默认），并降低分辨率到 720 / 步数降到 20 左右                                                                                                                                                                                                                                                             |
| 任务结束后显存仍被占用                                                               | 「模型常驻」（03 模型选择里，默认开）保留了模型。关掉它再跑一轮即可释放；取消任务也会顺带结束常驻进程                                                                                                                                                                                                                                                              |
| 日志长时间不刷新但任务在跑                                                             | 部分安全软件会缓冲 SSE；页面每 1.5 秒轮询 `/api/status` 兜底，属正常。采样期间日志会显示 `进度 12/30 · 40% · 37.2s/it`，卡住不动才是真卡住                                                                                                                                                                                                                    |
| 一步要好几分钟、GPU 100% 但功耗只有 40–50 W                                            | 分辨率把显存档位挤进了 `streamed`：权重每层都从内存经 PCIe 搬一次，GPU 大半时间在等数据。降分辨率到 768 即可切回 `resident`，6 GB 显存设备实测 **1024 约 300 s/步 → 768 约 37 s/步**                                                                                                                                                                                    |
| 想试 `torch.compile` 的反量化加速（只有 CC ≥ 7.0 的卡有意义）                              | 双击 `4-install_triton.bat`：它会检测环境 → 推导匹配的 Triton 版本 → 测速选源 → 安装 → 用**真实反量化 kernel** 做冒烟测试，通过才把 `compile_dequant` 置为 `true`。再运行一次可保留/卸载/重装。CC 低于 7.0 的卡会被明确提示「不达标」，装完也不会启用。命令与排错见 [docs/triton-windows.md](docs/triton-windows.md)；不想自己动手就把 [docs/triton-windows-ai.md](docs/triton-windows-ai.md) 整份丢给 AI，让它按任务书代劳 |
| 端口被占用                                                                     | 改 `config.json` 的 `port`，或关掉占用 7860 的进程                                                                                                                                                                                                                                                                           |

### 可选：安装 xformers（注意力后端）

默认走 PyTorch 自带的 SDPA（`attention_backend: native`），不装任何额外依赖。想让高级项里的 `xformers` 按钮可用，三步：

```bat
:: 1) 确认 torch 是 2.8.0+cu128
.venv\Scripts\python.exe -c "import torch;print(torch.__version__, torch.version.cuda)"

:: 2) 用 uv 装对应版本（这个 venv 里没有 pip，python -m pip 会报 No module named pip）
uv pip install --python .venv\Scripts\python.exe --no-deps xformers==0.0.32.post1

:: 3) 冒烟：不只看 import，真跑一次 attention
.venv\Scripts\python.exe -c "import torch,xformers,xformers.ops as xo;q=torch.randn(1,64,8,32,device='cuda',dtype=torch.float16);print(xformers.__version__, xo.memory_efficient_attention(q,q,q).shape)"
```

版本对应：**torch 2.8.x → `xformers 0.0.32.post1`**、torch 2.10+ → `0.0.35`。其它版本查 PyPI 该版本的 `requires-dist`（里面写死 `torch==x.y.z`）或跑 `uv pip index versions xformers`。`--no-deps` 必须带——否则依赖解析可能顺手把 torch 换成另一份构建。

装完重启 `2-launch.bat`，02 参数 → 高级选项 → 注意力后端 里的 `xformers` 才会从置灰变成可选；默认值仍是 `native`，要自己点。卸载：`uv pip uninstall --python .venv\Scripts\python.exe xformers`。

---

## 8. 目录结构

```
SeeThrough-GGUF-Win/
├── 1-setup.bat / 2-launch.bat  # 环境重建 / 启动
├── 3-env.bat                   # 打开已激活 .venv 的命令提示符（设置页按钮也调它）
├── 4-install_triton.bat        # 可选：安装/卸载 triton-windows（加速反量化编译）
├── requirements.txt           # STM 依赖 + Flask（PyPI）
├── requirements-torch.txt     # PyTorch cu128（专用索引，含架构说明）
├── config.json                # 端口、路径、默认值、约束
├── LICENSE                    # Apache-2.0（整个仓库）
├── NOTICE                     # 版权与上游署名
├── MODIFICATIONS.md           # 对上游代码的逐行修改声明（Apache-2.0 §4b）
├── THIRD_PARTY_NOTICES.md     # 第三方组件与模型许可清单
├── .gitignore                 # 排除 .venv / 模型 / outputs / 缓存
├── README.md / README.en.md    # 简体中文 / English 两份说明
├── backend/
│   ├── server.py              # Flask：上传 / 校验 / 子进程 / SSE / 下载 / 取消
│   ├── selfcheck.py           # 环境自检
│   ├── worker.py              # 常驻推理进程（复用已加载的模型）
│   ├── components_paths.py    # 组件缓存路径（HF_HOME / TORCH_HOME → components/）
│   ├── install_triton.py      # Triton 安装向导（版本推导 / 测速选源 / 冒烟测试 / 配置同步）
│   ├── netcheck.py            # 下载源探测（官方 / 镜像）
│   ├── download_models.py     # 按探测结果下载两个 GGUF
│   ├── tls.py                 # 合并系统根证书（应对安全软件中间人）
│   ├── test_pipeline.py       # 后端链路集成测试
│   ├── requirements.txt
│   └── patches/apply_use_lama.py / apply_layer_fix.py
├── frontend/                  # index.html / styles.css / app.js（原生 JS + GSAP）
├── docs/
│   ├── screenshots/           # 界面截图（桩推理产出，仅示意）
│   ├── ai-setup.md            # 从零部署任务书：整份交给 AI，让它克隆 + 安装 + 验收
│   ├── triton-windows.md      # triton-windows 安装 / 手动安装 / 卸载 / 排错
│   ├── triton-windows.en.md   # 上一份的英文版
│   └── triton-windows-ai.md   # 同一件事的任务书：整份交给 AI，让它代劳
├── stm/                       # 上游代码快照（Apache-2.0，已内置并打补丁）
├── models/                    # GGUF（自行下载，不入库）
├── components/                # 首次推理的辅助组件（HF / torch.hub，不入库，1-setup.bat 自动创建）
├── outputs/jobs/<id>/         # 每次任务的输入、输出、meta.json（不入库）
└── licenses/
    └── README.md              # 许可说明与义务
```

---

## 9. 署名与许可

本项目仓库：<https://github.com/Mingri196/SeeThrough-GGUF-Win>

本项目基于 [IceFog72/see-through-gguf](https://github.com/IceFog72/see-through-gguf)（Apache-2.0，基线 commit `02cba07`），而后者建立在 [shitagaki-lab/see-through](https://github.com/shitagaki-lab/see-through) 与论文 *See-through: Layer Decomposition for Live2D Character Animation*（<https://arxiv.org/abs/2602.03749>）之上。上述署名按要求予以保留；本项目对上游代码的修改见 [MODIFICATIONS.md](MODIFICATIONS.md)。

| 组件                                  | 协议                                                 | 文本                                      |
| ----------------------------------- | -------------------------------------------------- | --------------------------------------- |
| 本项目代码（backend / frontend / 脚本 / 文档） | Apache-2.0                                         | [LICENSE](LICENSE) + [NOTICE](NOTICE)   |
| see-through-gguf（STM）代码（内置 `stm/`）  | Apache-2.0                                         | [LICENSE](LICENSE)（原始 `stm/LICENSE` 保留） |
| GSAP 3.12.5（frontend/vendor）        | GreenSock Standard License                         | 文件头保留版权声明                               |
| GGUF 权重与运行期组件                       | 见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) | 不随仓库分发                                  |

**关于模型权重**：本仓库只包含代码与脚本，不打包任何权重。经 2026-10-29 核对，两个 GGUF 仓库（`icefog72/seethroughv0.0.2_layerdiff3d_gguf`、`icefog72/seethroughv0.0.1_marigold_gguf`）**均未声明 license**，其来源模型适用 `openrail++`（CreativeML OpenRAIL++-M，带使用限制）。对外分享生成图或再分发权重前，请自行遵守相应限制并向模型作者确认授权。
