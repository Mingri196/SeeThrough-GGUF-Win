# triton-windows：交给 AI 操作（任务书）

> **这份文档是给 AI 读的**，也可以当人的说明书。想装 / 卸 / 排查 triton-windows，把这个文件直接丢给 AI（或说「按 `docs/triton-windows-ai.md` 操作」）即可，不用复制任何提示词。
>
> 人只需要补一句你想要什么：**「装」「卸」「我装了但没变快」**——其余判定、命令、红线都在下面。手动命令与原理见 [triton-windows.md](triton-windows.md)。

---

## 1. 给 AI 的角色说明

你正在协助用户操作 **SeeThrough-GGUF-Win**（Windows 本地网页工具，上游 `see-through-gguf` / STM 已内置在 `stm/`）。任务范围仅限 **triton-windows 的安装 / 卸载 / 效果排查**，以及为此必需的只读诊断。

**项目根目录**＝当前打开的这个仓库目录，即含 `1-setup.bat`、`config.json`、`4-install_triton.bat`、`.venv` 的那一层。不同人 clone 的位置不同，**不要写死任何绝对路径，也不要凭猜测 `cd`**：

- 不确定就先报 `Get-Location`（PowerShell）或 `pwd`（bash）的结果；
- 如果当前目录不是项目根目录，先确认正确位置再动手；找不到就停下来问用户。

**关键背景**（判断依据，别跳过）：

| 事实 | 含义 |
|---|---|
| `config.json` → `compile_dequant` | 开关。为 `true` 时 `backend/server.py` 会给推理子进程注入 `SEETHROUGH_COMPILE_DEQUANT=1`，`stm/common/utils/ggml_runtime.py` 用 `torch.compile` 编译 Q4_K / Q6_K 反量化核 |
| 项目固定 `torch 2.8.0+cu128` | 所以 triton 必须装 `3.4.*`。对应表：2.4/2.5→3.1、2.6→3.2、2.7→3.3、**2.8→3.4**、2.9→3.5、2.10→3.6。装错版本会 `undefined symbol` |
| 显卡需 CC ≥ 7.0 | 反量化核用了 `evict_last` 缓存提示，sm < 70 上 ptxas 直接拒绝（`GPUTooOldForTriton`）。**这是预期结果，不是故障** |
| Windows 上 `torch.compile` 还要 MSVC `cl.exe` | 缺了报 `Compiler: cl is not found`。这属于环境缺失，只报告，不替用户装 Visual Studio |
| 收益上限 | 反量化 kernel 单点约 5.88×，但整轮出图只省 **10–13%**（瓶颈在 UNet 采样）。别向用户承诺翻倍 |

**唯一推荐入口**：`4-install_triton.bat`。它是 `backend/install_triton.py` 的壳，负责探测 → 版本推导 → 测速选源 → 安装 → **用真实反量化 kernel 做冒烟测试** → 只在通过时才把 `compile_dequant` 置 `true`。不要用一串手写 `pip install` 去替代它。

---

## 2. 执行流程（三种目标，先判定再动手）

### 第 0 步：判断用户要哪一种

| 用户说的话 | 走哪条 |
|---|---|
| 「装 / 试一下 triton / 能不能加速」 | → §3 安装 |
| 「卸掉 / 回滚 / 不想要了 / 出问题了」 | → §4 卸载 |
| 「装了但没变快 / 好像没生效」 | → §5 排查 |

用户没说清就先问一句，**不要三种一起做**。

### 第 1 步：只读诊断（三条路径都要先做）

```bat
:: torch 版本 + 显卡算力
.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available(), torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None)"

:: 已装的 triton
.venv\Scripts\python.exe -c "import importlib.metadata as m; print(m.version('triton-windows'))"

:: cl.exe 是否存在
where cl
```

再看 `config.json` 里 `compile_dequant` 的当前值。

诊断阶段的硬约束：**不安装、不改文件、不改配置**。

### 第 2 步：按判定表分流

| 诊断结果 | 动作 |
|---|---|
| 没有 `.venv` | **停止**。告诉用户先跑 `1-setup.bat`，脚本不会碰系统 Python |
| CC < 7.0 | **停止安装**，直接说明「装上也编不出 kernel，收益为 0，项目默认已关闭该加速」，并问用户是否仍要装（默认建议不装）。用户坚持才继续，且装完**不得**把 `compile_dequant` 设为 `true` |
| CC ≥ 7.0 且没装 triton | → §3 安装 |
| CC ≥ 7.0 且已装但版本与 torch 不匹配 | → §3 用 `--reinstall` |
| 已装且版本匹配、`compile_dequant` 已为 `true` | 说明已在启用状态，引导走 §5 排查效果 |
| 无 NVIDIA GPU / CUDA 不可用 | **停止**，说明 triton 装了也用不上 |

---

## 3. 模式 A：安装

### 命令

```bat
:: 无人值守（自动选最快源），推荐
4-install_triton.bat --yes

:: 国内网络卡在测速上时，直接指定源：1 清华 / 2 阿里 / 3 PyPI 官方
4-install_triton.bat --channel 2

:: 已装但版本不对，重装
4-install_triton.bat --reinstall --yes
```

交互式（不带参数）会依次问：已安装则「保留/卸载/重装」→ 显卡门槛二次确认 → 选源。**不要替用户乱答**：门槛确认这类是否继续的提问，交由用户决定，你只转述。

### 完成判据

以向导输出为准，不要自己另写验证脚本替代：

| 结尾输出 | 结论 |
|---|---|
| `[完成] 已启用 compile_dequant（config.json → true）` | 成功。告诉用户**必须重启服务**（关掉 `2-launch.bat` 窗口再开）才生效 |
| `[已安装，但未启用] ... 保持 false` | 包已装、冒烟没过（多为 CC < 7.0 或缺 MSVC）。功能不受影响，把 `reason` 原文转述给用户 |
| `[拒绝] 项目根目录下没有 .venv` | 让用户先 `1-setup.bat` |
| `[失败] 安装命令返回非零` | 网络/版本问题，把 pip 输出尾部原样贴给用户，不要反复重试超过 1 次 |

安装后如果需要确认状态，只读检查这三项即可：`uv pip list | findstr triton`、`config.json` 的 `compile_dequant`、向导最后 15 行输出。

> 重启服务这一步通常要用户自己做（他可能正在页面里跑任务）。不要擅自 `taskkill` 用户的进程；页面上还挂着「模型常驻」时要提醒他先在 03 里关掉或取消任务，否则旧 worker 不会带上新环境变量。

---

## 4. 模式 B：卸载 / 回滚

### 命令

```bat
4-install_triton.bat --uninstall
```

它会 `uv pip uninstall triton-windows`，并**自动把 `compile_dequant` 改回 `false`**。

### 完成判据

1. 输出里有「已卸载」和「已把 config.json 的 compile_dequant 改回 false」；
2. 复查 `config.json` 的 `compile_dequant` 确实是 `false`——**这一步不能省**。只卸包不改配置，下次启动仍会注入 `SEETHROUGH_COMPILE_DEQUANT=1`，`ggml_runtime.py` 在 import 阶段就要编译 kernel，缺后端会让整个推理直接失败；
3. 复查 `.venv` 里已无 `triton-windows`（`uv pip list` 或 `importlib.metadata`）；
4. 验证 STM 反量化模块能正常 import：

```bat
.venv\Scripts\python.exe -c "import os,sys; os.environ['SEETHROUGH_COMPILE_DEQUANT']='0'; sys.path[:0]=['stm/common','stm']; from utils import ggml_runtime; print('ok')"
```

### 红线

- **不要删缓存目录**：`%USERPROFILE%\.triton\cache`、`%TEMP%\torchinductor_*` 留着不影响正确性，删了要重新编译。用户明确要求清才清，且只清目录内部，不递归删父目录。
- 不要动 `outputs/jobs/` 下的历史任务。

---

## 5. 模式 C：排查「装了没效果」

只读排查，**不改配置、不重装、不改源码**。

1. **确认编译真的通过**：按 [triton-windows.md](triton-windows.md) 第 3.3 节，把冒烟脚本存到 `outputs/smoke_triton.py`（`outputs/` 不入库）并运行，看是否 `allclose: True`。报错就把原文对照该文档第 5 节。
2. **确认实际档位**：看本次任务日志里的 `resident` / `streamed`。分辨率高时会掉进 `streamed`，反量化占比大、Triton 收益相对明显；`resident` 档或低分辨率下，反量化只占一小段，本来就只有 10–13%。
3. **给出结论，二选一**：
   - 「本来收益就那么点」——属正常，建议用户优先降分辨率 / 减步数而不是折腾编译器；
   - 「编译没生效 / 回退到 eager」——给定位方法（例如在 `ggml_runtime.py` 的编译分支加一次性打印），但**先不要直接改文件**，等用户点头。
4. 常见原因对照：旧常驻 worker 未重启（环境变量是起进程时注入的）、`config.json` 被改回 `false`、torch/triton 版本错配后编译静默失败。

---

## 6. 红线清单（任何模式都适用）

- 只操作项目内的 `.venv`；**不动系统 Python**、不 `pip install --user`、不改 `PATH`、不写注册表。
- **不改 `stm/` 下的上游代码**，不改反量化 kernel，不绕过 `evict_last` / `sm_70` 限制。
- **不设 `STG_TRITON_TEST_CC` / `STG_TRITON_TEST_TORCH`** 这类测试钩子——它们会伪造显卡算力与 torch 版本，专门用于调试脚本分支，用在真实环境会让判定完全失真。
- 冒烟没过**不得**手动把 `compile_dequant` 改成 `true`。
- 不替用户安装 Visual Studio / CUDA Toolkit / 显卡驱动。
- 不 `git commit` / `git push`，不改 `README.md` 与 `docs/` 里的既有文档（除非用户明确要求）。
- 不擅自结束用户的进程（`taskkill` / 重启服务），只提醒。
- 不猜测路径；不确定先报当前目录再问。
- 遇到与本文档描述不符的输出，**以实际输出为准并停下来报告**，不要为了「跑通」而改判断标准。

---

## 7. 汇报模板

做完后按这个结构回话，别贴大段无关日志：

```text
【环境】torch 2.8.0+cu128 · CC 8.6 · triton 未装 / 已装 3.4.x · MSVC cl.exe 有 / 无
【动作】运行了 4-install_triton.bat --yes（或 --uninstall / 只读排查）
【结果】成功 / 已装未启用 / 失败（贴向导最后 15 行原样输出）
【配置】config.json 的 compile_dequant：true / false（改前 → 改后）
【下一步】例如「请关掉 2-launch.bat 窗口重新启动；页面里若还挂着模型常驻，先在 03 里关掉」
【风险/遗留】例如「CC 6.1 无法编译，已保持关闭」「缺 cl.exe，建议装 VS Build Tools 的 C++ 桌面开发」
```

---

## 8. 参考

- 脚本安装 / 手动安装 / 卸载细节、版本对应表、常见报错：**[triton-windows.md](triton-windows.md)**
- 入口脚本：`4-install_triton.bat`；实现在 `backend/install_triton.py`
- 开关：`config.json` → `compile_dequant`；注入点：`backend/server.py`；被编译的核：`stm/common/utils/ggml_runtime.py`
