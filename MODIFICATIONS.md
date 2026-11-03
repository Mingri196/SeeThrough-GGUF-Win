# 修改声明 / Modifications

本项目对上游 [IceFog72/see-through-gguf](https://github.com/IceFog72/see-through-gguf) 的代码做了**最小、可逆**的修改，用于把「LaMa 修补」「边缘精修」暴露成真正的命令行开关，并修复三处后处理缺陷（贴边图层被误判为空层、眼白盖住虹膜、掩码索引用错类型导致崩溃）。

- 上游基线：`02cba07`（2026-09-14，*feat: add ragdoll confidence parameter and post-processing script*）
- 修改者：Mingri196（<https://github.com/Mingri196/SeeThrough-GGUF-Win>，非上游作者，修改未获上游背书）
- 修改日期：2026-10-29（LaMa 开关）、2026-10-30（图层修补）、2026-11-02（边缘精修）、2026-11-03（切图保存健壮性）、2026-11-03（移除附带字体）、2026-11-03（头部裁剪范围）
- 许可证：Apache-2.0，全文见根目录 [LICENSE](LICENSE)

## 被修改的文件

### 1. `common/utils/inference_utils.py`

| 位置 | 改动 |
|---|---|
| 文件首行 | 增加标记注释 `# __STG_PATCHED_INFERENCE_UTILS__`、`# __STG_PATCHED_INFERENCE_UTILS_LAYER_FIX__`、`# __STG_PATCHED_INFERENCE_UTILS_HEAD_CROP__`（供补丁幂等判断） |
| `further_extr()` 签名 | 在 `tblr_split=True` 之后插入参数 `use_lama=True` |
| `further_extr()` 内 hair 分支 | `cluster_inpaint_part(**part_info)` → `cluster_inpaint_part(inpaint='lama' if use_lama else 'cv2', **part_info)` |
| `dump_parts_psd()` 之前 | 新增辅助函数 `_stg_reorder_eyes(items)`：把眼睛三层（eyewhite / irides / eyelash）收拢后按固定顺序重排 |
| `dump_parts_psd()` 内排序 | 深度排序之后调用 `part_dict_list = _stg_reorder_eyes(part_dict_list)` |
| `process_cuts()` 内深度中位数 | 掩码是 uint8，`depth[mask]` 会走 numpy 花式索引（把 0/1 当行号）；改成 `depth[mask.astype(bool)]` |
| v3 分支 `_crop_head()` | 头部裁剪的外扩由固定 `w // 5` / `h // 5` 改成 `w // div` / `h // div`，`div` 取自环境变量 `STG_HEAD_CROP`（默认 `2`，`=5` 退回上游行为，`=1` 不扩） |

### 2. `inference/scripts/inference_psd.py`

| 位置 | 改动 |
|---|---|
| 文件首行 | 两个标记注释：`# __STG_PATCHED_INFERENCE_PSDPY__`、`# __STG_PATCHED_INFERENCE_PSDPY_EDGE_REFINE__`（供补丁幂等判断） |
| 参数定义区 | `--use_lama`（`BooleanOptionalAction`，默认 `True`）、`--soft_alpha`（`BooleanOptionalAction`，**默认 `True`**）、`--edge_refine`（`store_true`，**默认 `False`**） |
| 调用 `further_extr()` 处 | 传入 `use_lama=args.use_lama` |
| `save_masked_input()` 签名 | 末尾新增参数 `edge_refine=False, soft_alpha=True` |
| `save_masked_input()` 无 alpha 分支 | `soft_alpha` 打开（默认）时用 `_stg_soft_alpha()` 保留边缘过渡；`--no-soft_alpha` 退回上游的 `(prob >= 0.5) * 255` |
| `save_masked_input()` 有 alpha 分支 | `edge_refine` 打开时也跑一次分割器，再交给 `_stg_refine_alpha_band()` |
| 新增 `_stg_refine_alpha_band()` | 只把分割器概率写进原 alpha 的边缘带（±3 px），带宽之外保持用户原值 |
| `further_extr()` 保存切图处 | `Image.fromarray(part_dict['img'])` 改为 `np.ascontiguousarray(...)`，并跳过 0 尺寸的空层（骨骼切割后某个关节没分到像素会触发 `SystemError: tile cannot extend outside image`） |
| `save_part()` 落盘分支 | 同上，PNG 与 depth 写出前都转连续内存 |
| main 循环 | 传 `args.edge_refine`；打开时带 alpha 的图也会加载分割器 |

### 3. `common/utils/io_utils.py`（图层修补）

| 位置 | 改动 |
|---|---|
| 文件首行 | 增加标记注释 `# __STG_PATCHED_IO_UTILS_LAYER_FIX__` |
| `load_part()` 空层判定 | `if np.sum(mask[:-p_test, :-p_test]) > 4:` → 默认统计整张 mask（`np.count_nonzero(mask)`） |

### 4. `common/utils/visualize.py`（移除附带字体）

| 位置 | 改动 |
|---|---|
| 文件首行 | 增加标记注释 `# __STG_PATCHED_VISUALIZE_FONT__` |
| 新增 `_stg_text_font(size)` | 返回 `ImageFont.load_default(size)`（Pillow < 10.1 时退回无参调用） |
| `visualize_pos_keypoints()` 绘制置信度处 | `ImageFont.truetype("assets/arial.ttf", size=20)` → `_stg_text_font(20)` |
| `pil_draw_text()` 默认字体分支 | `ImageFont.truetype("assets/arial.ttf", size=font_size)` → `_stg_text_font(font_size)` |

### 5. 移除的上游资源文件

| 文件 | 原因 |
|---|---|
| `common/assets/arial.ttf`（367 KB） | Monotype 商业字体。字体内嵌授权声明限定「限于本机自用、不得复制或分发」，随 Apache-2.0 仓库公开分发与该条款冲突。该文件在本项目中不参与任何执行路径（见 `## 说明`），已从仓库移除，改用 Pillow 自带字形 |

## 说明

- 除上述四个文件外，没有对其他上游文件做任何修改；`seg_wdepth()` / `seg_wdepth_psd()` 中的另外两处 `cluster_inpaint_part` 调用保持原样。
- `use_lama` 默认值 `True` 与上游原行为一致：即使不传该参数，仍使用 LaMa。
- **`--soft_alpha` 默认开启**，这是与上游唯一的默认行为差异：无 alpha 的输入默认保留分割器的连续概率，边缘不再被 `0.5` 硬切成锯齿。要完全回到上游行为，传 `--no-soft_alpha`（注意是下划线：`BooleanOptionalAction` 的否定名跟随参数名本身，连字符写法会被 argparse 判为未知参数）。
- `--edge_refine` 默认**关闭**，且**只影响已带 alpha 的图**：打开后才会额外跑一次分割器并只替换边缘 ±3 像素。
- Web 界面映射成主从两个开关：主「边缘精修」→ `--soft_alpha`（默认开）；其下的从属开关「透明图也精修」→ `--edge_refine`（默认关）。主开关关闭时从属项置灰不可点，后端也会把 `edge_refine` 强制回 `false`。
- **头部裁剪范围**：v3 路径下，头部各层（headwear / face / 眼睛 / 耳朵等）是把头部裁出来放大再推理得到的。裁剪框按 `head` 层的 bbox 算，而 `head` 只含头皮和脸，**不含头发和发饰**，上游只外扩 1/5 宽度，框会正好卡在发饰边缘。实测发包任务：`head` 88×85、发饰 125×119、裁剪框 122×119，**发饰占框 99%**，模型拿到一张"铺满发饰"的图，分不清花和头发，把整条顶部都判成 `headwear`，合成图头顶多出一块横带。改成外扩 1/2 后发饰占框降到约 52%，模型有头发和背景作参照。代价是头部放大倍数从 6.3× 降到 4.4×，五官的绝对像素略少。环境变量 `STG_HEAD_CROP` 可调（`5` = 上游行为，`3` = 折中，`1` = 不扩）。
- **图层修补**的三处改动都由环境变量 `STG_LAYER_FIX` 控制（默认开启），Web 界面的「图层修补」开关按任务注入。该变量支持细粒度取值：`0` / `off` = 全关（完全等同上游原始行为）、`1` / `on` = 全开，也可以写成 `drop` / `order` / `mask` / `drop,order` 只开其中几项（分别对应下面三处修复）：
  - **丢层**：原判定先裁掉底部/右侧 10% 条带再数 alpha 像素，贴着画布下沿的层（典型是鞋子，bbox 常正好到 768 底边）会被整体算成 0 而整层丢弃。改为统计整张 mask。
  - **叠序**：`depth_median` 排序在眼睛三层深度极接近时（实测 eyewhite 0.341 / irides 0.349）会颠倒，导致眼白盖住虹膜。改为把这三层收拢后按 `eyewhite → irides → eyelash` 固定顺序排列，整体位置不变。
  - **掩码索引**：`process_cuts()` 把掩码转成 uint8 后再用 `depth[mask]` 取值。numpy 对整数数组走的是花式索引（把 0/1 当行号去取第 0/1 行），不是布尔掩码 —— 算出来的深度中位数一直是错的；左右分割时若某个连通域高度只剩 1 像素（手部贴边被切窄），访问第 1 行直接越界，整个推理以 `IndexError: index 1 is out of bounds for axis 0 with size 1` 退出。改成先转 bool 再索引。
  - 实测（同一张 768 输出重跑后处理）：PSD 层数 22 → 23（footwear 回归）；相对原图的空白像素 4728 → 991（两只脚各约 1900 px 的缺口消失）；眼睛层顺序由「眼白在上」变为「虹膜在上」。
- 补丁脚本：[`backend/patches/apply_use_lama.py`](backend/patches/apply_use_lama.py)、[`backend/patches/apply_layer_fix.py`](backend/patches/apply_layer_fix.py)、[`backend/patches/apply_head_crop.py`](backend/patches/apply_head_crop.py)。都是幂等的（靠上面的标记注释判断），对未打补丁的文件会先写 `*.bak` 再改（仅本地生成，不随仓库分发）。
- `stm/` 已作为上游快照直接内置在本仓库中（已打补丁，`.git` 已移除）。本文件即 Apache-2.0 §4(b) 要求的「修改声明」。
- 补丁脚本仍然保留，用于把同样的改动重新施加到**更新后的上游版本**上（幂等，已打过的文件会直接跳过）。
- 还原方式：从上游 commit `02cba07` 重新获取这四个文件覆盖即可（本仓库不再保留 `.bak` 备份，历史版本请以上游仓库为准）。
- **字体移除的影响面为零**：全项目只有 `visualize.py` 这两处引用 `assets/arial.ttf`，都是相对路径。而后端只执行 `inference/scripts/inference_psd.py`，它的导入链不经过 `visualize`；真正会调用这两个函数的 `parse_live2d.py` / `syn_data.py` / `scrap_model.py` 属于训练与标注链路，本项目不执行。即便执行到，项目根也没有 `assets/` 目录、全程没有 `chdir`，相对路径本来就定位不到这个文件，Pillow 会回落到系统字体（实测加载的是 `C:\WINDOWS\fonts\arial.ttf`）。改用 `load_default()` 后行为等价，且不再依赖任何外部字体。
