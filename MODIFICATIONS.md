# 修改声明 / Modifications

本项目对上游 [IceFog72/see-through-gguf](https://github.com/IceFog72/see-through-gguf) 的代码做了**最小、可逆**的修改，用于把「LaMa 修补」暴露成一个真正的命令行开关，并修复三处后处理缺陷（贴边图层被误判为空层、眼白盖住虹膜、掩码索引用错类型导致崩溃）。

- 上游基线：`02cba07`（2026-09-14，*feat: add ragdoll confidence parameter and post-processing script*）
- 修改者：Mingri196（<https://github.com/Mingri196/SeeThrough-GGUF-Win>，非上游作者，修改未获上游背书）
- 修改日期：2026-10-29（LaMa 开关）、2026-10-30（图层修补）
- 许可证：Apache-2.0，全文见根目录 [LICENSE](LICENSE)

## 被修改的文件

### 1. `common/utils/inference_utils.py`

| 位置 | 改动 |
|---|---|
| 文件首行 | 增加标记注释 `# __STG_PATCHED_INFERENCE_UTILS__` 与 `# __STG_PATCHED_INFERENCE_UTILS_LAYER_FIX__`（供补丁幂等判断） |
| `further_extr()` 签名 | 在 `tblr_split=True` 之后插入参数 `use_lama=True` |
| `further_extr()` 内 hair 分支 | `cluster_inpaint_part(**part_info)` → `cluster_inpaint_part(inpaint='lama' if use_lama else 'cv2', **part_info)` |
| `dump_parts_psd()` 之前 | 新增辅助函数 `_stg_reorder_eyes(items)`：把眼睛三层（eyewhite / irides / eyelash）收拢后按固定顺序重排 |
| `dump_parts_psd()` 内排序 | 深度排序之后调用 `part_dict_list = _stg_reorder_eyes(part_dict_list)` |
| `process_cuts()` 内深度中位数 | 掩码是 uint8，`depth[mask]` 会走 numpy 花式索引（把 0/1 当行号）；改成 `depth[mask.astype(bool)]` |

### 2. `inference/scripts/inference_psd.py`

| 位置 | 改动 |
|---|---|
| 文件首行 | 增加标记注释 `# __STG_PATCHED_INFERENCE_PSDPY__` |
| 参数定义区 | 新增 `parser.add_argument('--use_lama', action=argparse.BooleanOptionalAction, default=True, help='use LaMa inpainting during post-processing (default on)')` |
| 调用 `further_extr()` 处 | 传入 `use_lama=args.use_lama` |

### 3. `common/utils/io_utils.py`（图层修补）

| 位置 | 改动 |
|---|---|
| 文件首行 | 增加标记注释 `# __STG_PATCHED_IO_UTILS_LAYER_FIX__` |
| `load_part()` 空层判定 | `if np.sum(mask[:-p_test, :-p_test]) > 4:` → 默认统计整张 mask（`np.count_nonzero(mask)`） |

## 说明

- 除上述三个文件外，没有对其他上游文件做任何修改；`seg_wdepth()` / `seg_wdepth_psd()` 中的另外两处 `cluster_inpaint_part` 调用保持原样。
- `use_lama` 默认值 `True` 与上游原行为一致：即使不传该参数，仍使用 LaMa。
- **图层修补**的三处改动都由环境变量 `STG_LAYER_FIX` 控制（默认开启），Web 界面的「图层修补」开关按任务注入。该变量支持细粒度取值：`0` / `off` = 全关（完全等同上游原始行为）、`1` / `on` = 全开，也可以写成 `drop` / `order` / `mask` / `drop,order` 只开其中几项（分别对应下面三处修复）：
  - **丢层**：原判定先裁掉底部/右侧 10% 条带再数 alpha 像素，贴着画布下沿的层（典型是鞋子，bbox 常正好到 768 底边）会被整体算成 0 而整层丢弃。改为统计整张 mask。
  - **叠序**：`depth_median` 排序在眼睛三层深度极接近时（实测 eyewhite 0.341 / irides 0.349）会颠倒，导致眼白盖住虹膜。改为把这三层收拢后按 `eyewhite → irides → eyelash` 固定顺序排列，整体位置不变。
  - **掩码索引**：`process_cuts()` 把掩码转成 uint8 后再用 `depth[mask]` 取值。numpy 对整数数组走的是花式索引（把 0/1 当行号去取第 0/1 行），不是布尔掩码 —— 算出来的深度中位数一直是错的；左右分割时若某个连通域高度只剩 1 像素（手部贴边被切窄），访问第 1 行直接越界，整个推理以 `IndexError: index 1 is out of bounds for axis 0 with size 1` 退出。改成先转 bool 再索引。
  - 实测（同一张 768 输出重跑后处理）：PSD 层数 22 → 23（footwear 回归）；相对原图的空白像素 4728 → 991（两只脚各约 1900 px 的缺口消失）；眼睛层顺序由「眼白在上」变为「虹膜在上」。
- 补丁脚本：[`backend/patches/apply_use_lama.py`](backend/patches/apply_use_lama.py)、[`backend/patches/apply_layer_fix.py`](backend/patches/apply_layer_fix.py)。都是幂等的（靠上面的标记注释判断），对未打补丁的文件会先写 `*.bak` 再改（仅本地生成，不随仓库分发）。
- `stm/` 已作为上游快照直接内置在本仓库中（已打补丁，`.git` 已移除）。本文件即 Apache-2.0 §4(b) 要求的「修改声明」。
- 补丁脚本仍然保留，用于把同样的改动重新施加到**更新后的上游版本**上（幂等，已打过的文件会直接跳过）。
- 还原方式：从上游 commit `02cba07` 重新获取这三个文件覆盖即可（本仓库不再保留 `.bak` 备份，历史版本请以上游仓库为准）。
