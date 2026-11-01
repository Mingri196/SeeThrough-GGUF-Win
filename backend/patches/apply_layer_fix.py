#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
apply_layer_fix.py  —  SeeThrough-GGUF-Win 的「图层修补」补丁（幂等）

修三个上游后处理缺陷（都只在 PSD / 切图装配阶段，不涉及采样，不重画任何像素）：

1. 丢层（drop）：`load_part()` 用 `np.sum(mask[:-p_test, :-p_test]) > 4` 判断是不是空层，
   先砍掉底部/右侧 10% 条带再数像素。贴着画布下沿的内容（典型是鞋子，bbox 常常
   正好到 768 底边）会被整体算成 0 → 整层被丢弃。补丁改成数整张 mask。

2. 叠序（order）：`dump_parts_psd()` 按 depth_median 从远到近堆层。眼睛三层深度极接近
   （eyewhite 0.341 / irides 0.349 / eyelash 0.352），深度噪声会让眼白压在虹膜
   上面 → 眼睛发白。补丁把眼睛三层收拢到一起，按 eyewhite → irides → eyelash
   的固定顺序重排（整体仍停在原位置，不影响与脸、头发的相对关系）。

3. 掩码（mask）：`process_cuts()` 把掩码转成 uint8 后再用 `depth[mask]` 取值。numpy 对
   整数数组走的是花式索引（把 0/1 当行号），不是布尔掩码 —— 算出来的 depth_median
   一直是错的；连通域高度只有 1 像素时更会直接 IndexError 崩掉整个推理。补丁改成
   先转成 bool 再索引。

开关：环境变量 STG_LAYER_FIX，默认全开。
    STG_LAYER_FIX=1            → 全开（默认）
    STG_LAYER_FIX=0            → 全关（完全等同上游原始行为）
    STG_LAYER_FIX=drop         → 只开丢层
    STG_LAYER_FIX=order        → 只开眼白叠序
    STG_LAYER_FIX=mask         → 只开掩码索引
    STG_LAYER_FIX=drop,order   → 开其中几项
Web 界面的「图层修补」开关会按任务注入推理子进程 / 常驻进程。

- 幂等：已打过补丁则跳过。
- 安全：修改前写 .bak 备份。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from apply_use_lama import STM, patch_file  # noqa: E402

IO_UTILS = STM / "common" / "utils" / "io_utils.py"
INFERENCE_UTILS = STM / "common" / "utils" / "inference_utils.py"

# 开关解析：两个文件里各放一份（它们互相不 import）
SWITCH_HELPER = '''def _stg_fix_enabled(name):
    # __STG_LAYER_FIX__ 图层修补开关。STG_LAYER_FIX=0/off 全关，=1/on 全开，
    # 也可以写成 "drop,order,mask" 只开其中几项（名字对应补丁里的三处修复）。
    v = str(os.environ.get("STG_LAYER_FIX", "1")).strip().lower()
    if v in ("0", "off", "false", "no"):
        return False
    if v in ("", "1", "on", "true", "yes"):
        return True
    return name in [x.strip() for x in v.split(",") if x.strip()]


'''

REORDER_HELPER = '''def _stg_reorder_eyes(items):
    # __STG_LAYER_FIX__ 眼白在虹膜之下、睫毛之上；深度太接近时按深度排会颠倒
    if not _stg_fix_enabled("order"):
        return items
    rank = {"eyewhite": 0, "irides": 1, "eyelash": 2}
    eyes = [i for i in items if str(i.get("tag", "")).rsplit("-", 1)[0] in rank]
    if len(eyes) < 2:
        return items
    eye_ids = {id(i) for i in eyes}
    rest = [i for i in items if id(i) not in eye_ids]
    slot = min(items.index(i) for i in eyes)
    eyes.sort(key=lambda i: rank[str(i.get("tag", "")).rsplit("-", 1)[0]])
    return rest[:slot] + eyes + rest[slot:]


'''


def main() -> int:
    print("== 图层修补补丁（丢层 + 眼白叠序 + 掩码索引）==")
    if not STM.exists():
        print(f"[提示] STM 仓库不存在（{STM}），跳过。")
        return 0

    ok = patch_file(
        IO_UTILS,
        [
            # 1) 插入开关解析函数
            (
                r"import numpy as np\n",
                "import numpy as np\n\n" + SWITCH_HELPER,
                True,
            ),
            # 2) 空层判定：默认数整张 mask；关掉「丢层」时退回原来的「排除底部条带」
            (
                r"    if np\.sum\(mask\[:-p_test, :-p_test\]\) > 4:",
                '    if np.count_nonzero(mask if _stg_fix_enabled("drop") '
                "else mask[:-p_test, :-p_test]) > 4:",
                True,
            ),
        ],
        label="IO_UTILS_LAYER_FIX",
    )

    ok = ok and patch_file(
        INFERENCE_UTILS,
        [
            # 3) 插入开关解析函数
            (
                r"import numpy as np\n",
                "import numpy as np\n\n" + SWITCH_HELPER,
                True,
            ),
            # 4) 插入眼睛叠序的辅助函数
            (
                r"def dump_parts_psd\(",
                REORDER_HELPER + "def dump_parts_psd(",
                True,
            ),
            # 5) 排完深度后，把眼睛三层按固定顺序收拢重排
            (
                r"    part_dict_list\.sort\(key=lambda x: x\['depth_median'\], reverse=True\)",
                "    part_dict_list.sort(key=lambda x: x['depth_median'], reverse=True)\n"
                "    part_dict_list = _stg_reorder_eyes(part_dict_list)",
                True,
            ),
            # 6) process_cuts：掩码是 uint8，depth[mask] 会走花式索引（取第 0/1 行），
            #    算出的 depth_median 一直是错的；连通域只有 1 像素高时还会 IndexError 崩掉。
            #    改成先转 bool 再索引。
            (
                r"        if np\.any\(mask\):\n            depth_median = np\.median\(depth\[mask\]\)\n",
                "        if np.any(mask):\n"
                "            depth_median = np.median(\n"
                '                depth[mask.astype(bool)] if _stg_fix_enabled("mask") else depth[mask]\n'
                "            )\n",
                True,
            ),
        ],
        label="INFERENCE_UTILS_LAYER_FIX",
    )

    if ok:
        print(
            "[OK] 图层修补补丁处理完成"
            "（开关：config.json → layer_fix；STG_LAYER_FIX=0 全关，或 drop/order/mask 选开）。"
        )
        return 0
    print("[失败] 补丁应用出现异常。")
    return 1


if __name__ == "__main__":
    sys.exit(main())
