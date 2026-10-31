#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
apply_layer_fix.py  —  SeeThrough-GGUF-Win 的「图层修补」补丁（幂等）

修两个上游后处理缺陷（都只在 PSD / 切图装配阶段，不涉及采样，不重画任何像素）：

1. 丢层：`load_part()` 用 `np.sum(mask[:-p_test, :-p_test]) > 4` 判断是不是空层，
   先砍掉底部/右侧 10% 条带再数像素。贴着画布下沿的内容（典型是鞋子，bbox 常常
   正好到 768 底边）会被整体算成 0 → 整层被丢弃。补丁改成数整张 mask。

2. 叠序：`dump_parts_psd()` 按 depth_median 从远到近堆层。眼睛三层深度极接近
   （eyewhite 0.341 / irides 0.349 / eyelash 0.352），深度噪声会让眼白压在虹膜
   上面 → 眼睛发白。补丁把眼睛三层收拢到一起，按 eyewhite → irides → eyelash
   的固定顺序重排（整体仍停在原位置，不影响与脸、头发的相对关系）。

开关：环境变量 STG_LAYER_FIX=0 关闭（两处都回退成上游原始行为），默认开启。
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

HELPER = '''def _stg_reorder_eyes(items):
    # __STG_LAYER_FIX__ 眼白在虹膜之下、睫毛之上；深度太接近时按深度排会颠倒
    if os.environ.get("STG_LAYER_FIX", "1") == "0":
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
    print("== 图层修补补丁（丢层 + 眼白叠序）==")
    if not STM.exists():
        print(f"[提示] STM 仓库不存在（{STM}），跳过。")
        return 0

    ok = patch_file(
        IO_UTILS,
        [
            # 1) 空层判定：默认数整张 mask；STG_LAYER_FIX=0 时退回原来的「排除底部条带」
            (
                r"    if np\.sum\(mask\[:-p_test, :-p_test\]\) > 4:",
                '    if np.count_nonzero(mask if os.environ.get("STG_LAYER_FIX", "1") != "0" '
                "else mask[:-p_test, :-p_test]) > 4:",
                True,
            ),
        ],
        label="IO_UTILS_LAYER_FIX",
    )

    ok = ok and patch_file(
        INFERENCE_UTILS,
        [
            # 2) 插入眼睛叠序的辅助函数
            (
                r"def dump_parts_psd\(",
                HELPER + "def dump_parts_psd(",
                True,
            ),
            # 3) 排完深度后，把眼睛三层按固定顺序收拢重排
            (
                r"    part_dict_list\.sort\(key=lambda x: x\['depth_median'\], reverse=True\)",
                "    part_dict_list.sort(key=lambda x: x['depth_median'], reverse=True)\n"
                "    part_dict_list = _stg_reorder_eyes(part_dict_list)",
                True,
            ),
        ],
        label="INFERENCE_UTILS_LAYER_FIX",
    )

    if ok:
        print("[OK] 图层修补补丁处理完成（开关：config.json → layer_fix，或 STG_LAYER_FIX=0 关闭）。")
        return 0
    print("[失败] 补丁应用出现异常。")
    return 1


if __name__ == "__main__":
    sys.exit(main())
