#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
apply_head_crop.py  —  SeeThrough-GGUF-Win 的「头部裁剪范围」补丁（幂等）

问题：v3 路径下，头部各层（headwear / face / 眼睛 / 耳朵等）是把头部裁出来、
放大到推理分辨率再跑一次得到的。裁剪范围按 `head` 层的 bbox 算，而 `head` 只是
"头皮 + 脸"，**不含头发和发饰**，只外扩 1/5 宽度。结果框正好卡在发饰边缘：

    发包任务实测：head 88x85，发饰 125x119，裁剪框 122x119
    → 发饰占框 99%，模型拿到一张"铺满发饰"的图

模型分不清哪是花、哪是头发，把整条顶部都判成 headwear，合成图头顶就多出一块
横带（reconstruction.png 上很明显）。

补丁把外扩比例从 1/5 改成 1/2（可用环境变量覆盖）：

    STG_HEAD_CROP=2   外扩 1/2（默认，补丁后的行为）
    STG_HEAD_CROP=3   外扩 1/3
    STG_HEAD_CROP=5   外扩 1/5（上游原始行为，用于对比）
    STG_HEAD_CROP=1   不外扩，直接用 head 的 bbox

注意：改大裁剪框会降低头部区域的放大倍数（6.3x → 4.4x），眼睛、睫毛这类小层的
绝对像素会变少。如果发现五官变糊，可以往 3 调一档试。

- 幂等：已打过补丁则跳过。
- 安全：修改前写 .bak 备份。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from apply_use_lama import STM, INFERENCE_UTILS, patch_file  # noqa: E402

SNIPPET = '''            # __STG_HEAD_CROP__ 头部裁剪的外扩倍数（见 backend/patches/apply_head_crop.py）。
            # 上游固定 1/5：head 只含头皮和脸，框会卡在发饰边缘，发饰占框 99%，
            # 模型分不清花和头发，把整条顶部都判成 headwear。
            try:
                div = int(os.environ.get("STG_HEAD_CROP", "2"))
            except (TypeError, ValueError):
                div = 2
            div = max(1, div)

'''


def main():
    print("== 头部裁剪范围补丁 ==")
    if not STM.exists():
        print(f"[提示] STM 目录不存在（{STM}）。请先运行 1-setup.bat，本补丁将跳过。")
        return 0

    ok = patch_file(
        INFERENCE_UTILS,
        [
            # 1) 在 _crop_head 开头插入 div 计算
            (
                r"(def _crop_head\(img, xywh\):\n)(            x, y, w, h = xywh\n)",
                r"\1" + SNIPPET.replace("\\", "\\\\") + r"\2",
                True,
            ),
            # 2) 横向外扩改成 1/div
            (
                r"px = min\(iw - x - w, x, w // 5\)",
                "px = min(iw - x - w, x, w // div)",
                True,
            ),
            # 3) 纵向外扩改成 1/div
            (
                r"py = min\(ih - y - h, y, h // 5\)",
                "py = min(ih - y - h, y, h // div)",
                True,
            ),
        ],
        label="INFERENCE_UTILS_HEAD_CROP",
    )
    if not ok:
        return 1
    print("\n完成。想回到上游行为：设 STG_HEAD_CROP=5 再跑即可，不用改代码。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
