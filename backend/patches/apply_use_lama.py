#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
apply_use_lama.py  —  SeeThrough-GGUF-Win 的 use_lama 补丁（幂等）

STM（IceFog72/see-through-gguf）的 post-processing 默认总是使用 LaMa 修复
（cluster_inpaint_part 的 inpaint='lama' 默认值），但它没有暴露成命令行开关。
本脚本对克隆下来的 STM 做最小、可逆的字符串补丁，新增 --use_lama / --no-use_lama
开关，使 Web 界面的“LaMa 修复”开关（默认开启）成为真实可控项。

- 幂等：已打过补丁则跳过。
- 安全：修改前为原文件写入 .bak 备份。
- 仅在 STM 仓库存在时工作；若仓库缺失则提示并退出 0（由 setup 负责克隆）。
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent   # backend/patches -> 项目根
STM = ROOT / "stm"

INFERENCE_UTILS = STM / "common" / "utils" / "inference_utils.py"
INFERENCE_PSDPY = STM / "inference" / "scripts" / "inference_psd.py"


def patch_file(path: Path, edits, label: str):
    if not path.exists():
        print(f"[跳过] 未找到 {path.name}（STM 仓库可能尚未克隆）")
        return True
    text = path.read_text(encoding="utf-8")
    marker = f"__STG_PATCHED_{label}__"
    # 先判幂等：打过补丁的文件里，部分原文已被替换，不能再按原文匹配
    if marker in text:
        print(f"[跳过] {path.name} 已打过补丁")
        return True
    backup = path.with_suffix(path.suffix + ".bak")
    changed = False
    for pattern, repl, must in edits:
        if re.search(pattern, text):
            new_text, n = re.subn(pattern, repl, text, count=1)
            if n:
                text = new_text
                changed = True
        else:
            if must:
                print(f"[错误] 在 {path.name} 中未匹配到预期代码，补丁可能不适用（STM 版本变更）")
                return False
    if changed:
        if not backup.exists():
            backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
        # 写入标记，便于幂等判断
        text = f"# __STG_PATCHED_{label}__\n" + text
        path.write_text(text, encoding="utf-8")
        print(f"[完成] 已为 {path.name} 应用补丁（备份: {backup.name}）")
    else:
        print(f"[跳过] {path.name} 已包含补丁或无需修改")
    return True


def main():
    print("== use_lama 补丁 ==")
    if not STM.exists():
        print(f"[提示] STM 仓库不存在（{STM}）。请先运行 1-setup.bat 克隆仓库，本补丁将跳过。")
        return 0

    ok1 = patch_file(
        INFERENCE_UTILS,
        [
            # 1) further_extr 增加 use_lama 形参（默认开启）
            (
                r"def further_extr\(srcd: str, rotate=True, save_to_psd=False, tblr_split=True,",
                "def further_extr(srcd: str, rotate=True, save_to_psd=False, tblr_split=True, use_lama=True,",
                True,
            ),
            # 2) hair 分支的 inpaint 调用按 use_lama 选择 lama / cv2
            (
                r"cluster_inpaint_part\(\*\*part_info\)",
                "cluster_inpaint_part(inpaint='lama' if use_lama else 'cv2', **part_info)",
                True,
            ),
        ],
        label="INFERENCE_UTILS",
    )

    ok2 = patch_file(
        INFERENCE_PSDPY,
        [
            # 3) 新增 --use_lama 开关（BooleanOptionalAction，默认 True）
            (
                r"(parser\.add_argument\('--tblr_split', action='store_true',[^\n]*\)\n)",
                r"\1    parser.add_argument('--use_lama', action=argparse.BooleanOptionalAction, default=True, help='use LaMa inpainting during post-processing (default on)')\n",
                True,
            ),
            # 4) 将 args.use_lama 传入 further_extr
            (
                r"tblr_split=args\.tblr_split,",
                "tblr_split=args.tblr_split, use_lama=args.use_lama,",
                True,
            ),
        ],
        label="INFERENCE_PSDPY",
    )

    if ok1 and ok2:
        print("[OK] use_lama 补丁处理完成。")
        return 0
    print("[失败] 补丁应用出现异常。")
    return 1


if __name__ == "__main__":
    sys.exit(main())
