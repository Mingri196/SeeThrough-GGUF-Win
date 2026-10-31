#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""后端链路集成测试（不需要 GPU / 不需要真实模型）。

用桩推理脚本替换 STM，验证真实代码路径：
上传校验 → 参数校验 → 子进程 → SSE 日志 → 输出扫描 → 文件下载 → 取消 → 并发保护

运行：.venv\\Scripts\\python.exe backend\\test_pipeline.py
"""

from __future__ import annotations

import io
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

PASS, FAIL = [], []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        PASS.append(name)
        print(f"  [PASS] {name}")
    else:
        FAIL.append(f"{name} {detail}".strip())
        print(f"  [FAIL] {name} {detail}")


STUB = r'''
import argparse, os, sys, time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--srcp")
parser.add_argument("--save_dir")
parser.add_argument("--gguf_layerdiff")
parser.add_argument("--gguf_depth")
parser.add_argument("--resolution", type=int, default=1024)
parser.add_argument("--inference_steps", type=int, default=30)
parser.add_argument("--inference_steps_depth", type=int, default=-1)
parser.add_argument("--resolution_depth", default="auto")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--attention-backend", default="native")
parser.add_argument("--disable_progressbar", action="store_true")
parser.add_argument("--quantized", action="store_true")
parser.add_argument("--tblr_split", action="store_true")
parser.add_argument("--save_to_psd", action="store_true")
parser.add_argument("--ragdoll_split", action="store_true")
parser.add_argument("--ragdoll-confidence", type=float, default=0.3)
parser.add_argument("--use_lama", action=argparse.BooleanOptionalAction, default=True)
args = parser.parse_args()

print("stub inference start", flush=True)
print(f"resolution={args.resolution} steps={args.inference_steps} seed={args.seed}", flush=True)
print(f"quantized={args.quantized} tblr={args.tblr_split} lama={args.use_lama} psd={args.save_to_psd}", flush=True)
print(f"compile_env={os.environ.get('SEETHROUGH_COMPILE_DEQUANT')}", flush=True)
print(f"depth_res={args.resolution_depth}", flush=True)

out = Path(args.save_dir)
out.mkdir(parents=True, exist_ok=True)
slow = os.environ.get("STG_STUB_SLOW") == "1"
if slow:
    for i in range(6):
        print(f"progress {i}/6", flush=True)
        time.sleep(0.35)

from PIL import Image
img = Image.new("RGBA", (64, 64), (200, 40, 20, 255))
img.save(out / "layer_body.png")
img.save(out / "layer_hair.png")

if args.save_to_psd:
    (out / "result.psd").write_bytes(b"8BPS-stub-main")
    (out / "result_depth.psd").write_bytes(b"8BPS-stub-depth")

print("stub inference done", flush=True)
'''


def build_env() -> Path:
    tmp = Path(tempfile.mkdtemp(prefix="stg-test-"))
    stm = tmp / "stm"
    (stm / "inference" / "scripts").mkdir(parents=True)
    (stm / "common" / "utils").mkdir(parents=True)
    (stm / "common" / "utils" / "inference_utils.py").write_text("# __STG_PATCHED_INFERENCE_UTILS__\n", encoding="utf-8")
    (stm / "inference" / "scripts" / "inference_psd.py").write_text(STUB, encoding="utf-8")

    models = tmp / "models"
    models.mkdir()
    (models / "layerdiff3d-Q5_1.gguf").write_bytes(b"GGUF" * 16)
    (models / "marigold3d-Q8_0.gguf").write_bytes(b"GGUF" * 16)
    # 嵌套子目录里的另一档量化，用于测试模型扫描与选择
    pack = models / "pack"
    pack.mkdir()
    (pack / "layerdiff3d-Q4_K_M.gguf").write_bytes(b"GGUF" * 8)
    (pack / "marigold3d-Q8_0.gguf").write_bytes(b"GGUF" * 8)

    (tmp / "outputs").mkdir()
    return tmp


def png_bytes() -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (96, 120), (10, 20, 30)).save(buf, format="PNG")
    return buf.getvalue()


def parse_sse(raw: str) -> dict:
    events = {"log": [], "done": None}
    for block in raw.split("\n\n"):
        lines = [ln for ln in block.splitlines() if ln.startswith(("event:", "data:"))]
        if not lines:
            continue
        event = "message"
        data = ""
        for ln in lines:
            if ln.startswith("event:"):
                event = ln[6:].strip()
            elif ln.startswith("data:"):
                data += ln[5:].strip()
        if not data:
            continue
        try:
            payload = json.loads(data)
        except json.JSONDecodeError:
            continue
        if event == "log":
            events["log"].append(payload)
        elif event == "done":
            events["done"] = payload
    return events


def main() -> int:
    tmp = build_env()

    import server  # noqa: E402  (路径打桩后再导入)

    server.STM_DIR = tmp / "stm"
    server.STM_UTILS = tmp / "stm" / "common" / "utils" / "inference_utils.py"
    server.INFERENCE_SCRIPT = tmp / "stm" / "inference" / "scripts" / "inference_psd.py"
    server.LAYERDIFF_GGUF = tmp / "models" / "layerdiff3d-Q5_1.gguf"
    server.DEPTH_GGUF = tmp / "models" / "marigold3d-Q8_0.gguf"
    server.MODELS_DIR = tmp / "models"
    server.OUTPUT_DIR = tmp / "outputs"
    server.JOBS_DIR = tmp / "outputs" / "jobs"
    server.JOBS_DIR.mkdir(parents=True, exist_ok=True)
    server.app.config["TESTING"] = True

    client = server.app.test_client()

    print("\n[1] 配置与模型检查")
    cfg = client.get("/api/config").get_json()
    check("config 返回约束", cfg["constraints"]["resolution_step"] == 64 and cfg["defaults"]["tblr_split"] is True)
    models = client.get("/api/check_models").get_json()
    check("模型检查 ready", models["ready"] is True)
    check("补丁状态为已应用", models["items"]["use_lama_patch"]["applied"] is True)

    listing = client.get("/api/models").get_json()
    rels = {m["rel"] for m in listing["models"]}
    check("模型扫描包含嵌套目录", {"pack/layerdiff3d-Q4_K_M.gguf", "pack/marigold3d-Q8_0.gguf"} <= rels, str(sorted(rels)))
    roles = {m["rel"]: m["role"] for m in listing["models"]}
    check("模型角色识别", roles.get("pack/layerdiff3d-Q4_K_M.gguf") == "layerdiff" and roles.get("pack/marigold3d-Q8_0.gguf") == "depth")

    print("\n[2] 上传与参数校验")
    resp = client.post(
        "/api/run",
        data={
            "image": (io.BytesIO(png_bytes()), "hero.png"),
            "params": json.dumps({"width": 100, "height": 300, "inference_steps": 0}),
        },
        content_type="multipart/form-data",
    )
    body = resp.get_json()
    fields = {e["field"] for e in body.get("details", [])}
    check("非法参数返回 400", resp.status_code == 400 and body["error"] == "invalid_params")
    check("非法参数定位到字段", {"width", "height", "inference_steps"} <= fields, str(fields))

    resp = client.post(
        "/api/run",
        data={"image": (io.BytesIO(b"not an image"), "x.txt"), "params": "{}"},
        content_type="multipart/form-data",
    )
    check("不支持扩展名被拦截", resp.status_code == 400 and resp.get_json()["error"] == "unsupported_format")

    resp = client.post(
        "/api/run",
        data={"image": (io.BytesIO(png_bytes()), "scan.tif"), "params": "{}"},
        content_type="multipart/form-data",
    )
    check("TIFF 不再被接受", resp.status_code == 400 and resp.get_json()["error"] == "unsupported_format")

    resp = client.post(
        "/api/run",
        data={"image": (io.BytesIO(b"\x89PNG\r\n\x1a\nbroken"), "broken.png"), "params": "{}"},
        content_type="multipart/form-data",
    )
    check("损坏图片被拦截", resp.status_code == 400 and resp.get_json()["error"] == "unsupported_format")

    resp = client.post(
        "/api/run",
        data={
            "image": (io.BytesIO(png_bytes()), "hero.png"),
            "params": json.dumps({"width": 512, "height": 512, "inference_steps": 8, "layerdiff_model": "../config.json"}),
        },
        content_type="multipart/form-data",
    )
    detail = (resp.get_json().get("details") or [{}])[0]
    check("模型路径越界被拒绝", resp.status_code == 400 and detail.get("code") == "outside_models", str(detail))

    resp = client.post(
        "/api/run",
        data={
            "image": (io.BytesIO(png_bytes()), "hero.png"),
            "params": json.dumps({"width": 512, "height": 512, "inference_steps": 8, "resolution_depth": 500}),
        },
        content_type="multipart/form-data",
    )
    detail = (resp.get_json().get("details") or [{}])[0]
    check(
        "深度分辨率非 64 倍数被拒绝",
        resp.status_code == 400 and detail.get("field") == "resolution_depth" and detail.get("code") == "not_multiple",
        str(detail),
    )

    print("\n[3] 完整推理链路（桩进程）")
    params = {
        "width": 512,
        "height": 512,
        "inference_steps": 12,
        "inference_steps_depth": -1,
        "seed_mode": "fixed",
        "seed": 1234,
        "tblr_split": True,
        "use_lama": True,
        "save_psd": True,
        "ragdoll_split": False,
        "ragdoll_confidence": 0.3,
        "quantized": True,
        "layerdiff_model": "pack/layerdiff3d-Q4_K_M.gguf",
        "depth_model": "pack/marigold3d-Q8_0.gguf",
        "resolution_depth": 512,
    }
    resp = client.post(
        "/api/run",
        data={"image": (io.BytesIO(png_bytes()), "hero.png"), "params": json.dumps(params)},
        content_type="multipart/form-data",
    )
    body = resp.get_json()
    check("提交成功", resp.status_code == 200 and body["ok"] is True, str(body))
    check("固定种子原样回传", body.get("seed_used") == 1234)
    job_id = body["job_id"]

    stream = client.get(f"/api/stream/{job_id}")
    events = parse_sse(stream.get_data(as_text=True))
    done = events["done"] or {}
    check("SSE 收到日志", len(events["log"]) > 3, f"{len(events['log'])} 行")
    check("SSE 收到 done 事件", done.get("status") == "completed", str(done.get("status")))
    check("耗时字段存在", isinstance(done.get("elapsed"), (int, float)))

    outputs = done.get("outputs", [])
    kinds = {o["kind"] for o in outputs}
    check("输出包含图片与 PSD", {"image", "psd_main", "psd_depth"} <= kinds, str(kinds))

    psd = next(o for o in outputs if o["kind"] == "psd_main")
    file_resp = client.get(psd["url"])
    check("文件接口可下载", file_resp.status_code == 200 and file_resp.data.startswith(b"8BPS"))
    file_resp.close()
    dl = client.get(psd["download_url"])
    check("下载带 attachment", "attachment" in (dl.headers.get("Content-Disposition") or ""))
    dl.close()  # Windows 上未关闭的响应会锁住文件，影响后续删除测试

    status = client.get(f"/api/status/{job_id}").get_json()
    check("状态接口可查询", status["status"] == "completed" and status["seed_used"] == 1234)
    check("日志带时间与级别", all("ts" in ln and "level" in ln for ln in status["log_tail"][:3]))

    check("关闭 torch.compile 反量化", any("compile_env=0" in ln["msg"] for ln in events["log"]))
    cmd_line = next((ln["msg"] for ln in events["log"] if ln["msg"].startswith("执行:")), "")
    check("命令行使用所选模型", "pack\\layerdiff3d-Q4_K_M.gguf" in cmd_line, cmd_line[:120])
    check("首轮加载模型", any("本轮加载模型" in ln["msg"] for ln in events["log"]), "")
    check("命令行带深度分辨率", "--resolution_depth 512" in cmd_line, cmd_line[:160])
    check("命令行显式带 --quantized", "--quantized" in cmd_line, cmd_line[:80])
    check("命令行带 --tblr_split", "--tblr_split" in cmd_line)
    check("命令行不带 --no-use_lama", "--no-use_lama" not in cmd_line)

    print("\n[4] 随机种子唯一性")
    seeds = []
    streams = []
    for _ in range(2):
        r = client.post(
            "/api/run",
            data={"image": (io.BytesIO(png_bytes()), "a.png"), "params": json.dumps({**params, "seed_mode": "random"})},
            content_type="multipart/form-data",
        ).get_json()
        seeds.append(r["seed_used"])
        streams.append(client.get(f"/api/stream/{r['job_id']}").get_data(as_text=True))
    check("随机种子不重复", seeds[0] != seeds[1], str(seeds))
    check("第二轮复用常驻模型", "复用显存中的模型" in streams[1], streams[1][:120])

    print("\n[5] PSD 导出失败与推理失败区分")
    r = client.post(
        "/api/run",
        data={
            "image": (io.BytesIO(png_bytes()), "a.png"),
            "params": json.dumps({**params, "save_psd": False}),
        },
        content_type="multipart/form-data",
    ).get_json()
    done2 = (parse_sse(client.get(f"/api/stream/{r['job_id']}").get_data(as_text=True))["done"] or {})
    check("无 PSD 时为 warning 而非失败", done2.get("status") in ("completed", "completed_with_warning"))

    print("\n[6] 模型缺失预检")
    server.LAYERDIFF_GGUF = tmp / "models" / "missing.gguf"
    # 不传模型选择，走 config 回落分支，才能触发「默认模型缺失」预检
    fallback_params = {**params, "layerdiff_model": None, "depth_model": None}
    resp = client.post(
        "/api/run",
        data={"image": (io.BytesIO(png_bytes()), "a.png"), "params": json.dumps(fallback_params)},
        content_type="multipart/form-data",
    )
    check("缺模型返回 400", resp.status_code == 400 and resp.get_json()["error"] == "models_missing")
    check("缺模型项明确", resp.get_json()["missing"] == ["layerdiff"], str(resp.get_json().get("missing")))
    server.LAYERDIFF_GGUF = tmp / "models" / "layerdiff3d-Q5_1.gguf"

    print("\n[7] 并发保护与取消")
    import os

    os.environ["STG_STUB_SLOW"] = "1"
    slow_params = {**params, "keep_model": False}   # 常驻进程的环境在启动时固定，这里用一次性进程
    r = client.post(
        "/api/run",
        data={"image": (io.BytesIO(png_bytes()), "a.png"), "params": json.dumps(slow_params)},
        content_type="multipart/form-data",
    ).get_json()
    job2 = r["job_id"]
    time.sleep(0.4)
    busy = client.post(
        "/api/run",
        data={"image": (io.BytesIO(png_bytes()), "a.png"), "params": json.dumps(slow_params)},
        content_type="multipart/form-data",
    )
    check("并发提交被拒绝", busy.status_code == 409 and busy.get_json()["error"] == "busy")

    cancel = client.post(f"/api/jobs/{job2}/cancel")
    check("取消请求被接受", cancel.status_code == 200)
    done3 = (parse_sse(client.get(f"/api/stream/{job2}").get_data(as_text=True))["done"] or {})
    check("取消后状态为 canceled", done3.get("status") == "canceled", str(done3.get("status")))
    os.environ.pop("STG_STUB_SLOW", None)

    print("\n[8] 清理")
    deleted = client.delete(f"/api/jobs/{job_id}")
    left = (server.JOBS_DIR / job_id).exists()
    leftover = [str(p.relative_to(server.JOBS_DIR / job_id)) for p in (server.JOBS_DIR / job_id).rglob("*")] if left else []
    check("任务目录可清理", deleted.status_code == 200 and not left, f"status={deleted.status_code} left={left} {leftover[:6]}")

    server.JOBS.clear()
    shutil.rmtree(tmp, ignore_errors=True)

    print("\n" + "=" * 56)
    print(f"通过 {len(PASS)} 项，失败 {len(FAIL)} 项")
    if FAIL:
        for item in FAIL:
            print(" - " + item)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
