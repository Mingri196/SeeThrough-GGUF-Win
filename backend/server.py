#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SeeThrough-GGUF-Win 后端服务
============================

职责
----
1. 提供静态前端页面
2. 校验上传图片 + 参数（与前端共用同一套字段名与约束）
3. 以子进程方式调用 STM 的 inference_psd.py
4. 通过 SSE 推送结构化日志（时间 + 级别）
5. 扫描真实输出目录，返回可预览图片与可下载 PSD
6. 支持取消任务与清理临时文件

设计要点
--------
* 任务元数据写入 outputs/jobs/<id>/meta.json，服务重启后仍可查询与下载
* 同一时间只允许一个推理任务（单卡机器上并行只会互相抢占显存）
* GGUF 必须显式传 --quantized，否则 STM 会关闭流式加载并把整份权重反量化
  （inference_utils.py 中 `if not quantized: stream = False`），这正是
  6 GB 显存 / 16 GB 内存设备上显存爆掉的根因
"""

from __future__ import annotations

import io
import json
import os
import platform
import queue
import random
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

from flask import Flask, Response, abort, jsonify, request, send_from_directory

# ---------------------------------------------------------------------------
# 路径与配置
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
FRONTEND = ROOT / "frontend"
# STG_CONFIG 仅用于测试/多实例：指向另一份 config.json
CONFIG_PATH = Path(os.environ.get("STG_CONFIG") or (ROOT / "config.json"))
if not CONFIG_PATH.is_absolute():
    CONFIG_PATH = ROOT / CONFIG_PATH


def _resolve(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (ROOT / p)


DEFAULT_CONFIG = {
    "host": "127.0.0.1",
    "port": 7860,
    "stm_path": "stm",
    "output_dir": "outputs",
    "layerdiff_gguf": "models/layerdiff3d-Q5_1.gguf",
    "depth_gguf": "models/marigold3d-Q8_0.gguf",
    "attention_backend": "native",
    # "auto" = 启动时自动探测 huggingface.co / hf-mirror.com 并选更合适的；也可写死某个 URL
    "hf_endpoint": "auto",
    # STM 默认用 torch.compile 编译反量化核；Windows 上 inductor 需要 MSVC cl，通常不可用
    "compile_dequant": False,
    "max_upload_mb": 50,
    "defaults": {
        "width": 1024,
        "height": 1024,
        "inference_steps": 30,
        "inference_steps_depth": 4,
        "mask_resolution": "auto",
        "seed": 42,
        "tblr_split": True,
        "use_lama": True,
        "soft_alpha": True,
        "edge_refine": False,
        "save_psd": True,
        "ragdoll_split": False,
        "ragdoll_confidence": 0.3,
        "ragdoll_overlap": 0.2,
        "group_offload": False,
        "save_cut_images": True,
        "layer_fix": True,
        "parts": None,
    },
    "constraints": {
        "resolution_min": 256,
        "resolution_max": 2048,
        "resolution_step": 64,
        "steps_min": 1,
        "steps_max": 150,
        "depth_steps_max": 100,
        "mask_resolution_min": 256,
        "mask_resolution_max": 4096,
        "seed_max": 2147483647,
        "ragdoll_confidence_min": 0.05,
        "ragdoll_confidence_max": 0.95,
        "ragdoll_overlap_min": 0.0,
        "ragdoll_overlap_max": 0.5,
    },
}

CONFIG = json.loads(json.dumps(DEFAULT_CONFIG))  # deep copy
if CONFIG_PATH.exists():
    try:
        user_cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        for key, value in user_cfg.items():
            if isinstance(value, dict) and isinstance(CONFIG.get(key), dict):
                CONFIG[key].update(value)
            else:
                CONFIG[key] = value
    except Exception as exc:  # 配置损坏不致命
        print(f"[WARN] config.json 解析失败，使用默认配置: {exc}")

STM_DIR = _resolve(CONFIG["stm_path"])
INFERENCE_SCRIPT = STM_DIR / "inference" / "scripts" / "inference_psd.py"
POSTPROCESS_SCRIPT = STM_DIR / "inference" / "scripts" / "ragdoll_postprocess.py"
LAYERDIFF_GGUF = _resolve(CONFIG["layerdiff_gguf"])
DEPTH_GGUF = _resolve(CONFIG["depth_gguf"])
MODELS_DIR = _resolve(CONFIG.get("models_dir", "models"))
OUTPUT_DIR = _resolve(CONFIG["output_dir"])
JOBS_DIR = OUTPUT_DIR / "jobs"

# 组件缓存：默认项目内 components/（config.json 的 components_dir 可改成 "user"）
sys.path.insert(0, str(Path(__file__).resolve().parent))
import components_paths  # noqa: E402

try:
    components_paths.ensure_dirs()
except OSError as exc:
    print(f"[WARN] 组件目录创建失败: {exc}")

ALLOWED_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

# STM 的图层 tag（与 stm/common/utils/inference_utils.py 保持一致，顺序即上游顺序）
BODY_PARTS = (
    "front hair", "back hair", "head", "neck", "neckwear", "topwear", "handwear",
    "bottomwear", "legwear", "footwear", "tail", "wings", "objects",
)
HEAD_PARTS = (
    "headwear", "face", "irides", "eyebrow", "eyewhite", "eyelash", "eyewear",
    "ears", "earwear", "nose", "mouth",
)
PART_ALIASES = {"hair": ("front hair", "back hair"), "eyes": ("irides", "eyebrow", "eyewhite", "eyelash")}
ALL_PARTS = BODY_PARTS + HEAD_PARTS

# 注意力后端：native = PyTorch 自带 SDPA（零依赖）；xformers 需要额外安装
ATTENTION_BACKENDS = ("native", "xformers")


def detect_attention_backends() -> dict:
    """只查包有没有装，不真正 import（避免几秒的加载开销）。"""
    import importlib.util

    available = {"native": True}
    try:
        available["xformers"] = importlib.util.find_spec("xformers") is not None
    except (ImportError, ValueError):
        available["xformers"] = False
    return available


ATTENTION_AVAILABLE = detect_attention_backends()

# 进程启动时刻：设置页「启动与端口」「维护与诊断」显示运行时长
SERVER_STARTED_AT = time.time()

# pose_keypoints.json 里前 23 个点是 RTMPose wholebody 的身体+足部，骨骼切割只看这些
BODY_KEYPOINTS = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
    "left_big_toe", "left_small_toe", "left_heel",
    "right_big_toe", "right_small_toe", "right_heel",
)

# 启动时写入；/api/check_models 会把它带给前端
NETWORK_INFO: dict | None = None

# use_lama 补丁是否已应用到 STM 源码
PATCH_MARKER = "__STG_PATCHED_INFERENCE_UTILS__"
STM_UTILS = STM_DIR / "common" / "utils" / "inference_utils.py"

app = Flask(__name__, static_folder=str(FRONTEND), static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = int(CONFIG["max_upload_mb"]) * 1024 * 1024
app.json.ensure_ascii = False


# ---------------------------------------------------------------------------
# 任务
# ---------------------------------------------------------------------------
_ERROR_TEMPLATES = ("推理进程异常退出",)

_ERROR_KEYWORDS = (
    "out of memory",
    "outofmemory",
    "not enough memory",
    "cuda error",
    "no kernel image",
    "illegal memory access",
    "device-side assert",
)


def _format_error_text(msg: str | None) -> str:
    """把固定模板的报错收敛成有信息的形式：只留退出码。

    「推理进程异常退出（退出码 2）」→「退出码 2」；
    其它报错原样返回，避免误伤 OOM 这类本来就有用的长文本。
    """
    text = str(msg or "").strip()
    if not text:
        return ""
    if any(t in text for t in _ERROR_TEMPLATES):
        found = re.search(r"退出码\s*(-?\d+)", text)
        # 有退出码就只留数字；连退出码都没有的纯模板没有信息量，留空
        return f"退出码 {found.group(1)}" if found else ""
    return text


def _error_hint(history: list[dict]) -> str | None:
    """从任务日志里挑一条最像“根因”的错误，供历史记录展示。

    优先级：显存/内核这类关键词全文 → 退出码（只留数字）→ 最后一条真实报错。
    「推理进程异常退出…」是固定模板，本身不含信息，不直接展示。
    """
    msgs = [str(e.get("msg") or "").strip() for e in history if e.get("level") == "error"]
    msgs = [m for m in msgs if m]
    if not msgs:
        return None

    for msg in msgs:
        low = msg.lower()
        if any(key in low for key in _ERROR_KEYWORDS):
            return msg[:200]

    # 进程非零退出：退出码本身就是结论，去掉模板前缀
    for msg in reversed(msgs):
        formatted = _format_error_text(msg)
        # 纯模板收敛成空串，跳过继续往上找
        if formatted and formatted != msg.strip():
            return formatted

    real = [m for m in msgs if not any(t in m for t in _ERROR_TEMPLATES)]
    return real[-1][:200] if real else None


class Job:
    """一次推理任务。日志同时保存在内存队列（供 SSE）与 meta.json（供重启后查询）。"""

    def __init__(self, job_id: str, job_dir: Path, params: dict, seed_used: int):
        self.job_id = job_id
        self.job_dir = job_dir
        self.out_dir = job_dir / "out"
        self.params = params
        self.seed_used = seed_used
        self.status = "queued"          # queued | running | completed | completed_with_warning | failed | canceled
        self.error = None               # 机器可读错误码
        self.error_detail = None        # 服务端原始报错（只进日志/详情，不直接进横幅）
        self.outputs: list[dict] = []
        self.q: queue.Queue = queue.Queue()
        self.history: list[dict] = []
        self.seq = 0
        self.process: subprocess.Popen | None = None
        self.canceled = False
        self.done = threading.Event()
        self.started_at = time.time()
        self.finished_at: float | None = None

    # -- 日志 ---------------------------------------------------------------
    def log(self, msg: str, level: str = "info") -> None:
        self.seq += 1
        entry = {"seq": self.seq, "ts": int(time.time() * 1000), "level": level, "msg": msg}
        self.history.append(entry)
        if len(self.history) > 4000:                      # 极端长日志的兜底
            self.history = self.history[-3000:]
        self.q.put(entry)

    def close_stream(self) -> None:
        self.q.put(None)                                  # SSE 结束哨兵

    # -- 落盘 ---------------------------------------------------------------
    def to_meta(self) -> dict:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "error": self.error,
            "error_detail": self.error_detail,
            "error_hint": _error_hint(self.history),
            "params": self.params,
            "seed_used": self.seed_used,
            "outputs": self.outputs,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "canceled": self.canceled,
        }

    def save_meta(self) -> None:
        try:
            (self.job_dir / "meta.json").write_text(
                json.dumps(self.to_meta(), ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except OSError as exc:
            print(f"[WARN] 写入 meta.json 失败: {exc}")


JOBS: dict[str, Job] = {}
JOBS_LOCK = threading.Lock()
LAST_SEED: int | None = None
SEED_LOCK = threading.Lock()


def _ensure_job_dir(job_id: str) -> Path:
    d = JOBS_DIR / job_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_job(job_id: str) -> Job | None:
    """内存优先；服务重启后从 meta.json 还原，避免下载链接失效。"""
    job = JOBS.get(job_id)
    if job:
        return job
    meta_path = JOBS_DIR / job_id / "meta.json"
    if not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    job = Job(job_id, JOBS_DIR / job_id, meta.get("params", {}), meta.get("seed_used", 0))
    job.status = meta.get("status", "completed")
    job.error = meta.get("error")
    job.error_detail = meta.get("error_detail")
    job.outputs = meta.get("outputs", [])
    job.started_at = meta.get("started_at", job.started_at)
    job.finished_at = meta.get("finished_at")
    job.canceled = meta.get("canceled", False)
    job.done.set()
    return job


def active_job() -> Job | None:
    for job in list(JOBS.values()):
        if job.status in ("queued", "running"):
            return job
    return None


def next_seed() -> int:
    """随机模式：保证与上一次不同（连续两次随机不重复）。"""
    global LAST_SEED
    limit = int(CONFIG["constraints"]["seed_max"])
    with SEED_LOCK:
        while True:
            value = random.randint(0, limit)
            if value != LAST_SEED:
                LAST_SEED = value
                return value


# ---------------------------------------------------------------------------
# 参数校验（与前端共用字段名）
# ---------------------------------------------------------------------------
def _as_int(value):
    if isinstance(value, bool) or value is None or value == "":
        raise ValueError
    if isinstance(value, float):
        if not value.is_integer():
            raise ValueError
        return int(value)
    return int(str(value).strip())


def _as_bool(value, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def _clean_parts(value, err) -> list[str] | None:
    """规范化 --parts 列表；返回 None 表示全量（交给上游默认）。

    接受数组，也接受 JSON 字符串或逗号分隔串（curl 手打时方便）。
    别名 hair / eyes 按上游规则展开。
    """
    if value is None or value == "" or value == []:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            parsed = json.loads(text)
        except Exception:
            parsed = None
        if isinstance(parsed, list):
            value = parsed
        else:
            value = [chunk.strip() for chunk in text.split(",") if chunk.strip()]
    if not isinstance(value, (list, tuple)):
        err("parts", "not_list")
        return None

    picked: list[str] = []
    unknown: list[str] = []
    for item in value:
        if not isinstance(item, str):
            unknown.append(str(item))
            continue
        key = item.strip()
        if not key:
            continue
        for tag in PART_ALIASES.get(key, (key,)):
            if tag in ALL_PARTS:
                if tag not in picked:
                    picked.append(tag)
            elif key not in unknown:
                unknown.append(key)

    if unknown:
        err("parts", "unknown", names=", ".join(sorted(unknown)[:5]))
        return None
    if not picked:
        err("parts", "empty")
        return None
    ordered = [tag for tag in ALL_PARTS if tag in picked]
    return None if len(ordered) == len(ALL_PARTS) else ordered


def validate_params(raw: dict, patch_ok: bool) -> tuple[dict, list[dict]]:
    """返回 (规范化参数, 字段级错误列表)。错误码由前端本地化，便于中英一致。"""
    c = CONFIG["constraints"]
    d = CONFIG["defaults"]
    errors: list[dict] = []
    clean: dict = {}

    def err(field: str, code: str, **extra):
        errors.append({"field": field, "code": code, **extra})

    # --- 宽高（引擎只支持正方形输出，宽高必须相等） ---
    for field in ("width", "height"):
        try:
            value = _as_int(raw.get(field, d[field]))
        except (TypeError, ValueError):
            err(field, "not_integer")
            clean[field] = None
            continue
        if value < c["resolution_min"] or value > c["resolution_max"]:
            err(field, "out_of_range", min=c["resolution_min"], max=c["resolution_max"])
        elif value % c["resolution_step"] != 0:
            err(field, "not_multiple", step=c["resolution_step"])
        clean[field] = value

    if (
        isinstance(clean.get("width"), int)
        and isinstance(clean.get("height"), int)
        and clean["width"] != clean["height"]
    ):
        err("height", "must_equal_width")

    # --- 推理步数 ---
    try:
        steps = _as_int(raw.get("inference_steps", d["inference_steps"]))
        if not (c["steps_min"] <= steps <= c["steps_max"]):
            err("inference_steps", "out_of_range", min=c["steps_min"], max=c["steps_max"])
        clean["inference_steps"] = steps
    except (TypeError, ValueError):
        err("inference_steps", "not_integer")
        clean["inference_steps"] = None

    # --- 深度步数（-1 = 自动） ---
    try:
        depth_steps = _as_int(raw.get("inference_steps_depth", d["inference_steps_depth"]))
        if depth_steps != -1 and not (1 <= depth_steps <= c["depth_steps_max"]):
            err("inference_steps_depth", "out_of_range", min=1, max=c["depth_steps_max"], allow_auto=True)
        clean["inference_steps_depth"] = depth_steps
    except (TypeError, ValueError):
        err("inference_steps_depth", "not_integer")
        clean["inference_steps_depth"] = None

    # --- 深度分辨率（auto / -1 跟随主分辨率 / 正整数） ---
    raw_depth_res = raw.get("resolution_depth", "auto")
    if raw_depth_res in (None, "", "auto", "Auto", "AUTO"):
        clean["resolution_depth"] = "auto"
    elif str(raw_depth_res).strip() == "-1":
        clean["resolution_depth"] = -1
    else:
        try:
            depth_res = _as_int(raw_depth_res)
        except (TypeError, ValueError):
            err("resolution_depth", "not_integer")
            clean["resolution_depth"] = None
        else:
            if depth_res < c["resolution_min"] or depth_res > c["resolution_max"]:
                err("resolution_depth", "out_of_range", min=c["resolution_min"], max=c["resolution_max"])
            elif depth_res % c["resolution_step"] != 0:
                err("resolution_depth", "not_multiple", step=c["resolution_step"])
            clean["resolution_depth"] = depth_res

    # --- 前景分割分辨率（只对没有 alpha 的图生效；auto / 正整数） ---
    raw_mask_res = raw.get("mask_resolution", d.get("mask_resolution", "auto"))
    if raw_mask_res in (None, "", "auto", "Auto", "AUTO"):
        clean["mask_resolution"] = "auto"
    else:
        try:
            mask_res = _as_int(raw_mask_res)
        except (TypeError, ValueError):
            err("mask_resolution", "not_integer")
            clean["mask_resolution"] = None
        else:
            if mask_res < c["mask_resolution_min"] or mask_res > c["mask_resolution_max"]:
                err(
                    "mask_resolution",
                    "out_of_range",
                    min=c["mask_resolution_min"],
                    max=c["mask_resolution_max"],
                )
            clean["mask_resolution"] = mask_res

    # --- 注意力后端（native / xformers；xformers 需已安装） ---
    backend = str(raw.get("attention_backend") or CONFIG.get("attention_backend", "native")).strip().lower()
    if backend not in ATTENTION_BACKENDS:
        err("attention_backend", "invalid_value", allowed=list(ATTENTION_BACKENDS))
        clean["attention_backend"] = "native"
    elif not ATTENTION_AVAILABLE.get(backend, False):
        err(
            "attention_backend",
            "not_available",
            available=[k for k, v in ATTENTION_AVAILABLE.items() if v],
        )
        clean["attention_backend"] = "native"
    else:
        clean["attention_backend"] = backend

    # --- 种子 ---
    mode = raw.get("seed_mode", "fixed")
    if mode not in ("fixed", "random"):
        err("seed_mode", "invalid_option")
        mode = "fixed"
    clean["seed_mode"] = mode
    if mode == "fixed":
        try:
            seed = _as_int(raw.get("seed", d["seed"]))
            if not (0 <= seed <= c["seed_max"]):
                err("seed", "out_of_range", min=0, max=c["seed_max"])
            clean["seed"] = seed
        except (TypeError, ValueError):
            err("seed", "not_integer")
            clean["seed"] = None

    # --- 开关 ---
    clean["tblr_split"] = _as_bool(raw.get("tblr_split"), d["tblr_split"])
    clean["use_lama"] = _as_bool(raw.get("use_lama"), d["use_lama"])
    clean["soft_alpha"] = _as_bool(raw.get("soft_alpha"), bool(d.get("soft_alpha", True)))
    clean["edge_refine"] = _as_bool(raw.get("edge_refine"), bool(d.get("edge_refine", False)))
    # 从属开关以主开关为前提：主关时强制关闭，避免旧记忆或手改请求让它白跑一次模型
    if not clean["soft_alpha"]:
        clean["edge_refine"] = False
    clean["save_psd"] = _as_bool(raw.get("save_psd"), d["save_psd"])
    clean["ragdoll_split"] = _as_bool(raw.get("ragdoll_split"), d["ragdoll_split"])
    clean["quantized"] = _as_bool(raw.get("quantized"), True)
    clean["keep_model"] = _as_bool(raw.get("keep_model"), True)
    clean["group_offload"] = _as_bool(raw.get("group_offload"), bool(d.get("group_offload", False)))
    clean["save_cut_images"] = _as_bool(raw.get("save_cut_images"), bool(d.get("save_cut_images", True)))
    clean["layer_fix"] = _as_bool(raw.get("layer_fix"), bool(d.get("layer_fix", True)))

    clean["parts"] = _clean_parts(raw.get("parts", d.get("parts")), err)

    if clean["use_lama"] is False and not patch_ok:
        err("use_lama", "patch_missing")

    try:
        conf = float(raw.get("ragdoll_confidence", d["ragdoll_confidence"]))
        if not (c["ragdoll_confidence_min"] <= conf <= c["ragdoll_confidence_max"]):
            err(
                "ragdoll_confidence",
                "out_of_range",
                min=c["ragdoll_confidence_min"],
                max=c["ragdoll_confidence_max"],
                step=0.05,
            )
        clean["ragdoll_confidence"] = round(conf, 2)
    except (TypeError, ValueError):
        err("ragdoll_confidence", "not_number")
        clean["ragdoll_confidence"] = None

    try:
        overlap = float(raw.get("ragdoll_overlap", d["ragdoll_overlap"]))
        if not (c["ragdoll_overlap_min"] <= overlap <= c["ragdoll_overlap_max"]):
            err(
                "ragdoll_overlap",
                "out_of_range",
                min=c["ragdoll_overlap_min"],
                max=c["ragdoll_overlap_max"],
                step=0.05,
            )
        clean["ragdoll_overlap"] = round(overlap, 2)
    except (TypeError, ValueError):
        err("ragdoll_overlap", "not_number")
        clean["ragdoll_overlap"] = None

    return clean, errors


def resolve_hf_endpoint() -> str | None:
    """决定运行期下载组件用哪个源：显式配置优先，否则用启动时探测的结果。"""
    configured = str(CONFIG.get("hf_endpoint") or "").strip()
    if configured and configured.lower() != "auto":
        return configured
    if NETWORK_INFO and NETWORK_INFO.get("endpoint"):
        return str(NETWORK_INFO["endpoint"])
    return None


def guess_model_role(filename: str) -> str:
    name = filename.lower()
    # 深度模型先判：原版仓库名是 seethroughv0.0.1_marigold，也带 seethrough，
    # 若先匹配 layerdiff 会被错当成图层模型
    if "marigold" in name or "depth" in name:
        return "depth"
    if "layerdiff" in name or "seethrough" in name:
        return "layerdiff"
    return "unknown"


def _dir_size(path: Path) -> int:
    """目录总字节数（只 stat，不读内容）。"""
    total = 0
    try:
        for child in path.rglob("*"):
            try:
                if child.is_file():
                    total += child.stat().st_size
            except OSError:
                continue
    except OSError:
        pass
    return total


def is_diffusers_dir(path: Path) -> bool:
    """目录形态的模型：有 model_index.json 才算一个 diffusers 仓库。"""
    return path.is_dir() and (path / "model_index.json").is_file()


def scan_models() -> dict:
    """扫描 models/ 下的 GGUF 单文件与 diffusers 目录（原版 safetensors）。

    目录形态的判定只看 model_index.json，所以模型放在 models/ 里或再套一层
    子目录都能识别；目录内部不再继续下钻，避免把 unet/、vae/ 当模型。
    """
    items: list[dict] = []
    base = MODELS_DIR.resolve()
    if MODELS_DIR.exists():
        for path in sorted(MODELS_DIR.rglob("*.gguf")):
            if not path.is_file():
                continue
            try:
                rel = path.relative_to(MODELS_DIR).as_posix()
                size = path.stat().st_size
            except OSError:
                continue
            items.append(
                {
                    "rel": rel,
                    "name": path.name,
                    "size": size,
                    "role": guess_model_role(path.name),
                    "kind": "gguf",
                }
            )

        seen_dirs: set[Path] = set()
        for idx in sorted(MODELS_DIR.rglob("model_index.json")):
            model_dir = idx.parent
            if model_dir in seen_dirs:
                continue
            seen_dirs.add(model_dir)
            try:
                rel = model_dir.relative_to(MODELS_DIR).as_posix()
            except ValueError:
                continue
            if rel in (".", ""):
                continue
            items.append(
                {
                    "rel": rel,
                    "name": model_dir.name,
                    "size": _dir_size(model_dir),
                    "role": guess_model_role(model_dir.name),
                    "kind": "dir",
                }
            )

    defaults: dict[str, str | None] = {"layerdiff": None, "depth": None}
    for role, key in (("layerdiff", "layerdiff_gguf"), ("depth", "depth_gguf")):
        candidate = _resolve(CONFIG[key]).resolve()
        try:
            if base == candidate.parent or base in candidate.parents:
                defaults[role] = candidate.relative_to(base).as_posix()
        except OSError:
            pass

    return {"models_dir": str(MODELS_DIR), "models": items, "defaults": defaults}


def resolve_model_choice(value: str, field: str) -> tuple[str | None, dict | None]:
    """把前端传来的相对路径限制在 models/ 目录内，避免任意文件读取。

    两种形态都接受：GGUF 单文件，或含 model_index.json 的 diffusers 目录。
    """
    if not isinstance(value, str) or not value.strip():
        return None, {"field": field, "code": "not_selected"}
    base = MODELS_DIR.resolve()
    target = (MODELS_DIR / value.strip()).resolve()
    if base != target and base not in target.parents:
        return None, {"field": field, "code": "outside_models"}
    if target.is_dir():
        if not is_diffusers_dir(target):
            return None, {"field": field, "code": "not_diffusers_dir"}
        return str(target), None
    if target.suffix.lower() != ".gguf":
        return None, {"field": field, "code": "not_gguf"}
    if not target.is_file():
        return None, {"field": field, "code": "missing"}
    return str(target), None


def build_command(job: Job, input_path: Path) -> list[str]:
    p = job.params
    layer_path = str(p.get("layerdiff_path") or LAYERDIFF_GGUF)
    depth_path = str(p.get("depth_path") or DEPTH_GGUF)
    # 单个 .gguf → 走量化加载；目录（含 model_index.json）→ 走原生 diffusers
    layer_is_gguf = layer_path.lower().endswith(".gguf")
    depth_is_gguf = depth_path.lower().endswith(".gguf")

    args = [
        sys.executable,
        str(INFERENCE_SCRIPT),
        "--srcp", str(input_path),
        "--save_dir", str(job.out_dir),
        ("--gguf_layerdiff" if layer_is_gguf else "--repo_id_layerdiff"), layer_path,
        ("--gguf_depth" if depth_is_gguf else "--repo_id_depth"), depth_path,
        "--resolution", str(p["width"]),
        "--inference_steps", str(p["inference_steps"]),
        "--seed", str(job.seed_used),
        "--attention-backend", str(p.get("attention_backend") or CONFIG.get("attention_backend", "native")),
        # 不传 --disable_progressbar：采样期间 tqdm 会输出步进，供日志展示
    ]
    if p["quantized"] and layer_is_gguf:
        # 关键：不加这个，STM 会关闭流式加载（inference_utils: if not quantized: stream = False）
        # 目录形态没有量化权重，传了只会让下游走无关分支
        args.append("--quantized")
    if p["inference_steps_depth"] != -1:
        args += ["--inference_steps_depth", str(p["inference_steps_depth"])]
    depth_res = p.get("resolution_depth", "auto")
    if depth_res not in (None, "auto"):
        args += ["--resolution_depth", str(depth_res)]
    mask_res = p.get("mask_resolution", "auto")
    if mask_res not in (None, "auto"):
        args += ["--mask-resolution", str(mask_res)]
    if p["tblr_split"]:
        args.append("--tblr_split")
    if p["save_psd"]:
        args.append("--save_to_psd")
    if p["ragdoll_split"]:
        args += [
            "--ragdoll_split",
            "--ragdoll-confidence", f"{p['ragdoll_confidence']:.2f}",
            "--ragdoll-overlap", f"{p['ragdoll_overlap']:.2f}",
        ]
    if not p["use_lama"]:
        args.append("--no-use_lama")
    if not p.get("soft_alpha", True):
        # BooleanOptionalAction 的否定形式跟随参数名本身（soft_alpha），
        # 生成的是 --no-soft_alpha；写成连字符会被 argparse 判为未知参数
        args.append("--no-soft_alpha")
    if p.get("edge_refine"):
        args.append("--edge_refine")
    if p.get("save_cut_images"):
        args.append("--save_cut_images")
    if p.get("group_offload"):
        args.append("--group_offload")
    if p.get("parts"):
        args += ["--parts", *[str(tag) for tag in p["parts"]]]
    return args


# ---------------------------------------------------------------------------
# 输出扫描
# ---------------------------------------------------------------------------
def scan_outputs(job: Job) -> list[dict]:
    items: list[dict] = []
    if not job.out_dir.exists():
        return items
    for f in sorted(job.out_dir.rglob("*")):
        if not f.is_file():
            continue
        ext = f.suffix.lower()
        if ext not in (".png", ".jpg", ".jpeg", ".webp", ".psd", ".json"):
            continue
        name = f.name
        if ext == ".psd":
            kind = "psd_depth" if name.endswith("_depth.psd") else "psd_main"
        elif ext == ".json":
            kind = "json"
        else:
            kind = "image"
        rel = f.relative_to(job.out_dir).as_posix()
        items.append(
            {
                "name": name,
                "rel": rel,
                "kind": kind,
                "size": f.stat().st_size,
                "url": f"/api/file/{job.job_id}/{rel}",
                "download_url": f"/api/file/{job.job_id}/{rel}?download=1",
            }
        )
    return items


# ---------------------------------------------------------------------------
# 推理线程
# ---------------------------------------------------------------------------
LEVEL_PATTERNS = (
    ("error", ("traceback", "error:", "[error]", "exception", "out of memory", "cuda error")),
    ("warn", ("warning", "[warn]", "deprecat")),
)


def classify_line(line: str) -> str:
    low = line.lower()
    for level, needles in LEVEL_PATTERNS:
        if any(n in low for n in needles):
            return level
    if "%|" in line or "it/s" in low or "s/it" in low:
        return "progress"
    return "info"


PROGRESS_RE = re.compile(r"(\d+)%\|[^|]*\|\s*(\d+)/(\d+)")
RATE_RE = re.compile(r"([\d.]+\s*(?:s/it|it/s))")


def parse_progress(line: str) -> tuple[int, int, int] | None:
    """从 tqdm 输出里取出 (百分比, 当前步, 总步数)。"""
    match = PROGRESS_RE.search(line)
    if not match:
        return None
    pct, step, total = (int(match.group(i)) for i in (1, 2, 3))
    return pct, step, total


def format_progress(pct: int, step: int, total: int, line: str, segment: int) -> str:
    rate = RATE_RE.search(line)
    suffix = f" · {rate.group(1).strip()}" if rate else ""
    prefix = f"第{segment}段 " if segment > 1 else ""
    return f"{prefix}进度 {step}/{total} · {pct}%{suffix}"


def _kill_tree(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    else:
        proc.terminate()


WORKER_READY = "@@STG_WORKER_READY@@"
WORKER_DONE = "@@STG_DONE@@"
WORKER: subprocess.Popen | None = None
WORKER_KEY: tuple | None = None
WORKER_LOCK = threading.Lock()


def _worker_start(env: dict) -> subprocess.Popen:
    proc = subprocess.Popen(
        [sys.executable, "-u", str(ROOT / "backend" / "worker.py")],
        cwd=str(ROOT),
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert proc.stdout is not None
    head = proc.stdout.readline() or ""
    if WORKER_READY not in head:
        _kill_tree(proc)
        raise RuntimeError((head.strip() or "worker 启动无响应")[:160])
    return proc


def _worker_stop() -> None:
    global WORKER, WORKER_KEY
    proc, WORKER, WORKER_KEY = WORKER, None, None
    if proc is None or proc.poll() is not None:
        return
    try:
        if proc.stdin:
            proc.stdin.write('{"cmd": "exit"}\n')
            proc.stdin.flush()
        proc.wait(timeout=5)
    except Exception:
        _kill_tree(proc)


def _run_via_worker(job: Job, argv: list[str], env: dict) -> int | None:
    """在常驻进程里跑一轮；返回码，或 None 表示需要回退到一次性进程。"""
    global WORKER, WORKER_KEY
    p = job.params
    key = (
        p.get("layerdiff_path"),
        p.get("depth_path"),
        int(p.get("width") or 0),
        bool(p.get("quantized")),
    )

    with WORKER_LOCK:
        if WORKER is not None and WORKER.poll() is not None:
            WORKER = None
            WORKER_KEY = None
        cold = WORKER is None
        if cold:
            try:
                WORKER = _worker_start(env)
                WORKER_KEY = None
            except Exception as exc:
                WORKER = None
                job.log(f"常驻进程启动失败，改用一次性进程：{exc}", "warn")
                return None

        reset = WORKER_KEY != key
        if cold:
            job.log("模型常驻：本轮加载模型，下一轮起免加载", "sys")
        elif reset:
            job.log("模型设置变化，常驻进程将重新加载模型", "sys")
        else:
            job.log("模型常驻：复用显存中的模型，跳过加载", "sys")

        try:
            WORKER.stdin.write(
                json.dumps({"argv": argv, "reset": reset, "layer_fix": bool(p.get("layer_fix", True))}) + "\n"
            )
            WORKER.stdin.flush()
        except Exception as exc:
            job.log(f"常驻进程通信失败，改用一次性进程：{exc}", "warn")
            _worker_stop()
            return None

        WORKER_KEY = key
        job.process = WORKER          # 取消时按进程树结束
        assert WORKER.stdout is not None
        rc = _consume_output(job, WORKER.stdout, stop_on_done=True)
        job.process = None

    if rc is None:
        _worker_stop()
        if job.canceled:
            return 1
        job.log("常驻进程意外退出，本轮改用一次性进程重跑", "warn")
        return None
    return rc


def _consume_output(job: Job, stream, stop_on_done: bool) -> int | None:
    """读子进程输出：解析进度、写日志；worker 模式下读到结束标记就返回返回码。"""
    segment = 0
    last_step = -1
    last_pct = -1
    for raw in stream:
        line = raw.rstrip("\r\n").strip()
        if not line:
            continue
        if line.startswith(WORKER_DONE):
            try:
                return int(json.loads(line[len(WORKER_DONE):].strip() or "{}").get("rc", 0))
            except Exception:
                return 0
        if line.startswith(WORKER_READY):
            continue
        level = classify_line(line)
        if level == "progress":
            parsed = parse_progress(line)
            if not parsed:
                continue
            pct, step, total = parsed
            # tqdm 结束时会把 total/total 再渲染一次，只有步数回到 0/1 才是新一段
            if segment == 0 or (step <= last_step and step <= 1):
                segment += 1
                last_pct = -1
                job.log(f"采样第 {segment} 段开始（共 {total} 步）", "sys")
            if step == last_step and pct - last_pct < 5:
                continue
            last_step, last_pct = step, pct
            job.log(format_progress(pct, step, total, line, segment), "progress")
            continue
        job.log(line, level)
    return None


def _finish_job(job: Job, rc: int, p: dict) -> None:
    if job.canceled:
        job.status = "canceled"
        job.log("任务已取消", "warn")
    elif rc != 0:
        job.status = "failed"
        job.error = "inference_failed"
        job.error_detail = f"exit code {rc}"
        job.log(f"推理进程异常退出（退出码 {rc}）", "error")
    else:
        job.outputs = scan_outputs(job)
        has_main_psd = any(o["kind"] == "psd_main" for o in job.outputs)
        has_image = any(o["kind"] == "image" for o in job.outputs)
        if not has_image and not has_main_psd:
            job.status = "completed_with_warning"
            job.error = "no_outputs"
            job.log("推理结束，但输出目录为空", "error")
        elif p["save_psd"] and not has_main_psd:
            job.status = "completed_with_warning"
            job.error = "psd_not_generated"
            job.log("推理完成，但未生成 PSD（导出可能失败），图层 PNG 仍可下载", "warn")
        else:
            job.status = "completed"
            job.log(f"完成，共 {len(job.outputs)} 个输出文件", "ok")

    job.finished_at = time.time()
    job.save_meta()
    job.close_stream()
    job.done.set()


def _run_subprocess(job: Job, cmd: list[str], env: dict) -> int | None:
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except Exception as exc:
        job.status = "failed"
        job.error = "launch_failed"
        job.error_detail = str(exc)
        job.log(f"无法启动推理进程: {exc}", "error")
        job.finished_at = time.time()
        job.save_meta()
        job.close_stream()
        job.done.set()
        return None

    job.process = proc
    assert proc.stdout is not None
    rc = _consume_output(job, proc.stdout, stop_on_done=False)
    if rc is None:
        rc = proc.wait()
    job.process = None
    return rc


def run_job(job: Job, input_path: Path) -> None:
    if job.canceled:                     # 排队阶段就被取消
        job.status = "canceled"
        job.finished_at = time.time()
        job.save_meta()
        job.close_stream()
        job.done.set()
        return

    job.status = "running"
    p = job.params
    job.log("任务已启动", "sys")
    job.log(
        f"模型 · 图层 {Path(p.get('layerdiff_path') or LAYERDIFF_GGUF).name} · "
        f"深度 {Path(p.get('depth_path') or DEPTH_GGUF).name}",
        "sys",
    )
    depth_label = "auto" if p["inference_steps_depth"] == -1 else str(p["inference_steps_depth"])
    mask_label = p.get("mask_resolution", "auto")
    mask_part = f" · 分割分辨率 {mask_label}" if mask_label not in (None, "auto") else ""
    job.log(
        f"参数 · 分辨率 {p['width']}×{p['height']} · 步数 {p['inference_steps']} · "
        f"深度步数 {depth_label} · 深度分辨率 {p.get('resolution_depth', 'auto')}{mask_part} · "
        f"种子 {job.seed_used} · 量化 {'on' if p['quantized'] else 'off'} · "
        f"tblr {'on' if p['tblr_split'] else 'off'} · LaMa {'on' if p['use_lama'] else 'off'} · "
        f"软边 {'on' if p.get('soft_alpha', True) else 'off'} · "
        f"透明图精修 {'on' if p.get('edge_refine') else 'off'} · "
        f"attn {p.get('attention_backend', 'native')} · "
        f"PSD {'on' if p['save_psd'] else 'off'}",
        "sys",
    )

    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    common = str(STM_DIR / "common")
    root = str(STM_DIR)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join([p for p in (common, root, existing) if p])
    endpoint = resolve_hf_endpoint()
    if endpoint:
        env["HF_ENDPOINT"] = endpoint
        # 走镜像时关掉 xet：它会直连 cas-bridge.xethub.hf.co，国内常读超时
        if "hf-mirror.com" in endpoint:
            env.setdefault("HF_HUB_DISABLE_XET", "1")
    # 组件缓存位置：项目内 components/{hf,torch}
    components_paths.apply_env(env)
    # 关闭 torch.compile 反量化：Windows 无 MSVC 时 inductor 会直接报错
    env["SEETHROUGH_COMPILE_DEQUANT"] = "1" if CONFIG.get("compile_dequant") else "0"
    # 图层修补补丁（丢层 / 眼白叠序）：按任务注入，补丁代码读这个环境变量
    env["STG_LAYER_FIX"] = "1" if p.get("layer_fix", True) else "0"
    env["STG_INFERENCE_SCRIPT"] = str(INFERENCE_SCRIPT)
    try:
        import tls

        tls.configure_env(env)  # 让子进程里 huggingface_hub 也信任系统根证书
    except Exception:
        pass

    cmd = build_command(job, input_path)
    job.log("执行: " + " ".join(f'"{c}"' if " " in str(c) else str(c) for c in cmd), "sys")

    keep_model = bool(p.get("keep_model"))
    if not keep_model:
        # 关闭常驻时必须先释放显存，否则常驻进程和冷启动进程会抢同一块 6 GB
        _worker_stop()

    rc = None
    if keep_model:
        rc = _run_via_worker(job, [str(c) for c in cmd[2:]], env)
    if rc is None:
        rc = _run_subprocess(job, cmd, env)
    if rc is None:          # 启动阶段已失败并收尾
        return
    _finish_job(job, rc, p)



# ---------------------------------------------------------------------------
# 路由
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return send_from_directory(str(FRONTEND), "index.html")


@app.route("/settings")
def settings_page():
    return send_from_directory(str(FRONTEND), "settings.html")


def _pkg_version(name: str) -> str | None:
    try:
        import importlib.metadata as metadata

        return metadata.version(name)
    except Exception:
        return None


def _total_ram_gb() -> float | None:
    try:
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
            return round(stat.ullTotalPhys / 1024**3, 1)
    except Exception:
        return None
    return None


def _gpu_info() -> dict | None:
    """用 nvidia-smi 取显卡信息：不导入 torch，避免在服务进程里初始化 CUDA 占显存。"""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
        line = (out.stdout or "").strip().splitlines()[0]
        name, memory, driver = [part.strip() for part in line.split(",")]
        return {"name": name, "memory_mb": int(float(memory)), "driver": driver}
    except Exception:
        return None


def _patch_state() -> dict:
    utils = STM_DIR / "common" / "utils" / "inference_utils.py"
    io_utils = STM_DIR / "common" / "utils" / "io_utils.py"
    inference_psd = STM_DIR / "inference" / "scripts" / "inference_psd.py"

    def has(path: Path, marker: str) -> bool:
        try:
            return marker in path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            return False

    return {
        "use_lama": has(utils, "__STG_PATCHED_INFERENCE_UTILS__"),
        "layer_fix_drop": has(io_utils, "__STG_PATCHED_IO_UTILS_LAYER_FIX__"),
        "layer_fix_order": has(utils, "__STG_PATCHED_INFERENCE_UTILS_LAYER_FIX__"),
        "edge_refine": has(inference_psd, "__STG_PATCHED_INFERENCE_PSDPY_EDGE_REFINE__"),
    }


def _os_name() -> str:
    """系统名称与版本。Windows 走注册表（ProductName 常写成 Windows 10，按 build 号纠正），其它平台退回 platform。"""
    try:
        import winreg

        key = winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion",
            0,
            winreg.KEY_READ | getattr(winreg, "KEY_WOW64_64KEY", 0),
        )
        try:
            def query(name: str) -> str:
                try:
                    value, _ = winreg.QueryValueEx(key, name)
                    return str(value).strip()
                except OSError:
                    return ""

            product = query("ProductName")
            display = query("DisplayVersion")
            build = query("CurrentBuildNumber") or query("CurrentBuild")
        finally:
            winreg.CloseKey(key)

        if product and build.isdigit() and int(build) >= 22000 and product.startswith("Windows 10"):
            product = "Windows 11" + product[len("Windows 10"):]
        parts = [part for part in (product, display, f"build {build}" if build else "") if part]
        if parts:
            return " · ".join(parts)
    except Exception:
        pass
    return f"{platform.system()} {platform.release()} ({platform.version()})"


def _cpu_name() -> str | None:
    """准确的 CPU 型号名：Windows 从注册表读，其它平台退回 platform.processor()。"""
    try:
        import winreg

        key = winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
        )
        try:
            value, _ = winreg.QueryValueEx(key, "ProcessorNameString")
            return re.sub(r"\s+", " ", str(value)).strip()
        finally:
            winreg.CloseKey(key)
    except Exception:
        pass
    try:
        return platform.processor() or platform.machine() or None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 系统资源读数（顶栏四格）
# ---------------------------------------------------------------------------
# nvidia-smi 每次都要起子进程（首次约 0.5 s、之后约 0.07 s），不能放在请求里同步跑。
# 后台线程每秒采一次写进缓存，/api/system 只吐缓存；前端每 2 s 轮询一次。
# 这四项只是给用户看的参考，任何一步失败都不应影响推理主流程。

_SYS_LOCK = threading.Lock()
_SYS_START_LOCK = threading.Lock()
_SYS_SNAPSHOT: dict | None = None
_SYS_THREAD: threading.Thread | None = None
_SYS_HW: dict = {"gpu": None, "cpu": None}


def _sys_query_gpu() -> dict | None:
    """一次 nvidia-smi 取全利用率 / 显存 / 温度；没有 N 卡时返回 None。"""
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=4,
            check=False,
        )
        line = (out.stdout or "").strip().splitlines()[0]
        name, util, used, total, temp = [part.strip() for part in line.split(",")]
        temp = temp.replace(".", "")
        return {
            "name": name,
            "util": int(float(util)),
            "vram_used_gb": round(float(used) / 1024, 2),
            "vram_total_gb": round(float(total) / 1024, 2),
            # 部分驱动不支持温度查询，返回 [N/A]
            "temp": int(float(temp)) if temp.isdigit() else None,
        }
    except Exception:
        return None


def _sys_collect() -> dict:
    gpu = _sys_query_gpu()
    cpu_pct = None
    mem_used = None
    mem_total = None
    try:
        import psutil

        cpu_pct = round(psutil.cpu_percent(interval=None), 1)
        vm = psutil.virtual_memory()
        mem_total = round(vm.total / 1024**3, 1)
        # Windows 上 used 把 standby 也算进去，用 total - available 才贴近任务管理器
        mem_used = round((vm.total - vm.available) / 1024**3, 1)
    except Exception:
        pass

    if mem_total is None:
        mem_total = _total_ram_gb()
    mem_pct = round(mem_used / mem_total * 100, 1) if (mem_used is not None and mem_total) else None

    return {
        "ts": time.time(),
        "gpu": gpu,
        "cpu": {"name": _SYS_HW.get("cpu") or _cpu_name(), "pct": cpu_pct},
        "mem": {"used_gb": mem_used, "total_gb": mem_total, "pct": mem_pct},
    }


def _sys_sampler() -> None:
    global _SYS_SNAPSHOT
    # cpu_percent(interval=None) 首次调用恒为 0（没有上一次采样做差），先空跑一次预热
    try:
        import psutil

        psutil.cpu_percent(interval=None)
    except Exception:
        pass

    while True:
        try:
            snap = _sys_collect()
            with _SYS_LOCK:
                _SYS_SNAPSHOT = snap
        except Exception:
            pass
        time.sleep(1.0)


def _sys_start() -> None:
    """懒启动：第一次请求 /api/system 时才拉起采样线程，并同步采一次保证首屏有数。"""
    global _SYS_THREAD, _SYS_SNAPSHOT
    with _SYS_START_LOCK:
        if _SYS_THREAD is not None:
            return
        _SYS_HW["cpu"] = _cpu_name()
        # psutil 的 cpu_percent 靠两次采样做差，刚预热完就立刻取会拿到 0.0，
        # 稍等一拍再采，首屏才不会显示一个假的 0%
        try:
            import psutil

            psutil.cpu_percent(interval=None)
            time.sleep(0.15)
        except Exception:
            pass
        try:
            first = _sys_collect()
            _SYS_SNAPSHOT = first
            _SYS_HW["gpu"] = (first.get("gpu") or {}).get("name")
        except Exception:
            _SYS_SNAPSHOT = None
        _SYS_THREAD = threading.Thread(target=_sys_sampler, name="sys-sampler", daemon=True)
        _SYS_THREAD.start()


def _hf_latency() -> int | None:
    """启动时网络探测记录下来的握手耗时（毫秒）。"""
    info = NETWORK_INFO or {}
    chosen = info.get("chosen")
    if not chosen:
        return None
    latency = ((info.get("results") or {}).get(chosen) or {}).get("latency_ms")
    try:
        return int(latency) if latency is not None else None
    except (TypeError, ValueError):
        return None


def _outputs_stats() -> dict:
    """outputs/jobs 下的任务数、总占用与最近一次任务时间（跳过下划线开头的临时目录）。"""
    count = 0
    total = 0
    latest: float | None = None
    if JOBS_DIR.exists():
        for entry in JOBS_DIR.iterdir():
            try:
                if not entry.is_dir() or entry.name.startswith("_"):
                    continue
                count += 1
                mtime = entry.stat().st_mtime
                latest = mtime if latest is None else max(latest, mtime)
            except OSError:
                continue
            for root, _dirs, files in os.walk(entry):
                for name in files:
                    try:
                        total += os.path.getsize(os.path.join(root, name))
                    except OSError:
                        pass
    return {
        "jobs": count,
        "size_gb": round(total / 1024**3, 2),
        "latest": datetime.fromtimestamp(latest).strftime("%Y-%m-%d %H:%M") if latest else None,
    }


def _config_stat() -> dict:
    """config.json 的位置、最后修改时间与大小：改了没重启时一眼能看出来。"""
    try:
        st = CONFIG_PATH.stat()
        return {
            "path": str(CONFIG_PATH),
            "exists": True,
            "mtime": datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
            "size": st.st_size,
        }
    except OSError:
        return {"path": str(CONFIG_PATH), "exists": False, "mtime": None, "size": 0}


def _config_stale() -> bool:
    """config.json 的最后修改时间晚于进程启动时间 → 改了没重启。"""
    try:
        return CONFIG_PATH.stat().st_mtime > SERVER_STARTED_AT
    except OSError:
        return False


def _valid_bind_host(value: str) -> bool:
    """允许 127.0.0.1 / 0.0.0.0 / localhost 或任何合法 IPv4。"""
    if value in ("127.0.0.1", "0.0.0.0", "localhost"):
        return True
    try:
        import ipaddress

        ipaddress.IPv4Address(value)
        return True
    except ValueError:
        return False


def _port_in_use(port: int) -> bool:
    """探测端口是否已被别的程序占用（只 bind 一下就关）。"""
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("0.0.0.0", port))
        return False
    except OSError:
        return True
    finally:
        sock.close()


def _stm_files_state() -> dict:
    """上游快照里推理链路必需的几个文件是否都在。"""
    checks = {
        "inference/scripts/inference_psd.py": STM_DIR / "inference" / "scripts" / "inference_psd.py",
        "inference/scripts/ragdoll_postprocess.py": STM_DIR / "inference" / "scripts" / "ragdoll_postprocess.py",
        "common/utils/inference_utils.py": STM_DIR / "common" / "utils" / "inference_utils.py",
        "common/utils/io_utils.py": STM_DIR / "common" / "utils" / "io_utils.py",
    }
    return {rel: path.exists() for rel, path in checks.items()}


@app.route("/api/config-file", methods=["POST"])
def api_config_file():
    """把 host / port 写回 config.json。只写文件，不改监听套接字——重启后生效。"""
    if request.remote_addr not in ("127.0.0.1", "::1", None):
        return jsonify({"ok": False, "error": "local_only"}), 403

    payload = request.get_json(silent=True) or {}
    host = str(payload.get("host") or CONFIG["host"]).strip()
    if not _valid_bind_host(host):
        return jsonify({"ok": False, "error": "invalid_host", "field": "host"}), 400

    try:
        port = int(payload.get("port", CONFIG["port"]))
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "invalid_port", "field": "port"}), 400
    if not 1 <= port <= 65535:
        return jsonify({"ok": False, "error": "invalid_port", "field": "port"}), 400

    if host == CONFIG["host"] and port == CONFIG["port"]:
        return jsonify({"ok": True, "host": host, "port": port, "unchanged": True,
                        "restart_required": _config_stale()})

    occupied = port != CONFIG["port"] and _port_in_use(port)

    data = dict(CONFIG)
    data["host"] = host
    data["port"] = port
    backup = CONFIG_PATH.parent / (CONFIG_PATH.name + ".bak")
    tmp = CONFIG_PATH.parent / (CONFIG_PATH.name + ".tmp")
    try:
        if CONFIG_PATH.exists():
            shutil.copyfile(CONFIG_PATH, backup)
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, CONFIG_PATH)
    except OSError as exc:
        return jsonify({"ok": False, "error": "write_failed", "detail": str(exc)[:160]}), 500

    # 内存里同步，页面立刻显示新值；真正监听的仍是旧套接字，重启才换
    CONFIG["host"] = host
    CONFIG["port"] = port
    return jsonify({"ok": True, "host": host, "port": port,
                    "restart_required": True, "port_in_use": occupied,
                    "backup": backup.name})


@app.route("/api/terminal", methods=["POST"])
def api_terminal():
    """在本机弹出已激活 .venv 的命令提示符（3-env.bat）。只接受本机调用。"""
    if request.remote_addr not in ("127.0.0.1", "::1", None):
        return jsonify({"ok": False, "error": "local_only"}), 403
    if os.name != "nt":
        return jsonify({"ok": False, "error": "unsupported"}), 400
    script = ROOT / "3-env.bat"
    if not script.exists():
        return jsonify({"ok": False, "error": "missing_env_bat"}), 500
    try:
        subprocess.Popen(
            ["cmd", "/k", str(script)],
            cwd=str(ROOT),
            creationflags=getattr(subprocess, "CREATE_NEW_CONSOLE", 0),
        )
    except Exception as exc:
        return jsonify({"ok": False, "error": "spawn_failed", "detail": str(exc)[:120]}), 500
    return jsonify({"ok": True, "path": str(ROOT)})


OPEN_TARGETS = {
    "root": ROOT,
    "outputs": OUTPUT_DIR,
    "models": MODELS_DIR,
    "stm": STM_DIR,
}


@app.route("/api/open-folder", methods=["POST"])
def api_open_folder():
    """在资源管理器里打开项目内的固定目录。只允许白名单，且只接受本机调用。"""
    if request.remote_addr not in ("127.0.0.1", "::1", None):
        return jsonify({"ok": False, "error": "local_only"}), 403
    name = str((request.get_json(silent=True) or {}).get("target") or "").strip()
    target = OPEN_TARGETS.get(name)
    if target is None:
        return jsonify({"ok": False, "error": "invalid_target"}), 400
    if not target.exists():
        return jsonify({"ok": False, "error": "not_found"}), 404
    try:
        if os.name == "nt":
            os.startfile(str(target))  # type: ignore[attr-defined]
        else:
            subprocess.Popen(["xdg-open", str(target)])
    except Exception as exc:
        return jsonify({"ok": False, "error": "open_failed", "detail": str(exc)[:120]}), 500
    return jsonify({"ok": True, "path": str(target)})


def _components_info() -> dict:
    """组件缓存（HF_HOME / TORCH_HOME）的实际位置、来源与体积。"""
    base = components_paths.base_dir()
    hf = components_paths.hf_home() or components_paths.user_cache_dir()
    th = components_paths.torch_home() or (Path.home() / ".cache" / "torch")
    size = 0
    files = 0
    for root in {hf, th}:
        if not root.exists():
            continue
        for dirpath, _dirs, names in os.walk(root):
            for name in names:
                try:
                    size += os.path.getsize(os.path.join(dirpath, name))
                    files += 1
                except OSError:
                    pass
    return {
        "source": "project" if base is not None else "user",
        "base": str(base) if base is not None else None,
        "hf_home": str(hf),
        "torch_home": str(th),
        "size_gb": round(size / 1024**3, 2),
        "files": files,
        "exists": hf.exists(),
    }


@app.route("/api/env")
def api_env():
    """设置页用的只读环境快照。全部走轻量查询，不 import torch。"""
    torch_version = _pkg_version("torch")
    cuda_tag = None
    if torch_version and "+cu" in torch_version:
        cuda_tag = torch_version.split("+", 1)[1].replace("cu", "").replace(".", "")[:3]
        cuda_tag = f"{cuda_tag[:2]}.{cuda_tag[2:]}" if len(cuda_tag) == 3 else cuda_tag

    models = scan_models()
    disk = shutil.disk_usage(str(ROOT))
    return jsonify(
        {
            "runtime": {
                "python": sys.version.split()[0],
                "executable": sys.executable,
                "torch": torch_version,
                "cuda": cuda_tag,
                "xformers": _pkg_version("xformers"),
                "triton": _pkg_version("triton-windows"),
            },
            "hardware": {
                "gpu": _gpu_info(),
                "os": _os_name(),
                "cpu": _cpu_name(),
                "cpu_threads": os.cpu_count(),
                "ram_gb": _total_ram_gb(),
                "hf_endpoint": resolve_hf_endpoint(),
                "hf_source": (NETWORK_INFO or {}).get("chosen"),
                "hf_latency_ms": _hf_latency(),
                "hf_checked_at": (NETWORK_INFO or {}).get("checked_at"),
            },
            "paths": {
                "root": str(ROOT),
                "stm": str(STM_DIR),
                "models": str(MODELS_DIR),
                "outputs": str(OUTPUT_DIR),
            },
            "components": _components_info(),
            "models": models.get("models", []),
            "model_defaults": models.get("defaults", {}),
            "patches": _patch_state(),
            "config": {
                "host": CONFIG["host"],
                "port": CONFIG["port"],
                "attention_backend": CONFIG.get("attention_backend", "native"),
                "attention_available": ATTENTION_AVAILABLE,
                "compile_dequant": bool(CONFIG.get("compile_dequant")),
                "layer_fix": bool(CONFIG["defaults"].get("layer_fix", True)),
                "hf_endpoint": CONFIG.get("hf_endpoint", "auto"),
                "max_upload_mb": CONFIG.get("max_upload_mb", 50),
                "stm_path": CONFIG.get("stm_path", "stm"),
                "models_dir": CONFIG.get("models_dir", "models"),
                "output_dir": CONFIG.get("output_dir", "outputs"),
            },
            "server": {
                "pid": os.getpid(),
                "started_at": datetime.fromtimestamp(SERVER_STARTED_AT).strftime("%Y-%m-%d %H:%M:%S"),
                "uptime_s": int(time.time() - SERVER_STARTED_AT),
                "started_epoch": round(SERVER_STARTED_AT, 3),
                "served_host": request.host,
                # config.json 在进程启动后被改过：页面提示「待重启生效」
                "config_stale": _config_stale(),
                "local_request": request.remote_addr in ("127.0.0.1", "::1", None),
                "cwd": str(ROOT),
            },
            "config_file": _config_stat(),
            "maintenance": {
                "venv": (ROOT / ".venv" / "Scripts" / "python.exe").exists()
                if os.name == "nt"
                else (ROOT / ".venv" / "bin" / "python").exists(),
                "stm_files": _stm_files_state(),
                "outputs": _outputs_stats(),
            },
            "disk": {
                "free_gb": round(disk.free / 1024**3, 1),
                "total_gb": round(disk.total / 1024**3, 1),
            },
        }
    )


@app.route("/api/system")
def api_system():
    """顶栏资源读数。只吐后台线程的缓存，接口本身不跑 nvidia-smi。"""
    _sys_start()
    with _SYS_LOCK:
        snap = _SYS_SNAPSHOT
    payload: dict = {"ok": True, "ready": bool(snap)}
    if snap:
        payload.update(snap)
    return jsonify(payload)


@app.route("/api/config")
def api_config():
    return jsonify(
        {
            "defaults": CONFIG["defaults"],
            "constraints": CONFIG["constraints"],
            "supported_ext": sorted(ALLOWED_EXT),
            "max_upload_mb": CONFIG["max_upload_mb"],
            "attention_backend": CONFIG.get("attention_backend", "native"),
            "attention_backends": ATTENTION_AVAILABLE,
            "hf_endpoint": resolve_hf_endpoint(),
            "models_dir": str(MODELS_DIR),
        }
    )


@app.route("/api/models")
def api_models():
    return jsonify(scan_models())


@app.route("/api/check_models")
def api_check_models():
    patch_ok = STM_UTILS.exists() and PATCH_MARKER in STM_UTILS.read_text(encoding="utf-8", errors="ignore")
    layerdiff = {
        "path": str(LAYERDIFF_GGUF),
        "exists": LAYERDIFF_GGUF.exists(),
        "size": LAYERDIFF_GGUF.stat().st_size if LAYERDIFF_GGUF.exists() else 0,
    }
    depth = {
        "path": str(DEPTH_GGUF),
        "exists": DEPTH_GGUF.exists(),
        "size": DEPTH_GGUF.stat().st_size if DEPTH_GGUF.exists() else 0,
    }
    stm_repo = {"path": str(STM_DIR), "exists": INFERENCE_SCRIPT.exists()}
    ready = layerdiff["exists"] and depth["exists"] and stm_repo["exists"]
    return jsonify(
        {
            "ready": ready,
            "network": NETWORK_INFO,
            "items": {
                "layerdiff": layerdiff,
                "depth": depth,
                "stm_repo": stm_repo,
                "use_lama_patch": {"applied": patch_ok, "path": str(STM_UTILS)},
            },
        }
    )


@app.route("/api/run", methods=["POST"])
def api_run():
    current = active_job()
    if current:
        return jsonify({"ok": False, "error": "busy", "job_id": current.job_id}), 409

    # --- STM 预检（模型文件在参数校验后按界面所选项检查） ---
    if not INFERENCE_SCRIPT.exists():
        return jsonify({"ok": False, "error": "models_missing", "missing": ["stm_repo"]}), 400

    # --- 上传文件 ---
    if "image" not in request.files:
        return jsonify({"ok": False, "error": "no_file"}), 400
    upload = request.files["image"]
    if not upload.filename:
        return jsonify({"ok": False, "error": "no_file"}), 400
    ext = os.path.splitext(upload.filename)[1].lower()
    if ext not in ALLOWED_EXT:
        return jsonify({"ok": False, "error": "unsupported_format", "details": [{"field": "image", "code": "ext", "ext": ext}]}), 400

    data = upload.read()
    if not data:
        return jsonify({"ok": False, "error": "empty_file"}), 400
    try:
        from PIL import Image

        with Image.open(io.BytesIO(data)) as probe:
            probe.verify()
    except Exception as exc:
        return jsonify(
            {"ok": False, "error": "unsupported_format", "details": [{"field": "image", "code": "decode", "detail": str(exc)}]}
        ), 400

    # --- 参数 ---
    try:
        raw_params = json.loads(request.form.get("params", "{}"))
        if not isinstance(raw_params, dict):
            raise ValueError
    except Exception:
        return jsonify({"ok": False, "error": "invalid_params", "details": [{"field": "params", "code": "not_json"}]}), 400

    patch_ok = STM_UTILS.exists() and PATCH_MARKER in STM_UTILS.read_text(encoding="utf-8", errors="ignore")
    clean, errors = validate_params(raw_params, patch_ok)

    # 模型选择：未指定时回落到 config.json 的默认路径
    for field, key, fallback in (
        ("layerdiff_model", "layerdiff_path", LAYERDIFF_GGUF),
        ("depth_model", "depth_path", DEPTH_GGUF),
    ):
        raw_value = raw_params.get(field)
        if raw_value in (None, ""):
            clean[key] = str(fallback)
            continue
        resolved, error = resolve_model_choice(raw_value, field)
        if error:
            errors.append(error)
        else:
            clean[key] = resolved

    if errors:
        return jsonify({"ok": False, "error": "invalid_params", "details": errors}), 400

    missing = []
    if not Path(clean["layerdiff_path"]).exists():
        missing.append("layerdiff")
    if not Path(clean["depth_path"]).exists():
        missing.append("depth")
    if missing:
        return jsonify({"ok": False, "error": "models_missing", "missing": missing}), 400

    seed_used = clean["seed"] if clean["seed_mode"] == "fixed" else next_seed()

    job_id = uuid.uuid4().hex[:12]
    job_dir = _ensure_job_dir(job_id)
    input_path = job_dir / f"input{ext}"
    try:
        input_path.write_bytes(data)
    except OSError as exc:
        return jsonify({"ok": False, "error": "storage_failed", "detail": str(exc)}), 500

    job = Job(job_id, job_dir, clean, seed_used)
    job.save_meta()
    with JOBS_LOCK:
        JOBS[job_id] = job
        # 内存里只保留最近的任务，历史仍可从 meta.json 还原
        if len(JOBS) > 40:
            for old_id, old_job in list(JOBS.items()):
                if old_id != job_id and old_job.status not in ("queued", "running"):
                    JOBS.pop(old_id, None)
                if len(JOBS) <= 30:
                    break

    threading.Thread(target=run_job, args=(job, input_path), daemon=True).start()
    return jsonify({"ok": True, "job_id": job_id, "seed_used": seed_used, "params": clean})


@app.route("/api/stream/<job_id>")
def api_stream(job_id: str):
    job = load_job(job_id)
    if not job:
        abort(404)

    def sse(event: str, payload: dict) -> str:
        return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"

    def gen():
        yield "retry: 3000\n\n"
        if job.done.is_set():                       # 已结束（含重启后还原的任务）
            for entry in job.history[-500:]:
                yield sse("log", entry)
        else:
            while True:
                try:
                    entry = job.q.get(timeout=1)
                except queue.Empty:
                    if job.done.is_set():
                        break
                    yield ": ping\n\n"
                    continue
                if entry is None:
                    break
                yield sse("log", entry)
        yield sse(
            "done",
            {
                "status": job.status,
                "error": job.error,
                "error_detail": job.error_detail,
                "outputs": job.outputs,
                "seed_used": job.seed_used,
                "params": job.params,
                "elapsed": round((job.finished_at or time.time()) - job.started_at, 1),
            },
        )

    return Response(gen(), mimetype="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/api/status/<job_id>")
def api_status(job_id: str):
    job = load_job(job_id)
    if not job:
        abort(404)
    return jsonify(
        {
            "job_id": job.job_id,
            "status": job.status,
            "error": job.error,
            "error_detail": job.error_detail,
            "outputs": job.outputs,
            "seed_used": job.seed_used,
            "params": job.params,
            "elapsed": round((job.finished_at or time.time()) - job.started_at, 1),
            "log_tail": job.history[-200:],
        }
    )


@app.route("/api/jobs")
def api_jobs():
    """历史任务列表：读 outputs/jobs/*/meta.json，按时间倒序。"""
    try:
        limit = max(1, min(int(request.args.get("limit") or 40), 200))
    except (TypeError, ValueError):
        limit = 40

    try:
        dirs = [d for d in JOBS_DIR.iterdir() if d.is_dir()]
        dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    except OSError:
        dirs = []

    items = []
    for d in dirs:
        if len(items) >= limit:
            break
        meta_path = d / "meta.json"
        if not meta_path.is_file():
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue

        params = meta.get("params") or {}
        started = meta.get("started_at")
        finished = meta.get("finished_at")
        elapsed = round(finished - started, 1) if started and finished else None
        model_path = params.get("layerdiff_path")
        has_input = any(
            (d / ("input" + ext)).is_file() for ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp")
        )

        items.append(
            {
                "job_id": meta.get("job_id") or d.name,
                "status": meta.get("status") or "unknown",
                "started_at": started,
                "finished_at": finished,
                "elapsed": elapsed,
                "width": params.get("width"),
                "height": params.get("height"),
                "steps": params.get("inference_steps"),
                "model": Path(model_path).name if model_path else None,
                "error": meta.get("error"),
                # error_hint 在任务结束时已按同一套规则算好并写入 meta；
                # 这里只做一次文案收敛，让旧版本写下的模板文案也能显示成「退出码 N」
                "error_hint": _format_error_text(
                    meta.get("error_hint") or meta.get("error_detail")
                ),
                "has_input": has_input,
            }
        )

    return jsonify({"ok": True, "jobs": items})


@app.route("/api/job_input/<job_id>")
def api_job_input(job_id: str):
    """历史记录里显示的原图（jobs/<id>/input.*），只读、限定目录。"""
    if not re.fullmatch(r"[0-9a-f]{12}", job_id):
        abort(400)
    job_dir = JOBS_DIR / job_id
    for ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
        candidate = job_dir / ("input" + ext)
        if candidate.is_file():
            return send_from_directory(str(job_dir), candidate.name)
    abort(404)


@app.route("/api/current")
def api_current():
    """页面刷新/重开后接回正在运行的任务。"""
    job = active_job()
    if not job:
        return jsonify({"ok": True, "job": None})
    return jsonify(
        {
            "ok": True,
            "job": {
                "job_id": job.job_id,
                "status": job.status,
                "params": job.params,
                "seed_used": job.seed_used,
                "elapsed": round(time.time() - job.started_at, 1),
                "log_tail": job.history[-200:],
            },
        }
    )


@app.route("/api/last")
def api_last():
    """最近一次成功任务（24 小时内），供页面刷新后回看结果。"""
    best = None
    if JOBS_DIR.exists():
        for meta_path in JOBS_DIR.glob("*/meta.json"):
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if meta.get("status") not in ("completed", "completed_with_warning"):
                continue
            if not meta.get("outputs"):
                continue
            finished = meta.get("finished_at") or 0
            if time.time() - finished > 86400:
                continue
            if not best or finished > best.get("finished_at", 0):
                best = meta

    if not best:
        return jsonify({"ok": True, "job": None})

    job = {
        "job_id": best.get("job_id"),
        "status": best.get("status"),
        "error": best.get("error"),
        "error_detail": best.get("error_detail"),
        "outputs": best.get("outputs") or [],
        "params": best.get("params") or {},
        "seed_used": best.get("seed_used"),
        "elapsed": round((best.get("finished_at") or 0) - (best.get("started_at") or 0), 1),
        "finished_at": best.get("finished_at"),
    }
    return jsonify({"ok": True, "job": job})


@app.route("/api/jobs/<job_id>/cancel", methods=["POST"])
def api_cancel(job_id: str):
    job = load_job(job_id)
    if not job:
        abort(404)
    if job.status not in ("queued", "running"):
        return jsonify({"ok": False, "error": "not_running"}), 409
    job.canceled = True
    job.status = "canceled"
    if job.process:
        _kill_tree(job.process)
    job.log("收到取消请求", "warn")
    return jsonify({"ok": True})


# ---------------------------------------------------------------------------
# 后处理重跑（不重跑采样）与「打开文件夹」
# ---------------------------------------------------------------------------
REASSEMBLE: dict[str, dict] = {}


def _out_subdirs(job: Job) -> list[Path]:
    if not job.out_dir.exists():
        return []
    return sorted(p for p in job.out_dir.iterdir() if p.is_dir() and (p / "src_img.png").exists())


def _reassemble_env(job: Job) -> dict:
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    parts = [str(STM_DIR / "common"), str(STM_DIR)]
    if env.get("PYTHONPATH"):
        parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(parts)
    # 后处理同样要用项目内的组件缓存（RTMPose 等）：不设 HF_HOME 会去系统缓存找，
    # 找不到就直连 huggingface.co，国内基本是 502；HF_ENDPOINT 一并带上走镜像
    endpoint = resolve_hf_endpoint()
    if endpoint:
        env["HF_ENDPOINT"] = endpoint
        if "hf-mirror.com" in endpoint:
            env.setdefault("HF_HUB_DISABLE_XET", "1")
    components_paths.apply_env(env)
    env["SEETHROUGH_COMPILE_DEQUANT"] = "1" if CONFIG.get("compile_dequant") else "0"
    env["STG_LAYER_FIX"] = "1" if job.params.get("layer_fix", True) else "0"
    return env


def _postprocess_argv(target: Path, opts: dict) -> list[str]:
    argv = [sys.executable, "-u", str(POSTPROCESS_SCRIPT), str(target)]
    if opts.get("save_psd", True):
        argv.append("--save_to_psd")
    if opts.get("save_cut_images", True):
        argv.append("--save_cut_images")
    if opts.get("tblr_split", True):
        argv.append("--tblr_split")
    if opts.get("ragdoll_split"):
        argv += [
            "--ragdoll_split",
            "--ragdoll-confidence", f"{float(opts.get('ragdoll_confidence', 0.3)):.2f}",
            "--ragdoll-overlap", f"{float(opts.get('ragdoll_overlap', 0.2)):.2f}",
        ]
    return argv


def _log_level_of(line: str) -> str:
    """把后处理输出的一行映射成日志级别，供控制台着色。"""
    low = line.lower()
    if "traceback" in low or "exception" in low or "error" in low or "failed" in low:
        return "error"
    if "warning" in low or "[warn]" in low:
        return "warn"
    return "info"


def _run_reassemble(job_id: str, opts: dict) -> None:
    job = load_job(job_id)
    state = REASSEMBLE.setdefault(job_id, {"log": []})
    # 历史任务默认只在 meta 里、load_job 每次重建对象；固化到内存后
    # /api/status 才能读到这次后处理写下的日志
    if job:
        JOBS.setdefault(job_id, job)

    def emit(msg: str, level: str | None = None) -> None:
        state["log"].append(msg)
        state["log"] = state["log"][-600:]
        if job:
            job.log(msg, level or _log_level_of(msg))

    started = time.time()
    try:
        targets = _out_subdirs(job) if job else []
        if not targets:
            emit("后处理重跑失败：找不到可处理的输出目录（缺少 src_img.png）", "error")
            state.update({"running": False, "ok": False, "rc": 1, "finished": time.time()})
            return
        env = _reassemble_env(job)
        flags = ["左右分割" if opts.get("tblr_split", True) else "不分割"]
        if opts.get("ragdoll_split"):
            flags.append("骨骼切割")
        if opts.get("save_psd", True):
            flags.append("PSD")
        if opts.get("save_cut_images", True):
            flags.append("切图")
        emit(f"后处理重跑 · 开始 · {' / '.join(flags)}", "info")

        rc = 0
        for target in targets:
            emit(f"[postprocess] {target.name}", "info")
            proc = subprocess.Popen(
                _postprocess_argv(target, opts),
                cwd=str(ROOT),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                line = line.rstrip()
                if line:
                    emit(line)
            rc = proc.wait() or rc
        job.outputs = scan_outputs(job)
        job.save_meta()
        cost = round(time.time() - started, 1)
        if rc == 0:
            emit(f"后处理重跑完成 · 用时 {cost} s", "info")
        else:
            emit(f"后处理重跑失败 · 退出码 {rc} · 用时 {cost} s", "error")
        state.update({"running": False, "ok": rc == 0, "rc": rc, "finished": time.time()})
    except Exception as exc:
        emit(f"后处理重跑异常：{type(exc).__name__}: {exc}", "error")
        state.update({"running": False, "ok": False, "rc": 1, "finished": time.time()})


@app.route("/api/jobs/<job_id>/reassemble", methods=["GET", "POST"])
def api_reassemble(job_id: str):
    job = load_job(job_id)
    if not job:
        abort(404)
    state = REASSEMBLE.get(job_id)
    if request.method == "GET":
        if not state:
            return jsonify({"ok": True, "running": False, "started": False, "outputs": job.outputs})
        return jsonify(
            {
                "ok": True,
                "running": bool(state.get("running")),
                "started": True,
                "result": state.get("ok"),
                "rc": state.get("rc"),
                "log_tail": state.get("log", [])[-80:],
                "outputs": job.outputs,
            }
        )

    if active_job() is not None:
        return jsonify({"ok": False, "error": "busy"}), 409
    if state and state.get("running"):
        return jsonify({"ok": False, "error": "already_running"}), 409
    if not _out_subdirs(job):
        return jsonify({"ok": False, "error": "no_output_dir"}), 409

    raw = request.get_json(silent=True) or {}
    opts = {
        "save_psd": _as_bool(raw.get("save_psd"), bool(job.params.get("save_psd", True))),
        "save_cut_images": _as_bool(raw.get("save_cut_images"), bool(job.params.get("save_cut_images", True))),
        "tblr_split": _as_bool(raw.get("tblr_split"), bool(job.params.get("tblr_split", True))),
        "ragdoll_split": _as_bool(raw.get("ragdoll_split"), bool(job.params.get("ragdoll_split", False))),
        "ragdoll_confidence": raw.get("ragdoll_confidence", job.params.get("ragdoll_confidence", 0.3)),
        "ragdoll_overlap": raw.get("ragdoll_overlap", job.params.get("ragdoll_overlap", 0.2)),
    }
    REASSEMBLE[job_id] = {"log": [], "running": True, "started": time.time()}
    threading.Thread(target=_run_reassemble, args=(job_id, opts), daemon=True).start()
    return jsonify({"ok": True, "running": True, "options": opts})


@app.route("/api/jobs/<job_id>/reveal", methods=["POST"])
def api_reveal(job_id: str):
    """在资源管理器里打开该任务的 out/ 目录（本地工具，仅 Windows 有意义）。"""
    job = load_job(job_id)
    if not job or not job.out_dir.exists():
        abort(404)
    target = job.out_dir
    psds = sorted(target.glob("*.psd"))
    try:
        if os.name == "nt":
            if psds:
                subprocess.Popen(["explorer", f"/select,{psds[0]}"])
            else:
                os.startfile(str(target))  # type: ignore[attr-defined]
        else:
            subprocess.Popen(["xdg-open", str(target)])
    except Exception as exc:
        return jsonify({"ok": False, "error": "open_failed", "detail": str(exc)[:120]}), 500
    return jsonify({"ok": True, "path": str(target)})


@app.route("/api/jobs/<job_id>/pose")
def api_job_pose(job_id: str):
    """读 pose_keypoints.json，返回置信度最低的几个关节，供界面提示该不该降门限。"""
    job = load_job(job_id)
    if not job:
        abort(404)
    candidates = sorted(job.out_dir.rglob("pose_keypoints.json")) if job.out_dir.exists() else []
    if not candidates:
        return jsonify({"ok": False, "error": "no_pose_data"}), 404
    path = candidates[0]
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return jsonify({"ok": False, "error": "unreadable", "detail": str(exc)[:120]}), 500

    scores = data.get("scores") or []
    joints = []
    for index, raw in enumerate(scores[: len(BODY_KEYPOINTS)]):
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        joints.append({"index": index, "name": BODY_KEYPOINTS[index], "score": round(value, 3)})

    try:
        threshold = float(job.params.get("ragdoll_confidence"))
    except (TypeError, ValueError):
        threshold = None

    below = [j for j in joints if threshold is not None and j["score"] < threshold]
    # 有低于门限的就只报这些（最多 6 个），否则报最低的 3 个供参考
    weakest = (sorted(below, key=lambda item: item["score"])[:6] if below else sorted(joints, key=lambda item: item["score"])[:3])
    return jsonify(
        {
            "ok": True,
            "rel": path.relative_to(job.out_dir).as_posix(),
            "count": len(scores),
            "joints": joints,
            "weakest": weakest,
            "threshold": threshold,
            "below_threshold": [j["name"] for j in below][:6],
        }
    )

@app.route("/api/jobs/<job_id>", methods=["DELETE"])
def api_delete_job(job_id: str):
    job = JOBS.get(job_id)
    if job and job.status in ("queued", "running"):
        return jsonify({"ok": False, "error": "still_running"}), 409
    job_dir = JOBS_DIR / job_id
    if not job_dir.exists():
        abort(404)
    # Windows 上刚下载完的文件可能仍被占用，短暂重试而不是静默失败
    removed = False
    for _ in range(6):
        shutil.rmtree(job_dir, ignore_errors=True)
        if not job_dir.exists():
            removed = True
            break
        time.sleep(0.15)
    if not removed:
        return jsonify({"ok": False, "error": "cleanup_failed", "detail": "文件被占用，请稍后再试"}), 500
    with JOBS_LOCK:
        JOBS.pop(job_id, None)
    return jsonify({"ok": True})


@app.route("/api/file/<job_id>/<path:rel>")
def api_file(job_id: str, rel: str):
    job_dir = JOBS_DIR / job_id
    base = (job_dir / "out").resolve()
    if not base.exists():
        abort(404)
    target = (base / rel).resolve()
    if target != base and base not in target.parents:
        abort(400)
    if not target.is_file():
        abort(404)
    return send_from_directory(
        str(base),
        rel,
        as_attachment=request.args.get("download") == "1",
        download_name=Path(rel).name,
    )


@app.errorhandler(413)
def too_large(_):
    return jsonify({"ok": False, "error": "file_too_large", "max_upload_mb": CONFIG["max_upload_mb"]}), 413


@app.errorhandler(500)
def server_error(exc):
    if request.path.startswith("/api/"):
        return jsonify({"ok": False, "error": "internal_error", "detail": str(exc)}), 500
    return "Internal Server Error", 500


if __name__ == "__main__":
    JOBS_DIR.mkdir(parents=True, exist_ok=True)

    # 合并系统根证书（安全软件做 TLS 中间人时，requests 默认会校验失败）
    try:
        import tls

        bundle = tls.configure_env()
        if bundle:
            print(f"[证书] 已合并系统根证书: {bundle}")
    except Exception as exc:
        print(f"[WARN] 证书合并失败: {exc}")

    # 运行前先检测网络：决定运行期组件从官方还是镜像下载
    try:
        import netcheck

        NETWORK_INFO = netcheck.detect()
    except Exception as exc:  # 探测失败不应阻止服务启动
        NETWORK_INFO = None
        print(f"[WARN] 下载源探测失败: {exc}")

    print("=" * 64)
    print("SeeThrough-GGUF-Win")
    print(f"  项目根目录   : {ROOT}")
    print(f"  STM 仓库     : {STM_DIR}  ({'存在' if INFERENCE_SCRIPT.exists() else '缺失'})")
    print(f"  LayerDiff3D  : {LAYERDIFF_GGUF}  ({'存在' if LAYERDIFF_GGUF.exists() else '缺失'})")
    print(f"  Marigold3D   : {DEPTH_GGUF}  ({'存在' if DEPTH_GGUF.exists() else '缺失'})")
    print(f"  Python       : {sys.version.split()[0]}  ({sys.executable})")
    if NETWORK_INFO:
        chosen = NETWORK_INFO.get("chosen")
        print(f"  下载源       : {chosen or '均不可达'} ({NETWORK_INFO.get('endpoint') or '—'})")
    print(f"  访问地址     : http://{CONFIG['host']}:{CONFIG['port']}")
    print("=" * 64)
    app.run(host=CONFIG["host"], port=int(CONFIG["port"]), threaded=True, use_reloader=False)
