#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""下载源探测：决定模型从 huggingface.co 还是 hf-mirror.com 下载。

规则（可被 --endpoint 覆盖）：
  1. 并行探测两个站点的模型 API；
  2. 在两个都可达时，官方站有 600 ms 的优先偏置（海外用户通常官方更快）；
  3. 官方不可达或明显更慢时选镜像（国内常见情况）；
  4. 两个都不可达时返回 None，由调用方决定报错或让用户挂代理。

本模块只做探测，不下载任何权重。
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

ENDPOINTS = {
    "huggingface": "https://huggingface.co",
    "hf-mirror": "https://hf-mirror.com",
}
PROBE_PATH = "/api/models/icefog72/seethroughv0.0.2_layerdiff3d_gguf"
OFFICIAL_BIAS_MS = 600


def probe(name: str, base: str, timeout: float = 4.0) -> dict:
    import requests

    started = time.perf_counter()
    try:
        resp = requests.get(base + PROBE_PATH, timeout=timeout)
        elapsed = round((time.perf_counter() - started) * 1000)
        ok = resp.status_code == 200
        return {
            "name": name,
            "url": base,
            "ok": ok,
            "status": resp.status_code,
            "latency_ms": elapsed,
        }
    except Exception as exc:  # 超时、DNS、连接被拒等
        return {
            "name": name,
            "url": base,
            "ok": False,
            "status": None,
            "latency_ms": None,
            "error": type(exc).__name__,
        }


def detect(prefer: str | None = None, timeout: float = 4.0) -> dict:
    """返回 {chosen, endpoint, results, checked_at}。prefer 可取 huggingface / hf-mirror / auto。"""
    with ThreadPoolExecutor(max_workers=len(ENDPOINTS)) as pool:
        futures = {name: pool.submit(probe, name, base, timeout) for name, base in ENDPOINTS.items()}
        results = {name: fut.result() for name, fut in futures.items()}

    reachable = [name for name, item in results.items() if item["ok"]]

    chosen = None
    reason = ""
    if prefer in ENDPOINTS:
        if prefer in reachable:
            chosen, reason = prefer, "manual"
        else:
            chosen, reason = None, "manual-unreachable"
    elif reachable:
        def score(name: str) -> int:
            latency = results[name]["latency_ms"] or 10**9
            return latency - (OFFICIAL_BIAS_MS if name == "huggingface" else 0)

        chosen = min(reachable, key=score)
        reason = "auto"
    else:
        reason = "unreachable"

    return {
        "chosen": chosen,
        "endpoint": ENDPOINTS.get(chosen) if chosen else None,
        "reason": reason,
        "results": results,
        "checked_at": int(time.time()),
    }


def describe(info: dict) -> str:
    if not info:
        return "未检测"
    if not info.get("chosen"):
        return "两个下载源均不可达（可挂代理后重试，或手动放入 models/）"
    parts = []
    for name, item in info["results"].items():
        if item["ok"]:
            parts.append(f"{name} {item['latency_ms']}ms")
        else:
            parts.append(f"{name} 不可达")
    return f"{info['chosen']} · " + " / ".join(parts)


if __name__ == "__main__":
    info = detect()
    print("下载源探测：")
    for name, item in info["results"].items():
        if item["ok"]:
            print(f"  [可达] {name:<12} {item['latency_ms']:>5} ms  {item['url']}")
        else:
            print(f"  [不可达] {name:<10} {item.get('error') or item.get('status')}  {item['url']}")
    if info["chosen"]:
        print(f"\n建议下载源：{info['chosen']} -> {info['endpoint']}")
    else:
        print("\n建议：两个源均不可达，请检查网络/代理，或手动下载模型放入 models/")
