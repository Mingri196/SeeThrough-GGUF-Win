#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""生成「certifi + Windows 系统证书」合并的 CA 包。

为什么需要：requests / huggingface_hub 只信任 certifi 内置根证书。国内常见的安全软件
（Kaspersky、部分企业代理）会做 TLS 中间人替换证书，根证书装在 Windows 证书库里，
于是 curl 能下、Python 下不了，报 `CERTIFICATE_VERIFY_FAILED: self-signed certificate
in certificate chain`。

这里把系统证书导出后与 certifi 合并成一个 PEM，通过 REQUESTS_CA_BUNDLE /
SSL_CERT_FILE / CURL_CA_BUNDLE 交给 requests、huggingface_hub 和 curl 使用。
不关闭证书校验。
"""

from __future__ import annotations

import os
import ssl
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUNDLE_PATH = ROOT / "outputs" / ".cache" / "ca-bundle.pem"
MAX_AGE_SECONDS = 7 * 24 * 3600


def ensure_bundle(force: bool = False) -> str | None:
    """返回合并包路径；失败时返回 None（调用方应回退到默认证书）。"""
    try:
        import certifi
    except Exception:
        return None

    if (
        not force
        and BUNDLE_PATH.exists()
        and (time.time() - BUNDLE_PATH.stat().st_mtime) < MAX_AGE_SECONDS
    ):
        return str(BUNDLE_PATH)

    try:
        parts = [Path(certifi.where()).read_text(encoding="utf-8")]
        context = ssl.create_default_context()
        context.load_default_certs()
        system_certs = context.get_ca_certs(binary_form=True)
        for der in system_certs:
            parts.append(ssl.DER_cert_to_PEM_cert(der))

        BUNDLE_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = BUNDLE_PATH.with_suffix(".tmp")
        tmp.write_text("".join(parts), encoding="utf-8")
        tmp.replace(BUNDLE_PATH)
        return str(BUNDLE_PATH)
    except Exception:
        return None


def configure_env(env: dict | None = None, force: bool = False) -> str | None:
    """把合并包写进环境变量。传入 env 时只改该字典（用于子进程）。"""
    bundle = ensure_bundle(force=force)
    if not bundle:
        return None
    target = os.environ if env is None else env
    for key in ("REQUESTS_CA_BUNDLE", "SSL_CERT_FILE", "CURL_CA_BUNDLE"):
        target.setdefault(key, bundle)
    return bundle


if __name__ == "__main__":
    path = ensure_bundle(force=True)
    print(path or "生成失败")
