# 许可证说明 / Licenses

本仓库整体以 **Apache License 2.0** 发布，完整文本见根目录 [LICENSE](../LICENSE)。

## 1. 本项目自有代码（backend / frontend / 脚本 / 文档）
- 维护者：Mingri196（<https://github.com/Mingri196/SeeThrough-GGUF-Win>）
- 协议：**Apache-2.0**
- 权利：可自由使用、修改、再分发、商用，并附带专利授权。
- 义务：保留 [LICENSE](../LICENSE) 与 [NOTICE](../NOTICE)；若修改了文件，需注明改动。

## 2. 内置的上游代码（`stm/`）
- 来源：<https://github.com/IceFog72/see-through-gguf>，基线 commit `02cba07`（2026-09-14）
- 协议：**Apache-2.0**（原始 `stm/LICENSE` 已保留）
- 本仓库已把该快照直接放入 `stm/`，方便一次 clone 即可安装；**不再在安装时另行克隆**。
- 对上游代码的修改全部记录在 [MODIFICATIONS.md](../MODIFICATIONS.md)：仅 `common/utils/inference_utils.py`、`inference/scripts/inference_psd.py`、`common/utils/io_utils.py` 三个文件。
- 上游还要求保留 **shitagaki-lab/see-through** 项目与论文的署名，已写入 [NOTICE](../NOTICE) 与 README。
- 不得使用上游名称或标识暗示官方背书。

## 3. GGUF 模型权重（不随仓库分发）
- 本项目**不打包、不分发任何模型权重**，两个 GGUF 由使用者自行下载。
- 经 2026-10-29 核对：`icefog72/seethroughv0.0.2_layerdiff3d_gguf` 与 `icefog72/seethroughv0.0.1_marigold_gguf` 两个仓库**均未声明 license**；其来源模型适用 `openrail++`（CreativeML OpenRAIL++-M），带使用限制。
- 若要对外分享生成图或再分发权重，请遵守 OpenRAIL++ 的限制与署名要求，并自行向模型作者确认授权。
- 首次运行还会下载若干辅助组件（分割 / LaMa / RTMPose / Marigold 文本编码器 / anime-seg），默认落在项目内 `components/`（不入库），许可见 [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)。

---

### 其它
- GSAP（`frontend/vendor/gsap.min.js`）：GreenSock Standard License（<https://gsap.com/standard-license>），非 OSI 开源协议；允许用于本项目这类免费工具，禁止用于与 GSAP 竞争的产品。
- Flask / PyTorch / diffusers / gguf / psd-tools 等 Python 依赖：随 `requirements*.txt` 安装，遵循各自许可证，不随仓库分发。
