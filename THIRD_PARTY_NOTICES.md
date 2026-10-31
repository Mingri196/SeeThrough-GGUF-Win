# 第三方组件与许可 / Third-Party Notices

本仓库**只包含代码与脚本，不包含任何模型权重**。以下是随仓库分发或由运行期下载的第三方内容及其许可状态。

## 1. 随仓库分发的代码

| 组件 | 位置 | 许可 | 说明 |
|---|---|---|---|
| see-through-gguf（STM） | `stm/`（已内置，`.git` 已移除） | **Apache-2.0**，全文见 [LICENSE](LICENSE) | 上游 <https://github.com/IceFog72/see-through-gguf>，基线 commit `02cba07`；本项目的修改见 [MODIFICATIONS.md](MODIFICATIONS.md) |
| See-through（上游的上游） | 概念与论文来源 | 见上游署名要求 | shitagaki-lab/see-through：<https://github.com/shitagaki-lab/see-through>；论文 *See-through: Layer Decomposition for Live2D Character Animation*，<https://arxiv.org/abs/2602.03749> |
| 上游内置的第三方代码 | `stm/common/modules/depth_anything_v2/` 等 | Apache-2.0（原始版权头已保留） | 含 Meta Platforms（DINOv2 / Depth Anything V2）与 Microsoft 的代码，本项目未做改动，也未删除其版权声明 |
| GSAP 3.12.5 | `frontend/vendor/gsap.min.js` | GreenSock Standard License（<https://gsap.com/standard-license>，非 OSI 开源许可） | 允许免费用于本类项目，禁止用于与 GSAP 竞争的产品；文件头版权声明已保留 |
| 本项目自有代码 | `backend/`、`frontend/`（除 vendor）、`*.bat`、`*.md` | **Apache-2.0**，全文见 [LICENSE](LICENSE) 与 [NOTICE](NOTICE) | — |

## 2. 运行期由用户机器下载的模型/权重（本仓库不分发）

许可信息于 2026-10-29 通过 `hf-mirror.com` 的模型 API 核对：

| 模型 | 仓库 | 声明的许可 | 用在哪里 |
|---|---|---|---|
| LayerDiff3D GGUF | `icefog72/seethroughv0.0.2_layerdiff3d_gguf` | **未声明**（API 无 license 字段） | 主模型，图层分解 |
| Marigold3D GGUF | `icefog72/seethroughv0.0.1_marigold_gguf` | **未声明**（API 无 license 字段） | 深度模型 |
| LayerDiff3D 原始权重 | `layerdifforg/seethroughv0.0.2_layerdiff3d` | `openrail++`（CreativeML OpenRAIL++-M） | GGUF 的来源 |
| Marigold 深度模型 | `prs-eth/marigold-depth-v1-1` | `openrail++` | 首次推理下载的文本编码器/tokenizer |
| 动漫实例分割 | `dreMaz/AnimeInstanceSegmentation` | MIT | 首次推理下载的前景/部件分割 |
| LaMa 修补权重 | `dreMaz/AnimeMangaInpainting`（`lama_large_512px.ckpt`，约 195 MiB） | MIT | 开启 `use_lama` 时首次下载 |
| RTMPose | `Tau-J/RTMPose` | Apache-2.0 | 勾选「骨骼切割」时首次下载 |
| anime-seg（ISNet 动漫抠图） | `skytnt/anime-seg`（`isnetis.ckpt`，约 195 MiB） | Apache-2.0 | 源图没有透明通道时抠前景 |

**注意**：

- 两个 GGUF 仓库**没有声明许可证**。因此不能对外断言它们适用某个具体协议；上游 README 也只说明「代码开源、权重从 Hugging Face 下载」。二次分发权重前请自行向模型作者确认。
- 它们派生自 `openrail++` 的模型，OpenRAIL++ 带有使用限制（use restrictions）与署名要求。用这些权重生成的图片若对外发布，需自行遵守相应限制。
- 辅助组件（分割、LaMa、RTMPose、Marigold 文本编码器、anime-seg）是**首次运行时下载到本机**的，默认落在项目内 `components/`（已 `.gitignore`），不进入本仓库；其许可分别如上表。
- Python 依赖（Flask、PyTorch、diffusers、gguf、psd-tools 等）不随仓库分发，由 `requirements*.txt` 安装，各自遵循原有许可。
