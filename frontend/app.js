/* 分层工作台 · SeeThrough GGUF 前端
 * ---------------------------------------------------------------------------
 * 职责：上传校验 → 参数校验（与后端同套字段与范围）→ 提交 → SSE 日志 → 结果渲染
 * 状态：idle / running / done / warn / failed / canceled
 * ------------------------------------------------------------------------- */
(function () {
  "use strict";

  /* ────────────────────────────── i18n ────────────────────────────── */

  const I18N = {
    zh: {
      "app.title": "分层工作台",
      "app.tagline": "单张角色图 → 图层 / 深度 / PSD",

      "model.checking": "检查模型",
      "model.ready": "模型就绪",
      "model.missing": "模型缺失",
      "model.details": "详情",
      "model.layerdiff": "LayerDiff3D",
      "model.depth": "Marigold3D",
      "model.stm": "STM 仓库",
      "model.patch": "LaMa 开关",
      "model.network": "下载源",
      "model.networkNone": "均不可达",
      "model.networkValue": "{name} · {ms} ms",
      "model.present": "已就位",
      "model.absent": "缺失",
      "model.patchOk": "已打补丁",
      "model.patchMissing": "未打补丁",

      "step.image": "图像",
      "step.imageHint": "单张",
      "step.params": "参数",
      "step.paramsHint": "分辨率与采样",
      "step.run": "运行",

      "file.drop": "拖入图片，或点击选择",
      "file.hint": "PNG · JPG · WEBP · BMP，单张最大 {n} MB",
      "file.remove": "移除",
      "file.spec": "{w} × {h} · {size}",

      "field.width": "宽",
      "field.height": "高",
      "field.squareNote": "引擎按正方形推理，宽高同步；需为 64 的倍数",
      "field.steps": "推理步数",
      "field.depth": "深度步数",
      "field.depthNote": "4 为默认",
      "field.depthRes": "深度分辨率",
      "field.depthResAutoHint": "按显存自动",
      "field.depthResFollowHint": "等于主分辨率",
      "field.depthResCustomHint": "64 的倍数",
      "field.depthResAuto": "自动",
      "field.maskRes": "前景分割分辨率",
      "field.maskResNote": "(仅对无透明通道的图生效)",
      "field.maskResAuto": "自动",
      "field.maskResCustom": "自定义",
      "field.attn": "注意力后端",
      "field.attnHint": "native 最稳",
      "field.attnMissing": "xformers 未安装",
      "field.depthResFollow": "跟随主分辨率",
      "field.depthResCustom": "自定义",
      "err.depthResRange": "范围 {min}–{max}",
      "err.depthResStep": "需为 {step} 的倍数",
      "field.depthAuto": "自动",
      "field.seed": "种子",
      "field.seedFixed": "固定",
      "field.seedRandom": "随机",
      "field.seedFixedNote": "固定种子可复现同一结果",
      "field.seedDice": "随机种子",
      "field.seedRandomNote": "每次运行由后端生成新种子并回填",

      "step.models": "模型选择",
      "field.layerModel": "图层模型",
      "field.depthModel": "深度模型",
      "models.note": "扫描 models/ 及其子目录；子目录按完整路径显示，便于区分同名的不同量化等级。",
      "keep.label": "模型常驻",
      "keep.note": "任务结束后保留模型，下一轮免加载",
      "keep.offload": "分组卸载",
      "keep.offloadNote": "按组在 CPU / GPU 间搬权重，更省显存但更慢",
      "models.empty": "models/ 下没有找到 {role} 模型",
      "models.none": "未找到",
      "err.modelNotSelected": "请选择模型",
      "err.modelOutside": "模型必须位于 models/ 目录内",
      "err.modelMissing": "所选模型文件已不存在，请重新扫描",
      "err.modelNotGguf": "只能选择 .gguf 文件",
      "action.rescan": "重新扫描",
      "action.settings": "设置",
      "split.rail": "拖动调整参数栏宽度，双击复位",
      "split.log": "拖动调整日志高度，双击复位",

      "toggle.tblr": "左右对称分割",
      "toggle.lama": "LaMa 修补",
      "toggle.layerFix": "图层修补",
      "toggle.layerFixNote": "修贴边图层被丢、眼白盖住虹膜",
      "toggle.psd": "导出 PSD",
      "toggle.quantized": "GGUF 量化 + 流式加载",
      "toggle.quantizedNote": "关闭后整份权重进显存，6 GB 卡易 OOM",
      "toggle.ragdoll": "骨骼切割",
      "toggle.ragdollNote": "RTMPose 肢体分层；姿态模型已随安装预取",
      "toggle.confidence": "骨骼置信度",
      "toggle.overlap": "关节重叠",
      "toggle.cut": "保留切图与骨架清单",
      "toggle.cutNote": "输出 cuts/ 目录与 manifest.json（轴心、骨骼、父子关系）",
      "field.parts": "输出图层",
      "field.partsNote": "默认全选；取消勾选的图层不会生成，也不会进 PSD",
      "parts.body": "身体组",
      "parts.head": "头部组",
      "parts.bodyNote": "始终整组参与采样，改这里只影响落盘",
      "parts.headNote": "整组跳过，省最多时间",
      "parts.selected": "已选 {n} / {total}",
      "action.partsAll": "全选",
      "action.partsNone": "清空",
      "action.partsBody": "仅身体组",
      "action.partsHead": "仅头部组",
      "toggle.lamaUnavailable": "未打补丁，此开关固定开启",
      "adv.title": "高级选项",

      "status.ready": "就绪",
      "status.running": "推理中",
      "status.canceling": "正在取消",
      "status.done": "已完成",
      "status.warn": "完成 · 有告警",
      "status.fail": "失败",
      "status.canceled": "已取消",

      "action.run": "开始分解",
      "action.running": "推理中",
      "action.cancel": "取消",
      "action.download": "下载",

      "stage.title": "成像台",
      "stage.sub": "推理结果与图层下载",
      "stage.lastRun": "上次任务",
      "stage.empty": "还没有结果",
      "stage.emptyHint": "上传一张角色图，点击「开始分解」",
      "stage.running": "正在推理",
      "stage.runningHint": "低显存设备上可能需要十几分钟，可随时取消",
      "stage.failed": "本次任务未完成",
      "stage.psd": "PSD",
      "stage.meta": "清单",
      "stage.openFolder": "打开文件夹",
      "action.post": "后处理重跑",
      "action.postStart": "开始",
      "post.hint": "不重跑采样，约 1–3 分钟",
      "post.running": "后处理中…",
      "post.done": "完成，产物已更新",
      "post.failed": "失败，请查看日志",
      "post.busy": "有任务在跑，稍后再试",
      "stage.pose": "关节置信度",
      "stage.poseHint": "低于 {v} 的关节会影响骨骼切分",
      "stage.summary": "{n} 个文件 · {res}",

      "log.title": "运行日志",
      "log.follow": "跟随",
      "log.clear": "清空",
      "log.empty": "等待任务开始",

      "run.res": "分辨率",
      "run.steps": "步数",
      "run.seed": "种子",
      "run.elapsed": "耗时",

      "notice.error": "错误",
      "notice.warn": "告警",
      "notice.info": "提示",

      "err.modelsMissing": "缺少推理所需文件：{items}。请按 README 下载 GGUF 并放入 models/。",
      "err.modelItem.stm_repo": "STM 仓库",
      "err.modelItem.layerdiff": "LayerDiff3D GGUF",
      "err.modelItem.depth": "Marigold3D GGUF",
      "err.format": "不支持的文件格式，请选择 PNG / JPG / WEBP / BMP。",
      "err.decode": "这个文件无法解码，可能已损坏。",
      "err.tooLarge": "文件超过 {mb} MB 上限。",
      "err.busy": "已有任务在运行，请等待完成或先取消。",
      "err.noFile": "请先选择一张图片。",
      "err.busyFile": "推理进行中，先取消或等待完成再更换图片。",
      "err.inference": "推理进程异常退出，详细报错见日志。",
      "err.launchFailed": "推理进程无法启动，详细报错见日志。",
      "err.noOutputs": "推理结束但没有任何输出文件，详见日志。",
      "err.psd": "推理已完成，但 PSD 没有生成。图层 PNG 仍可下载。",
      "err.canceled": "任务已取消，临时文件保留在该任务的输出目录。",
      "err.network": "无法连接本地服务，请确认 2-launch.bat 仍在运行。",
      "err.storage": "无法写入任务目录，请检查磁盘空间与权限。",
      "err.generic": "发生未预期的错误。",
      "err.invalidParams": "参数未通过后端校验，请检查标红字段。",

      "err.notInteger": "请填写整数",
      "err.outOfRange": "取值范围 {min}–{max}",
      "err.notMultiple": "必须是 {step} 的倍数",
      "err.mustEqual": "宽高必须相等（引擎按正方形推理）",
      "err.notNumber": "请填写数字",
      "err.patchMissing": "STM 未打补丁，无法关闭 LaMa",
      "err.stepsRange": "步数需在 1–150 之间",
      "err.partsEmpty": "至少选择一个输出图层",

      "level.sys": "SYS",
      "level.info": "INFO",
      "level.ok": "OK",
      "level.warn": "WARN",
      "level.error": "ERR",
      "level.progress": "RUN",

      "log.seedUsed": "实际种子 {seed}",
      "log.fileChanged": "已更换输入图片，旧结果已清除",
      "log.cancelRequest": "已请求取消…",
      "log.elapsed": "耗时 {sec} 秒",
      "misc.unknown": "未知",
    },

    en: {
      "app.title": "Layer Bench",
      "app.tagline": "One character image → layers / depth / PSD",

      "model.checking": "Checking models",
      "model.ready": "Models ready",
      "model.missing": "Models missing",
      "model.details": "Details",
      "model.layerdiff": "LayerDiff3D",
      "model.depth": "Marigold3D",
      "model.stm": "STM repo",
      "model.patch": "LaMa toggle",
      "model.network": "Download source",
      "model.networkNone": "unreachable",
      "model.networkValue": "{name} · {ms} ms",
      "model.present": "present",
      "model.absent": "missing",
      "model.patchOk": "patched",
      "model.patchMissing": "not patched",

      "step.image": "Image",
      "step.imageHint": "single file",
      "step.params": "Parameters",
      "step.paramsHint": "resolution & sampling",
      "step.run": "Run",

      "file.drop": "Drop an image, or click to browse",
      "file.hint": "PNG · JPG · WEBP · BMP, up to {n} MB",
      "file.remove": "Remove",
      "file.spec": "{w} × {h} · {size}",

      "field.width": "Width",
      "field.height": "Height",
      "field.squareNote": "The engine infers square images; height follows width. Multiples of 64.",
      "field.steps": "Inference steps",
      "field.depth": "Depth steps",
      "field.depthNote": "4 by default",
      "field.depthRes": "Depth resolution",
      "field.depthResAutoHint": "Auto by VRAM",
      "field.depthResFollowHint": "Same as main",
      "field.depthResCustomHint": "Multiple of 64",
      "field.depthResAuto": "Auto",
      "field.maskRes": "Foreground mask resolution",
      "field.maskResNote": "(only applies to images without an alpha channel)",
      "field.maskResAuto": "Auto",
      "field.maskResCustom": "Custom",
      "field.attn": "Attention backend",
      "field.attnHint": "native is safest",
      "field.attnMissing": "xformers is not installed",
      "field.depthResFollow": "Follow main",
      "field.depthResCustom": "Custom",
      "err.depthResRange": "Range {min}–{max}",
      "err.depthResStep": "Must be a multiple of {step}",
      "field.depthAuto": "Auto",
      "field.seed": "Seed",
      "field.seedFixed": "Fixed",
      "field.seedRandom": "Random",
      "field.seedFixedNote": "A fixed seed reproduces the same result",
      "field.seedDice": "Random seed",
      "field.seedRandomNote": "A new seed is generated for every run and shown here",

      "step.models": "Models",
      "field.layerModel": "Layer model",
      "field.depthModel": "Depth model",
      "models.note": "Scans models/ and its subfolders; subfolders are shown as full paths so same-named quants stay distinguishable.",
      "keep.label": "Keep model loaded",
      "keep.note": "Keeps the model in VRAM after a run so the next one skips loading",
      "keep.offload": "Group offload",
      "keep.offloadNote": "Moves weight groups between CPU and GPU: less VRAM, slower",
      "models.empty": "No {role} model found under models/",
      "models.none": "none",
      "err.modelNotSelected": "Select a model",
      "err.modelOutside": "Model must live inside models/",
      "err.modelMissing": "The selected model file no longer exists — rescan",
      "err.modelNotGguf": "Only .gguf files can be selected",
      "action.rescan": "Rescan",
      "action.settings": "Settings",
      "split.rail": "Drag to resize the sidebar, double-click to reset",
      "split.log": "Drag to resize the log panel, double-click to reset",

      "toggle.tblr": "Left/right split",
      "toggle.lama": "LaMa inpaint",
      "toggle.layerFix": "Layer fixes",
      "toggle.layerFixNote": "Recovers clipped-edge layers and fixes eye layer order",
      "toggle.psd": "Export PSD",
      "toggle.quantized": "GGUF quantized + streamed",
      "toggle.quantizedNote": "Turning this off loads full weights into VRAM — 6 GB cards will OOM",
      "toggle.ragdoll": "Ragdoll split",
      "toggle.ragdollNote": "RTMPose limb layers; pose models are prefetched by setup",
      "toggle.confidence": "Pose confidence",
      "toggle.overlap": "Joint overlap",
      "toggle.cut": "Keep cut images and rig manifest",
      "toggle.cutNote": "Writes cuts/ plus manifest.json (pivots, bones, parents)",
      "field.parts": "Output layers",
      "field.partsNote": "All selected by default; unchecked layers are not generated and stay out of the PSD",
      "parts.body": "Body group",
      "parts.head": "Head group",
      "parts.bodyNote": "The whole group is always sampled; this only changes what gets written",
      "parts.headNote": "Skip the whole group to save the most time",
      "parts.selected": "Selected {n} / {total}",
      "action.partsAll": "All",
      "action.partsNone": "None",
      "action.partsBody": "Body only",
      "action.partsHead": "Head only",
      "toggle.lamaUnavailable": "STM not patched — this switch is forced on",
      "adv.title": "Advanced",

      "status.ready": "Ready",
      "status.running": "Running",
      "status.canceling": "Cancelling",
      "status.done": "Completed",
      "status.warn": "Completed · warning",
      "status.fail": "Failed",
      "status.canceled": "Cancelled",

      "action.run": "Decompose",
      "action.running": "Running",
      "action.cancel": "Cancel",
      "action.download": "Download",

      "stage.title": "Stage",
      "stage.sub": "Outputs and layer downloads",
      "stage.lastRun": "Previous run",
      "stage.empty": "No result yet",
      "stage.emptyHint": "Upload a character image, then press Decompose",
      "stage.running": "Inferring",
      "stage.runningHint": "Low-VRAM cards may need 10+ minutes at 1024; you can cancel anytime",
      "stage.failed": "This run did not finish",
      "stage.psd": "PSD",
      "stage.meta": "Manifests",
      "stage.openFolder": "Open folder",
      "action.post": "Re-post-process",
      "action.postStart": "Start",
      "post.hint": "No sampling, about 1–3 min",
      "post.running": "Post-processing…",
      "post.done": "Done, outputs updated",
      "post.failed": "Failed, check the log",
      "post.busy": "Another task is running, try later",
      "stage.pose": "Pose confidence",
      "stage.poseHint": "joints below {v} affect the ragdoll split",

      "stage.summary": "{n} files · {res}",

      "log.title": "Run log",
      "log.follow": "Follow",
      "log.clear": "Clear",
      "log.empty": "Waiting for a run",

      "run.res": "Resolution",
      "run.steps": "Steps",
      "run.seed": "Seed",
      "run.elapsed": "Elapsed",

      "notice.error": "ERROR",
      "notice.warn": "WARNING",
      "notice.info": "NOTICE",

      "err.modelsMissing": "Missing files: {items}. Download the GGUF files into models/ as described in README.",
      "err.modelItem.stm_repo": "STM repo",
      "err.modelItem.layerdiff": "LayerDiff3D GGUF",
      "err.modelItem.depth": "Marigold3D GGUF",
      "err.format": "Unsupported file type. Use PNG / JPG / WEBP / BMP.",
      "err.decode": "This file cannot be decoded — it may be corrupted.",
      "err.tooLarge": "File exceeds the {mb} MB limit.",
      "err.busy": "A job is already running. Wait for it to finish or cancel it first.",
      "err.noFile": "Select an image first.",
      "err.busyFile": "A run is in progress — cancel it or wait before changing the image.",
      "err.inference": "The inference process exited abnormally. See the log for details.",
      "err.launchFailed": "The inference process could not start. See the log for details.",
      "err.noOutputs": "Inference finished but produced no files. See the log.",
      "err.psd": "Inference completed, but no PSD was produced. Layer PNGs are still downloadable.",
      "err.canceled": "Job cancelled. Temporary files remain in that job's output folder.",
      "err.network": "Cannot reach the local service. Is 2-launch.bat still running?",
      "err.storage": "Cannot write the job folder. Check disk space and permissions.",
      "err.generic": "Unexpected error.",
      "err.invalidParams": "The server rejected the parameters. Check the highlighted fields.",

      "err.notInteger": "Enter a whole number",
      "err.outOfRange": "Allowed range {min}–{max}",
      "err.notMultiple": "Must be a multiple of {step}",
      "err.mustEqual": "Width and height must match (the engine infers square images)",
      "err.notNumber": "Enter a number",
      "err.patchMissing": "STM is not patched, LaMa cannot be disabled",
      "err.stepsRange": "Steps must be between 1 and 150",
      "err.partsEmpty": "Select at least one output layer",

      "level.sys": "SYS",
      "level.info": "INFO",
      "level.ok": "OK",
      "level.warn": "WARN",
      "level.error": "ERR",
      "level.progress": "RUN",

      "log.seedUsed": "Seed used: {seed}",
      "log.fileChanged": "Input image changed — previous result cleared",
      "log.cancelRequest": "Cancellation requested…",
      "log.elapsed": "Elapsed {sec}s",
      "misc.unknown": "unknown",
    },
  };

  const LS_LANG = "stg.lang";
  let lang = localStorage.getItem(LS_LANG) === "en" ? "en" : "zh";

  function t(key, vars) {
    const table = I18N[lang] || I18N.zh;
    let text = table[key] != null ? table[key] : I18N.zh[key] != null ? I18N.zh[key] : key;
    if (vars) {
      Object.keys(vars).forEach((k) => {
        text = text.replace(new RegExp("\\{" + k + "\\}", "g"), String(vars[k]));
      });
    }
    return text;
  }

  /* ────────────────────────────── 状态 ────────────────────────────── */

  const el = (id) => document.getElementById(id);

  const state = {
    config: null,
    models: null,
    patchOk: true,
    file: null,
    fileUrl: null,
    running: false,
    jobId: null,
    seedMode: "fixed",
    attnBackend: "native",
    lastResults: null,
    fixedSeed: 42,
    logCount: 0,
    es: null,
    timer: null,
    startedAt: 0,
    lastProgressNode: null,
    lastSeq: 0,
    pollTimer: null,
    finalized: null,
    activeParams: null,
    lastElapsed: null,
    lastOutputCount: null,
    lastFailDetail: null,
    valid: true,
    noticeSticky: false,
  };

  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  /* ────────────────────────────── 工具 ────────────────────────────── */

  const pad = (n) => String(n).padStart(2, "0");

  function clockTime(epochMs) {
    const d = epochMs ? new Date(epochMs) : new Date();
    return `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
  }

  function formatBytes(bytes) {
    if (!bytes && bytes !== 0) return "—";
    if (bytes < 1024) return bytes + " B";
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
    if (bytes < 1024 ** 3) return (bytes / 1024 / 1024).toFixed(1) + " MB";
    return (bytes / 1024 ** 3).toFixed(2) + " GB";
  }

  function formatDuration(seconds) {
    const s = Math.max(0, Math.round(seconds));
    const m = Math.floor(s / 60);
    return m > 0 ? `${m}m ${pad(s % 60)}s` : `${s}s`;
  }

  function escapeHtml(str) {
    return String(str).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  }

  /* ────────────────────────────── 文案 ────────────────────────────── */

  function applyI18n() {
    document.documentElement.lang = lang === "zh" ? "zh-CN" : "en";
    document.title = t("app.title") + " · SeeThrough GGUF";

    document.querySelectorAll("[data-i18n]").forEach((node) => {
      node.textContent = t(node.getAttribute("data-i18n"));
    });

    document.querySelectorAll("[data-i18n-ph]").forEach((node) => {
      node.setAttribute("placeholder", t(node.getAttribute("data-i18n-ph")));
    });

    document.querySelectorAll("[data-i18n-aria]").forEach((node) => {
      node.setAttribute("aria-label", t(node.getAttribute("data-i18n-aria")));
    });

    document.querySelectorAll("[data-lang-opt]").forEach((node) => {
      node.setAttribute("aria-current", node.getAttribute("data-lang-opt") === lang ? "true" : "false");
    });

    el("dropHint").textContent = t("file.hint", { n: state.config ? state.config.max_upload_mb : 50 });
    el("seedNote").textContent = t(state.seedMode === "fixed" ? "field.seedFixedNote" : "field.seedRandomNote");
    el("lamaNote").textContent = state.patchOk ? "use_lama" : t("toggle.lamaUnavailable");

    if (state.models) renderModels(state.models);
    if (state.file) renderFileSpec();

    // 已产出的结果摘要与失败说明也要跟着换语言
    if (state.lastOutputCount != null && state.activeParams) {
      const res = `${state.activeParams.width} × ${state.activeParams.height}`;
      el("stageSub").textContent = t("stage.summary", { n: state.lastOutputCount, res });
    } else {
      el("stageSub").textContent = t("stage.sub");
    }
    if (state.lastFailDetail != null && !el("stageFailed").hidden) {
      el("failedHint").textContent = t("err.inference") + ` (${state.lastFailDetail})`;
    }

    updateRunStateLabel();
    updateFacts();
    refreshRunButton();
  }

  /* ────────────────────────────── 提示条 ────────────────────────────── */

  let noticeTimer = null;

  function showNotice(kind, message, sticky) {
    const box = el("notice");
    state.noticeSticky = !!sticky;
    const tagKey = kind === "warn" ? "notice.warn" : kind === "info" ? "notice.info" : "notice.error";
    el("noticeTag").textContent = t(tagKey);
    el("noticeMsg").textContent = message;
    box.dataset.kind = kind;
    box.hidden = false;
    clearTimeout(noticeTimer);
    if (!reduceMotion && window.gsap) {
      gsap.fromTo(box, { autoAlpha: 0, y: -6 }, { autoAlpha: 1, y: 0, duration: 0.3, ease: "power2.out" });
    }
  }

  function hideNotice(force) {
    if (state.noticeSticky && !force) return;
    el("notice").hidden = true;
  }

  el("noticeClose").addEventListener("click", hideNotice);

  /* ────────────────────────────── 日志 ────────────────────────────── */

  const LOG_LIMIT = 1500;
  const LEVEL_LABEL = {
    sys: "level.sys",
    info: "level.info",
    ok: "level.ok",
    warn: "level.warn",
    error: "level.error",
    progress: "level.progress",
  };

  function appendLog(entry) {
    // 序号去重：SSE 与轮询可能同时送达同一条日志
    if (entry.seq != null) {
      if (entry.seq <= state.lastSeq) return;
      state.lastSeq = entry.seq;
    }
    const level = LEVEL_LABEL[entry.level] ? entry.level : "info";
    const body = el("logBody");
    const idle = el("logIdle");
    if (idle) idle.remove();

    const normalized = String(entry.msg == null ? "" : entry.msg).replace(/\s+$/, "");
    if (!normalized) return;

    // 进度类日志复用一个节点，避免刷屏把有效信息顶走
    if (level === "progress") {
      if (state.lastProgressNode && state.lastProgressNode.parentNode === body) {
        state.lastProgressNode.querySelector(".log-msg").textContent = normalized;
        state.lastProgressNode.querySelector(".log-time").textContent = clockTime(entry.ts);
        followLog();
        return;
      }
    }

    const line = document.createElement("div");
    line.className = "log-line";
    line.dataset.level = level;

    const time = document.createElement("span");
    time.className = "log-time";
    time.textContent = clockTime(entry.ts);

    const tag = document.createElement("span");
    tag.className = "log-level";
    tag.textContent = t(LEVEL_LABEL[level]);

    const msg = document.createElement("span");
    msg.className = "log-msg";
    msg.textContent = normalized;

    line.append(time, tag, msg);
    body.appendChild(line);

    state.lastProgressNode = level === "progress" ? line : null;
    state.logCount += 1;
    el("logCount").textContent = String(state.logCount);

    while (body.childElementCount > LOG_LIMIT) {
      body.firstElementChild.remove();
    }
    followLog();
  }

  function followLog() {
    if (!el("logFollow").checked) return;
    const body = el("logBody");
    body.scrollTop = body.scrollHeight;
  }

  function clearLog() {
    const body = el("logBody");
    body.innerHTML = "";
    const idle = document.createElement("p");
    idle.className = "console-idle";
    idle.id = "logIdle";
    idle.setAttribute("data-i18n", "log.empty");
    idle.textContent = t("log.empty");
    body.appendChild(idle);
    state.logCount = 0;
    state.lastProgressNode = null;
    el("logCount").textContent = "0";
  }

  el("logClear").addEventListener("click", clearLog);

  /* ────────────────────────────── 模型检查 ────────────────────────────── */

  function renderModels(data) {
    const items = data.items || {};
    const list = el("modelList");
    list.innerHTML = "";

    const layerRel = selectedModel("layerdiff");
    const depthRel = selectedModel("depth");
    const nameOf = (rel) => (rel ? rel.split("/").pop() : t("models.none"));

    const rows = [
      ["model.layerdiff", layerRel ? { exists: true, path: modelFullPath(layerRel) } : { exists: false, path: "" }, nameOf(layerRel)],
      ["model.depth", depthRel ? { exists: true, path: modelFullPath(depthRel) } : { exists: false, path: "" }, nameOf(depthRel)],
      ["model.stm", items.stm_repo, items.stm_repo && items.stm_repo.exists ? t("model.present") : t("model.absent")],
    ];

    rows.forEach(([labelKey, payload, stateText]) => {
      const row = document.createElement("div");
      row.className = "row";
      const dt = document.createElement("dt");
      dt.textContent = t(labelKey);
      const dd = document.createElement("dd");
      const ok = payload && payload.exists;
      dd.className = ok ? "state-ok" : "state-bad";
      dd.textContent = stateText;
      dd.title = payload && payload.path ? payload.path : "";
      row.append(dt, dd);
      list.appendChild(row);
    });

    const patchRow = document.createElement("div");
    patchRow.className = "row";
    const pdt = document.createElement("dt");
    pdt.textContent = t("model.patch");
    const pdd = document.createElement("dd");
    const patched = !!(items.use_lama_patch && items.use_lama_patch.applied);
    pdd.className = patched ? "state-ok" : "state-warn";
    pdd.textContent = patched ? t("model.patchOk") : t("model.patchMissing");
    patchRow.append(pdt, pdd);
    list.appendChild(patchRow);

    // 下载源：启动时探测的结果
    const net = data.network;
    const netRow = document.createElement("div");
    netRow.className = "row";
    const ndt = document.createElement("dt");
    ndt.textContent = t("model.network");
    const ndd = document.createElement("dd");
    if (net && net.chosen) {
      const item = (net.results || {})[net.chosen] || {};
      ndd.className = "state-ok";
      ndd.textContent = t("model.networkValue", { name: net.chosen, ms: item.latency_ms ?? "?" });
      ndd.title = net.endpoint || "";
    } else {
      ndd.className = "state-warn";
      ndd.textContent = t("model.networkNone");
    }
    netRow.append(ndt, ndd);
    list.appendChild(netRow);

    updateReadout();
  }

  function loadModels() {
    return fetch("/api/check_models")
      .then((r) => r.json())
      .then((data) => {
        state.models = data;
        state.patchOk = !!(data.items && data.items.use_lama_patch && data.items.use_lama_patch.applied);
        el("modelDot").dataset.state = "checking";
        renderModels(data);

        if (!state.patchOk) {
          const lama = el("useLama");
          lama.checked = true;
          lama.disabled = true;
          el("lamaSwitch").classList.add("is-disabled");
        }

        updateReadout();
        updateModelNotice();
        refreshRunButton();
        return data;
      })
      .catch(() => {
        el("modelDot").dataset.state = "missing";
        el("modelText").textContent = t("model.missing");
        showNotice("error", t("err.network"));
        refreshRunButton();
        return null;
      });
  }

  const MODEL_SELECTS = { layerdiff: "layerModel", depth: "depthModel" };

  function selectedModel(role) {
    const node = el(MODEL_SELECTS[role]);
    return node ? node.value : "";
  }

  function modelFullPath(rel) {
    const base = (state.modelInfo && state.modelInfo.models_dir) || "";
    return rel ? (base ? base + "\\" + rel.replace(/\//g, "\\") : rel) : "";
  }

  function modelOptions(role) {
    const all = (state.modelInfo && state.modelInfo.models) || [];
    const filtered = all.filter((m) => (role === "layerdiff" ? m.role !== "depth" : m.role !== "layerdiff"));
    return filtered.length ? filtered : all;
  }

  function renderModelSelects() {
    [
      { role: "layerdiff", select: "layerModel", size: "layerModelSize" },
      { role: "depth", select: "depthModel", size: "depthModelSize" },
    ].forEach(({ role, select, size }) => {
      const node = el(select);
      const options = modelOptions(role);
      const previous = node.value || localStorage.getItem("stg.model." + role) || "";
      node.innerHTML = "";
      if (!options.length) {
        const opt = document.createElement("option");
        opt.value = "";
        opt.textContent = t("models.none");
        node.appendChild(opt);
        node.disabled = true;
        el(size).textContent = t("models.none");
        return;
      }
      node.disabled = false;
      options.forEach((m) => {
        const opt = document.createElement("option");
        opt.value = m.rel;
        // 子目录里的模型显示完整相对路径，同名的不同量化才区分得开
        opt.textContent = m.rel;
        opt.title = m.rel;
        node.appendChild(opt);
      });
      const defaults = (state.modelInfo && state.modelInfo.defaults) || {};
      const preferred = [previous, defaults[role]].find((v) => v && options.some((m) => m.rel === v));
      node.value = preferred || options[0].rel;
      const entry = options.find((m) => m.rel === node.value);
      el(size).textContent = entry ? formatBytes(entry.size) : t("models.none");
      localStorage.setItem("stg.model." + role, node.value);
    });

    if (state.models) renderModels(state.models);
    updateModelNotice();
    validateInline();
  }

  function updateReadout() {
    const items = (state.models && state.models.items) || {};
    const stmOk = !!(items.stm_repo && items.stm_repo.exists);
    const ready = stmOk && !!selectedModel("layerdiff") && !!selectedModel("depth");
    el("modelDot").dataset.state = ready ? "ready" : "missing";
    el("modelText").textContent = ready ? t("model.ready") : t("model.missing");
  }

  function updateModelNotice() {
    const missing = [];
    if (!selectedModel("layerdiff")) missing.push(t("field.layerModel"));
    if (!selectedModel("depth")) missing.push(t("field.depthModel"));
    if (missing.length) {
      showNotice("error", t("models.empty", { role: missing.join(" / ") }), true);
    } else if (state.noticeSticky) {
      state.noticeSticky = false;
      hideNotice(true);
    }
  }

  function loadModelFiles() {
    return fetch("/api/models")
      .then((r) => r.json())
      .then((data) => {
        state.modelInfo = data;
        renderModelSelects();
      })
      .catch(() => {
        showNotice("error", t("err.network"));
      });
  }

  el("modelRescan").addEventListener("click", () => {
    el("modelRescan").disabled = true;
    loadModelFiles().finally(() => {
      el("modelRescan").disabled = false;
    });
  });

  [["layerModel", "layerdiff"], ["depthModel", "depth"]].forEach(([id, role]) => {
    el(id).addEventListener("change", () => {
      const value = el(id).value;
      if (value) localStorage.setItem("stg.model." + role, value);
      const entry = modelOptions(role).find((m) => m.rel === value);
      el(role === "layerdiff" ? "layerModelSize" : "depthModelSize").textContent = entry
        ? formatBytes(entry.size)
        : t("models.none");
      setFieldError(role === "layerdiff" ? "modelLayer" : "modelDepth", "");
      if (state.models) renderModels(state.models);
      updateModelNotice();
      validateInline();
    });
  });

  el("modelMore").addEventListener("click", () => {
    const box = el("modelDetail");
    box.hidden = !box.hidden;
  });

  document.addEventListener("click", (event) => {
    const box = el("modelDetail");
    if (box.hidden) return;
    if (!box.contains(event.target) && event.target !== el("modelMore")) box.hidden = true;
  });

  /* ────────────────────────────── 上传 ────────────────────────────── */

  const drop = el("drop");
  const fileInput = el("fileInput");

  function allowedExts() {
    const list = (state.config && state.config.supported_ext) || [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"];
    return list.map((x) => x.toLowerCase());
  }

  function clearFileError() {
    const err = el("fileError");
    err.hidden = true;
    err.textContent = "";
    drop.classList.remove("is-invalid");
  }

  function setFileError(message) {
    const err = el("fileError");
    err.textContent = message;
    err.hidden = false;
  }

  function resetFile(keepNotice) {
    state.file = null;
    if (state.fileUrl) {
      URL.revokeObjectURL(state.fileUrl);
      state.fileUrl = null;
    }
    el("dropPreview").hidden = true;
    el("dropPreview").removeAttribute("src");
    el("dropInner").hidden = false;
    el("fileCard").hidden = true;
    fileInput.value = "";
    clearFileError();
    if (!keepNotice) hideNotice();
    refreshRunButton();
  }

  function renderFileSpec() {
    if (!state.file) return;
    el("fileSpec").textContent = t("file.spec", {
      w: state.file.width,
      h: state.file.height,
      size: formatBytes(state.file.size),
    });
  }

  function acceptFile(file) {
    if (!file) return Promise.resolve(false);
    if (state.running) {
      showNotice("warn", t("err.busyFile"));
      fileInput.value = "";
      return Promise.resolve(false);
    }
    clearFileError();

    const ext = "." + (file.name.split(".").pop() || "").toLowerCase();
    if (!allowedExts().includes(ext)) {
      setFileError(t("err.format"));
      return Promise.resolve(false);
    }

    const maxMb = (state.config && state.config.max_upload_mb) || 50;
    if (file.size > maxMb * 1024 * 1024) {
      setFileError(t("err.tooLarge", { mb: maxMb }));
      return Promise.resolve(false);
    }

    return new Promise((resolve) => {
      const url = URL.createObjectURL(file);
      const probe = new Image();
      probe.onload = () => {
        if (!probe.naturalWidth || !probe.naturalHeight) {
          URL.revokeObjectURL(url);
          setFileError(t("err.decode"));
          resolve(false);
          return;
        }
        commitFile(file, url, probe.naturalWidth, probe.naturalHeight);
        resolve(true);
      };
      probe.onerror = () => {
        URL.revokeObjectURL(url);
        setFileError(t("err.decode"));
        resolve(false);
      };
      probe.src = url;
    });
  }

  function commitFile(file, url, width, height) {
    if (state.fileUrl && state.fileUrl.startsWith("blob:")) URL.revokeObjectURL(state.fileUrl);
    state.fileUrl = url;
    state.file = { file, width, height, size: file.size };

    el("dropPreview").src = url;
    el("dropPreview").hidden = false;
    el("dropInner").hidden = true;
    el("fileThumb").src = url;
    el("fileName").textContent = file.name;
    el("fileName").title = file.name;
    el("fileCard").hidden = false;
    renderFileSpec();

    // 换图即清空上一张的结果与任务痕迹
    resetResultArea();
    if (state.jobId) {
      appendLog({ ts: Date.now(), level: "sys", msg: t("log.fileChanged") });
      state.jobId = null;
    }
    hideNotice();
    refreshRunButton();
  }

  drop.addEventListener("click", () => fileInput.click());
  drop.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      fileInput.click();
    }
  });

  fileInput.addEventListener("change", (event) => {
    const file = event.target.files && event.target.files[0];
    acceptFile(file);
  });

  ["dragenter", "dragover"].forEach((name) => {
    drop.addEventListener(name, (event) => {
      event.preventDefault();
      event.stopPropagation();
      drop.classList.add("is-drag");
    });
  });

  ["dragleave", "dragend"].forEach((name) => {
    drop.addEventListener(name, (event) => {
      event.preventDefault();
      if (name === "dragleave" && drop.contains(event.relatedTarget)) return;
      drop.classList.remove("is-drag");
    });
  });

  drop.addEventListener("drop", (event) => {
    event.preventDefault();
    event.stopPropagation();
    drop.classList.remove("is-drag");
    const file = event.dataTransfer && event.dataTransfer.files && event.dataTransfer.files[0];
    if (file) acceptFile(file);
  });

  // 防止把图片拖到页面空白处时浏览器直接打开它
  window.addEventListener("dragover", (e) => e.preventDefault());
  window.addEventListener("drop", (e) => {
    if (!drop.contains(e.target)) e.preventDefault();
  });

  el("fileRemove").addEventListener("click", () => {
    resetFile();
    resetResultArea();
    state.jobId = null;
  });

  /* ────────────────────────────── 参数 ────────────────────────────── */

  const widthInput = el("width");
  const heightInput = el("height");
  const stepsInput = el("steps");
  const stepsRange = el("stepsRange");
  const seedInput = el("seed");

  function constraints() {
    return (state.config && state.config.constraints) || {
      resolution_min: 256,
      resolution_max: 2048,
      resolution_step: 64,
      steps_min: 1,
      steps_max: 150,
      depth_steps_max: 100,
      seed_max: 2147483647,
      ragdoll_confidence_min: 0.05,
      ragdoll_confidence_max: 0.95,
      ragdoll_overlap_min: 0,
      ragdoll_overlap_max: 0.5,
      mask_resolution_min: 256,
      mask_resolution_max: 4096,
    };
  }

  /* 输出图层（--parts）：与 STM 的 tag 一一对应，顺序即上游顺序 */
  const PART_TAGS = {
    body: ["front hair", "back hair", "head", "neck", "neckwear", "topwear", "handwear", "bottomwear", "legwear", "footwear", "tail", "wings", "objects"],
    head: ["headwear", "face", "irides", "eyebrow", "eyewhite", "eyelash", "eyewear", "ears", "earwear", "nose", "mouth"],
  };
  const PART_TOTAL = PART_TAGS.body.length + PART_TAGS.head.length;

  function partInputs() {
    return Array.from(document.querySelectorAll("#partsBodyChips input, #partsHeadChips input"));
  }

  function buildParts() {
    [["body", "partsBodyChips"], ["head", "partsHeadChips"]].forEach(([group, id]) => {
      const box = el(id);
      if (!box) return;
      box.innerHTML = "";
      PART_TAGS[group].forEach((tag) => {
        const label = document.createElement("label");
        label.className = "parts-chip";
        const input = document.createElement("input");
        input.type = "checkbox";
        input.checked = true;
        input.dataset.group = group;
        input.dataset.tag = tag;
        input.addEventListener("change", syncPartsCount);
        label.appendChild(input);
        label.appendChild(document.createTextNode(tag));
        box.appendChild(label);
      });
    });
    syncPartsCount();
  }

  function selectedParts() {
    return partInputs().filter((node) => node.checked).map((node) => node.dataset.tag);
  }

  function syncPartsCount() {
    const body = partInputs().filter((node) => node.checked && node.dataset.group === "body").length;
    const head = partInputs().filter((node) => node.checked && node.dataset.group === "head").length;
    el("partsBodyCount").textContent = `${body} / ${PART_TAGS.body.length}`;
    el("partsHeadCount").textContent = `${head} / ${PART_TAGS.head.length}`;
    el("partsSelected").textContent = t("parts.selected", { n: body + head, total: PART_TOTAL });
    validateInline();
  }

  function setParts(mode) {
    partInputs().forEach((node) => {
      node.checked =
        mode === "all" ||
        (mode === "body" && node.dataset.group === "body") ||
        (mode === "head" && node.dataset.group === "head");
    });
    syncPartsCount();
  }

  function applyPartsDefault(parts) {
    const wanted = Array.isArray(parts) && parts.length ? new Set(parts) : null;
    partInputs().forEach((node) => {
      node.checked = wanted ? wanted.has(node.dataset.tag) : true;
    });
    syncPartsCount();
  }


  const attnButtons = () => Array.from(el("attnBackend").querySelectorAll("button"));

  function setAttnBackend(value) {
    state.attnBackend = value === "xformers" ? "xformers" : "native";
    attnButtons().forEach((btn) => {
      btn.setAttribute("aria-checked", btn.dataset.value === state.attnBackend ? "true" : "false");
    });
  }

  function syncAttnAvailability() {
    const available = (state.config && state.config.attention_backends) || { native: true, xformers: false };
    const preferred = (state.config && state.config.attention_backend) || "native";
    attnButtons().forEach((btn) => {
      const usable = available[btn.dataset.value] !== false;
      btn.disabled = !usable;
      btn.title = usable ? "" : t("field.attnMissing");
    });
    setAttnBackend(available[preferred] === false ? "native" : preferred);
    const usableNames = ["native", "xformers"].filter((key) => available[key] !== false);
    el("attnHint").textContent = usableNames.join(" / ");
    if (available.xformers === false) el("attnHint").textContent = t("field.attnMissing");
  }

  attnButtons().forEach((btn) => {
    btn.addEventListener("click", () => {
      if (btn.disabled) return;
      setAttnBackend(btn.dataset.value);
      validateInline();
    });
  });

  function applyDefaults() {
    const d = (state.config && state.config.defaults) || {};
    const c = constraints();

    widthInput.value = d.width != null ? d.width : 1024;
    heightInput.value = d.height != null ? d.height : 1024;
    stepsInput.value = d.inference_steps != null ? d.inference_steps : 30;
    el("depthSteps").value = String(d.inference_steps_depth != null ? d.inference_steps_depth : -1);
    seedInput.value = d.seed != null ? d.seed : 42;
    state.fixedSeed = Number(seedInput.value) || 42;

    el("tblrSplit").checked = d.tblr_split !== false;
    el("useLama").checked = d.use_lama !== false;
    el("savePsd").checked = d.save_psd !== false;
    el("ragdoll").checked = d.ragdoll_split === true;
    el("quantized").checked = true;
    el("ragdollConf").value = d.ragdoll_confidence != null ? d.ragdoll_confidence : 0.3;
    el("ragdollOverlap").value = d.ragdoll_overlap != null ? d.ragdoll_overlap : 0.2;
    el("groupOffload").checked = d.group_offload === true;
    el("cutImages").checked = d.save_cut_images !== false;
    el("layerFix").checked = d.layer_fix !== false;
    syncAttnAvailability();
    el("maskResMode").value = "auto";
    el("maskResRow").hidden = true;
    applyPartsDefault(d.parts);

    el("ragdollConf").min = c.ragdoll_confidence_min;
    el("ragdollConf").max = c.ragdoll_confidence_max;
    el("ragdollOverlap").min = c.ragdoll_overlap_min;
    el("ragdollOverlap").max = c.ragdoll_overlap_max;
    stepsInput.min = c.steps_min;
    stepsInput.max = c.steps_max;
    stepsRange.min = c.steps_min;
    stepsRange.max = c.steps_max;
    stepsRange.value = stepsInput.value;
    [widthInput, heightInput].forEach((node) => {
      node.min = c.resolution_min;
      node.max = c.resolution_max;
      node.step = c.resolution_step;
    });
    el("stepsRangeLabel").textContent = `${c.steps_min}–${c.steps_max}`;

    syncConfidenceOutput();
    syncOverlapOutput();
    updateFacts();
  }

  function syncRangeFromNumber() {
    const value = Number(stepsInput.value);
    if (Number.isFinite(value)) stepsRange.value = String(value);
  }

  function syncNumberFromRange() {
    stepsInput.value = stepsRange.value;
    validateInline();
    updateFacts();
  }

  function syncConfidenceOutput() {
    el("ragdollConfOut").textContent = Number(el("ragdollConf").value).toFixed(2);
  }

  function syncOverlapOutput() {
    el("ragdollOverlapOut").textContent = Number(el("ragdollOverlap").value).toFixed(2);
  }

  function syncResolution(source) {
    const target = source === widthInput ? heightInput : widthInput;
    target.value = source.value;
  }

  widthInput.addEventListener("input", () => {
    syncResolution(widthInput);
    validateInline();
    updateFacts();
  });
  heightInput.addEventListener("input", () => {
    syncResolution(heightInput);
    validateInline();
    updateFacts();
  });
  stepsInput.addEventListener("input", () => {
    syncRangeFromNumber();
    validateInline();
    updateFacts();
  });
  stepsRange.addEventListener("input", syncNumberFromRange);
  el("ragdollConf").addEventListener("input", syncConfidenceOutput);
  el("ragdollOverlap").addEventListener("input", syncOverlapOutput);

  [["partsAll", "all"], ["partsNone", "none"], ["partsBody", "body"], ["partsHead", "head"]].forEach(([id, mode]) => {
    el(id).addEventListener("click", () => setParts(mode));
  });

  [widthInput, heightInput, stepsInput].forEach((node) => {
    node.addEventListener("blur", validateInline);
  });

  /* 种子 */
  function setSeedMode(mode) {
    state.seedMode = mode === "random" ? "random" : "fixed";
    el("seedMode")
      .querySelectorAll("button")
      .forEach((btn) => btn.setAttribute("aria-checked", btn.dataset.mode === state.seedMode ? "true" : "false"));

    const random = state.seedMode === "random";
    if (!random) {
      seedInput.value = state.fixedSeed;
      seedInput.disabled = false;
      el("seedDice").disabled = false;
    } else {
      state.fixedSeed = Number(seedInput.value) || state.fixedSeed;
      seedInput.disabled = true;
      el("seedDice").disabled = true;
    }
    el("seedNote").textContent = t(random ? "field.seedRandomNote" : "field.seedFixedNote");
    validateInline();
  }

  el("seedMode").addEventListener("click", (event) => {
    const btn = event.target.closest("button[data-mode]");
    if (!btn) return;
    setSeedMode(btn.dataset.mode);
  });

  seedInput.addEventListener("input", () => {
    if (state.seedMode === "fixed") {
      state.fixedSeed = Number(seedInput.value);
      validateInline();
    }
  });

  el("seedDice").addEventListener("click", () => {
    if (state.seedMode !== "fixed") return;
    const max = constraints().seed_max;
    state.fixedSeed = Math.floor(Math.random() * Math.min(max, 2147483647));
    seedInput.value = String(state.fixedSeed);
    validateInline();
  });

  /* ────────────────────────────── 校验 ────────────────────────────── */

  const fieldErrors = {
    res: el("resError"),
    steps: el("stepsError"),
    seed: el("seedError"),
    modelLayer: el("layerModelError"),
    modelDepth: el("depthModelError"),
    depthRes: el("depthResError"),
    maskRes: el("maskResError"),
    attn: el("attnError"),
    parts: el("partsError"),
  };

  function setFieldError(name, message) {
    const node = fieldErrors[name];
    if (!node) return;
    node.textContent = message || "";
    node.hidden = !message;
  }

  function markInvalid(input, invalid) {
    input.classList.toggle("is-invalid", !!invalid);
    input.setAttribute("aria-invalid", invalid ? "true" : "false");
  }

  const depthResMode = el("depthResMode");
  const depthResInput = el("depthRes");

  function depthResolutionParam() {
    const mode = depthResMode.value;
    if (mode === "custom") return Number(depthResInput.value);
    if (mode === "-1") return -1;
    return "auto";
  }

  function syncDepthResUi() {
    const mode = depthResMode.value;
    const custom = mode === "custom";
    el("depthResRow").hidden = !custom;
    if (custom && !depthResInput.value) depthResInput.value = String(constraints().default_resolution_depth || 768);
    el("depthResHint").textContent = custom
      ? t("field.depthResCustomHint")
      : mode === "-1"
        ? t("field.depthResFollowHint")
        : t("field.depthResAutoHint");
    validateInline();
  }

  depthResMode.addEventListener("change", syncDepthResUi);
  depthResInput.addEventListener("input", validateInline);
  depthResInput.addEventListener("blur", validateInline);

  const maskResMode = el("maskResMode");
  const maskResInput = el("maskRes");

  function maskResolutionParam() {
    return maskResMode.value === "custom" ? Number(maskResInput.value) : "auto";
  }

  function syncMaskResUi() {
    const custom = maskResMode.value === "custom";
    el("maskResRow").hidden = !custom;
    if (custom && !maskResInput.value) maskResInput.value = "1024";
    validateInline();
  }

  maskResMode.addEventListener("change", syncMaskResUi);
  maskResInput.addEventListener("input", validateInline);
  maskResInput.addEventListener("blur", validateInline);

  function validateInline() {
    const c = constraints();
    let ok = true;
    // 分辨率
    const w = Number(widthInput.value);
    const h = Number(heightInput.value);
    let resMsg = "";
    if (!Number.isInteger(w) || !Number.isInteger(h) || widthInput.value === "" || heightInput.value === "") {
      resMsg = t("err.notInteger");
    } else if (w < c.resolution_min || w > c.resolution_max) {
      resMsg = t("err.outOfRange", { min: c.resolution_min, max: c.resolution_max });
    } else if (w % c.resolution_step !== 0) {
      resMsg = t("err.notMultiple", { step: c.resolution_step });
    } else if (w !== h) {
      resMsg = t("err.mustEqual");
    }
    setFieldError("res", resMsg);
    markInvalid(widthInput, !!resMsg);
    markInvalid(heightInput, !!resMsg);
    if (resMsg) ok = false;

    // 步数
    const steps = Number(stepsInput.value);
    let stepsMsg = "";
    if (!Number.isInteger(steps) || stepsInput.value === "") stepsMsg = t("err.notInteger");
    else if (steps < c.steps_min || steps > c.steps_max) stepsMsg = t("err.stepsRange");
    setFieldError("steps", stepsMsg);
    markInvalid(stepsInput, !!stepsMsg);
    if (stepsMsg) ok = false;

    // 种子
    let seedMsg = "";
    if (state.seedMode === "fixed") {
      const seed = Number(seedInput.value);
      if (seedInput.value === "" || !Number.isInteger(seed)) seedMsg = t("err.notInteger");
      else if (seed < 0 || seed > c.seed_max) seedMsg = t("err.outOfRange", { min: 0, max: c.seed_max });
    }
    setFieldError("seed", seedMsg);
    markInvalid(seedInput, !!seedMsg);
    if (seedMsg) ok = false;

    // 深度分辨率（仅自定义模式需要校验）
    let depthResMsg = "";
    if (depthResMode.value === "custom") {
      const rawValue = depthResInput.value.trim();
      const value = Number(rawValue);
      if (!rawValue || !Number.isInteger(value)) {
        depthResMsg = t("err.notInteger");
      } else if (value < c.resolution_min || value > c.resolution_max) {
        depthResMsg = t("err.outOfRange", { min: c.resolution_min, max: c.resolution_max });
      } else if (value % c.resolution_step !== 0) {
        depthResMsg = t("err.notMultiple", { step: c.resolution_step });
      }
    }
    setFieldError("depthRes", depthResMsg);
    markInvalid(depthResInput, !!depthResMsg);
    if (depthResMsg) ok = false;

    // 前景分割分辨率（仅自定义模式需要校验）
    let maskResMsg = "";
    if (maskResMode.value === "custom") {
      const rawMask = maskResInput.value.trim();
      const maskValue = Number(rawMask);
      if (!rawMask || !Number.isInteger(maskValue)) {
        maskResMsg = t("err.notInteger");
      } else if (maskValue < c.mask_resolution_min || maskValue > c.mask_resolution_max) {
        maskResMsg = t("err.outOfRange", { min: c.mask_resolution_min, max: c.mask_resolution_max });
      }
    }
    setFieldError("maskRes", maskResMsg);
    markInvalid(maskResInput, !!maskResMsg);
    if (maskResMsg) ok = false;

    // 输出图层：一个都不选等于没请求，上游会当成全量，这里直接拦下
    const partsMsg = selectedParts().length === 0 ? t("err.partsEmpty") : "";
    setFieldError("parts", partsMsg);
    if (partsMsg) ok = false;

    // 模型选择
    const layerMsg = selectedModel("layerdiff") ? "" : t("err.modelNotSelected");
    const depthMsg = selectedModel("depth") ? "" : t("err.modelNotSelected");
    setFieldError("modelLayer", layerMsg);
    setFieldError("modelDepth", depthMsg);
    if (layerMsg || depthMsg) ok = false;

    state.valid = ok;
    refreshRunButton();
    return ok;
  }

  /* ────────────────────────────── 运行状态 ────────────────────────────── */

  function refreshRunButton() {
    const modelsChosen = !!selectedModel("layerdiff") && !!selectedModel("depth");
    el("runBtn").disabled = state.running || !state.file || !modelsChosen || !state.valid;
  }

  function updateRunStateLabel() {
    setRunState(state.statusKey || "status.ready");
  }

  function setRunState(key) {
    state.statusKey = key;
    el("runState").textContent = t(key);
  }

  function updateFacts(extra) {
    // 运行读数绑定“本次任务”的参数，直到换图或再次开始
    const job = state.activeParams;
    const useJob = !!job;
    el("factRes").textContent = useJob
      ? `${job.width} × ${job.height}`
      : `${widthInput.value || "—"} × ${heightInput.value || "—"}`;
    el("factSteps").textContent = useJob ? String(job.inference_steps) : stepsInput.value || "—";
    el("factSeed").textContent = useJob
      ? job.seed != null
        ? String(job.seed)
        : "—"
      : state.seedMode === "random"
        ? state.lastSeed != null
          ? String(state.lastSeed)
          : "—"
        : seedInput.value || "—";
    if (extra && extra.elapsed != null) {
      el("factElapsed").textContent = formatDuration(extra.elapsed);
    } else if (useJob && state.lastElapsed != null) {
      el("factElapsed").textContent = formatDuration(state.lastElapsed);
    } else if (!state.running) {
      el("factElapsed").textContent = "—";
    }
  }

  function setRunning(running) {
    state.running = running;
    const btn = el("runBtn");
    btn.classList.toggle("is-busy", running);
    btn.querySelector(".run-label").textContent = running ? t("action.running") : t("action.run");
    el("cancelBtn").hidden = !running;
    fileInput.disabled = running;
    el("fileRemove").disabled = running;
    drop.setAttribute("aria-disabled", running ? "true" : "false");
    setRunState(running ? "status.running" : "status.ready");
    if (running) setStage("running");
    refreshRunButton();

    if (running) {
      state.startedAt = Date.now();
      clearInterval(state.timer);
      state.timer = setInterval(() => {
        if (!state.running) return;
        const sec = (Date.now() - state.startedAt) / 1000;
        el("factElapsed").textContent = formatDuration(sec);
      }, 1000);
    } else {
      clearInterval(state.timer);
      state.timer = null;
    }
  }

  /* ────────────────────────────── 成像台 ────────────────────────────── */

  function setStage(mode) {
    const map = { empty: "stageEmpty", running: "stageRunning", failed: "stageFailed", results: "results" };
    Object.keys(map).forEach((key) => {
      el(map[key]).hidden = key !== mode;
    });
  }

  function resetResultArea() {
    state.activeParams = null;
    state.lastElapsed = null;
    state.lastOutputCount = null;
    state.lastFailDetail = null;
    setStage("empty");
    el("tiles").innerHTML = "";
    el("psdFiles").innerHTML = "";
    el("psdBar").hidden = true;
    el("metaFiles").innerHTML = "";
    el("metaBar").hidden = true;
    el("stageTools").hidden = true;
    state.lastResults = null;
    postOpen(false);
    hidePose();
    el("stageSub").textContent = t("stage.sub");
    el("factElapsed").textContent = "—";
    state.lastSeed = null;
    updateFacts();
  }

  const JOINT_LABELS = {
    zh: [
      "鼻子", "左眼", "右眼", "左耳", "右耳",
      "左肩", "右肩", "左肘", "右肘", "左腕", "右腕",
      "左髋", "右髋", "左膝", "右膝", "左踝", "右踝",
      "左脚大趾", "左脚小趾", "左脚跟", "右脚大趾", "右脚小趾", "右脚跟",
    ],
    en: [
      "nose", "left eye", "right eye", "left ear", "right ear",
      "left shoulder", "right shoulder", "left elbow", "right elbow", "left wrist", "right wrist",
      "left hip", "right hip", "left knee", "right knee", "left ankle", "right ankle",
      "left big toe", "left small toe", "left heel", "right big toe", "right small toe", "right heel",
    ],
  };

  function hidePose() {
    el("poseStrip").hidden = true;
    el("poseList").innerHTML = "";
  }

  function renderPose(jobId) {
    hidePose();
    if (!jobId) return;
    fetch(`/api/jobs/${jobId}/pose`)
      .then((response) => (response.ok ? response.json() : null))
      .then((data) => {
        if (!data || !data.ok || !Array.isArray(data.weakest) || !data.weakest.length) return;
        const labels = JOINT_LABELS[lang] || JOINT_LABELS.zh;
        const threshold = Number(data.threshold);
        const hasThreshold = Number.isFinite(threshold);
        const list = el("poseList");
        list.innerHTML = "";
        data.weakest.forEach((joint) => {
          const chip = document.createElement("span");
          chip.className = "pose-chip" + (hasThreshold && joint.score < threshold ? " low" : "");
          const name = document.createElement("span");
          name.className = "name";
          name.textContent = labels[joint.index] || joint.name;
          const score = document.createElement("span");
          score.className = "score";
          score.textContent = joint.score.toFixed(2);
          chip.appendChild(name);
          chip.appendChild(score);
          list.appendChild(chip);
        });
        if (hasThreshold) {
          const hint = document.createElement("span");
          hint.className = "pose-hint";
          hint.textContent = t("stage.poseHint", { v: threshold.toFixed(2) });
          list.appendChild(hint);
        }
        el("poseStrip").hidden = false;
      })
      .catch(() => {});
  }

  /* ── 成像台右上：后处理重跑 / 打开文件夹 ── */
  let postCloseTimer = null;
  let postPollTimer = null;

  function cancelPostClose() {
    if (postCloseTimer) {
      clearTimeout(postCloseTimer);
      postCloseTimer = null;
    }
  }

  function schedulePostClose() {
    cancelPostClose();
    postCloseTimer = setTimeout(() => postOpen(false), 3000);
  }

  function syncPostOutputs() {
    el("postConfOut").textContent = Number(el("postConf").value).toFixed(2);
    el("postOverlapOut").textContent = Number(el("postOverlap").value).toFixed(2);
  }

  function postOpen(open) {
    const pop = el("postPop");
    if (!pop) return;
    pop.hidden = !open;
    el("postBtn").setAttribute("aria-expanded", open ? "true" : "false");
    if (!open) return;
    cancelPostClose();
    const p = state.activeParams || {};
    el("postTblr").checked = p.tblr_split !== false;
    el("postRagdoll").checked = p.ragdoll_split === true;
    el("postPsd").checked = p.save_psd !== false;
    el("postCut").checked = p.save_cut_images !== false;
    if (p.ragdoll_confidence != null) el("postConf").value = p.ragdoll_confidence;
    if (p.ragdoll_overlap != null) el("postOverlap").value = p.ragdoll_overlap;
    syncPostOutputs();
    el("postStatus").textContent = "";
  }

  function revealFolder() {
    if (!state.jobId) return;
    fetch(`/api/jobs/${state.jobId}/reveal`, { method: "POST" }).catch(() => {});
  }

  function pollPostProcess(jobId) {
    if (postPollTimer) clearInterval(postPollTimer);
    postPollTimer = setInterval(() => {
      fetch(`/api/jobs/${jobId}/reassemble`)
        .then((r) => r.json())
        .then((data) => {
          const tail = (data.log_tail || []).slice(-1)[0];
          if (data.running) {
            if (tail) el("postStatus").textContent = String(tail).slice(0, 48);
            return;
          }
          clearInterval(postPollTimer);
          postPollTimer = null;
          el("postStart").disabled = false;
          el("postStatus").textContent = data.result === false ? t("post.failed") : t("post.done");
          fetch(`/api/status/${jobId}`)
            .then((r) => r.json())
            .then((payload) => {
              if (payload && payload.outputs) renderResults(payload);
            })
            .catch(() => {});
        })
        .catch(() => {});
    }, 1500);
  }

  function startPostProcess() {
    const jobId = state.jobId;
    if (!jobId || el("postStart").disabled) return;
    el("postStart").disabled = true;
    el("postStatus").textContent = t("post.running");
    const body = {
      tblr_split: el("postTblr").checked,
      ragdoll_split: el("postRagdoll").checked,
      save_psd: el("postPsd").checked,
      save_cut_images: el("postCut").checked,
      ragdoll_confidence: Number(el("postConf").value),
      ragdoll_overlap: Number(el("postOverlap").value),
    };
    fetch(`/api/jobs/${jobId}/reassemble`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    })
      .then((r) => r.json().then((payload) => ({ ok: r.ok, payload })).catch(() => ({ ok: false, payload: {} })))
      .then(({ ok, payload }) => {
        if (!ok || !payload.ok) {
          el("postStart").disabled = false;
          el("postStatus").textContent =
            payload.error === "busy" || payload.error === "already_running" ? t("post.busy") : t("post.failed");
          return;
        }
        pollPostProcess(jobId);
      })
      .catch(() => {
        el("postStart").disabled = false;
        el("postStatus").textContent = t("err.network");
      });
  }

  el("postBtn").addEventListener("click", () => postOpen(el("postPop").hidden));
  el("stageTools").addEventListener("mouseenter", cancelPostClose);
  el("stageTools").addEventListener("mouseleave", schedulePostClose);
  el("postConf").addEventListener("input", syncPostOutputs);
  el("postOverlap").addEventListener("input", syncPostOutputs);
  el("postStart").addEventListener("click", startPostProcess);

  function renderResults(payload, animate) {
    if (payload.job_id) state.jobId = payload.job_id;
    state.lastResults = payload;
    el("stageTools").hidden = false;
    const outputs = payload.outputs || [];
    const images = outputs.filter((o) => o.kind === "image");
    const psds = outputs.filter((o) => o.kind === "psd_main" || o.kind === "psd_depth");
    const metas = outputs.filter((o) => o.kind === "json");

    el("tiles").innerHTML = "";
    el("psdFiles").innerHTML = "";
    el("metaFiles").innerHTML = "";

    if (outputs.length) {
      const openBtn = document.createElement("button");
      openBtn.type = "button";
      openBtn.className = "psd-file";
      openBtn.innerHTML =
        `<span class="name">${escapeHtml(t("stage.openFolder"))}</span>` +
        `<span class="arrow">↗</span>`;
      openBtn.title = psds.map((o) => o.name).join(" · ");
      openBtn.addEventListener("click", () => revealFolder());
      el("psdFiles").appendChild(openBtn);
      el("psdBar").hidden = false;
    } else {
      el("psdBar").hidden = true;
    }

    if (metas.length) {
      metas.forEach((meta) => {
        const link = document.createElement("a");
        link.className = "psd-file";
        link.href = meta.download_url || meta.url;
        link.setAttribute("download", meta.name);
        link.innerHTML =
          `<span class="name">${escapeHtml(meta.name)}</span>` +
          `<span class="size">${formatBytes(meta.size)}</span>` +
          `<span class="arrow">↓</span>`;
        el("metaFiles").appendChild(link);
      });
      el("metaBar").hidden = false;
    } else {
      el("metaBar").hidden = true;
    }

    renderPose(payload.job_id || state.jobId);

    const tpl = el("tplTile");
    images.forEach((img) => {
      const node = tpl.content.firstElementChild.cloneNode(true);
      const link = node.querySelector(".tile-link");
      link.href = img.url;
      const image = node.querySelector(".tile-img");
      image.src = img.url;
      image.alt = img.name;
      node.querySelector(".tile-name").textContent = img.name;
      node.querySelector(".tile-name").title = img.name;
      node.querySelector(".tile-size").textContent = formatBytes(img.size);
      const dl = node.querySelector(".tile-dl");
      dl.href = img.download_url || img.url;
      dl.setAttribute("download", img.name);
      el("tiles").appendChild(node);
    });

    updateFacts();
    state.lastOutputCount = images.length + psds.length;
    const res = payload.params ? `${payload.params.width} × ${payload.params.height}` : "";
    el("stageSub").textContent = t("stage.summary", { n: images.length + psds.length, res }) + (res ? "" : "");
    setStage("results");

    if (animate !== false && !reduceMotion && window.gsap && images.length) {
      gsap.from("#tiles .tile", {
        autoAlpha: 0,
        y: 8,
        duration: 0.3,
        stagger: 0.04,
        ease: "power2.out",
        clearProps: "all",
      });
    }
  }

  /* ────────────────────────────── 错误映射 ────────────────────────────── */

  function fieldCopy(field, error) {
    const map = { width: widthInput, height: heightInput, inference_steps: stepsInput, seed: seedInput };
    if (error.code === "not_integer") return t("err.notInteger");
    if (error.code === "out_of_range") return t("err.outOfRange", { min: error.min, max: error.max });
    if (error.code === "not_multiple") return t("err.notMultiple", { step: error.step });
    if (error.code === "must_equal_width") return t("err.mustEqual");
    if (error.code === "patch_missing") return t("err.patchMissing");
    if (error.code === "not_number") return t("err.notNumber");
    if (map[field]) return t("err.invalidParams");
    return t("err.invalidParams");
  }

  function applyBackendFieldErrors(details) {
    if (!Array.isArray(details)) return;
    details.forEach((error) => {
      const field = error.field;
      let bucket = null;
      if (field === "width" || field === "height") bucket = "res";
      else if (field === "inference_steps") bucket = "steps";
      else if (field === "seed") bucket = "seed";
      else if (field === "layerdiff_model") {
        const map = { not_selected: "err.modelNotSelected", outside_models: "err.modelOutside", missing: "err.modelMissing", not_gguf: "err.modelNotGguf" };
        setFieldError("modelLayer", t(map[error.code] || "err.modelNotSelected"));
        return;
      } else if (field === "depth_model") {
        const map = { not_selected: "err.modelNotSelected", outside_models: "err.modelOutside", missing: "err.modelMissing", not_gguf: "err.modelNotGguf" };
        setFieldError("modelDepth", t(map[error.code] || "err.modelNotSelected"));
        return;
      }
      if (bucket) {
        setFieldError(bucket, fieldCopy(field, error));
        if (bucket === "res") {
          markInvalid(widthInput, true);
          markInvalid(heightInput, true);
        }
        if (bucket === "steps") markInvalid(stepsInput, true);
        if (bucket === "seed") markInvalid(seedInput, true);
      }
    });
  }

  function handleRunFailure(payload, status) {
    const code = payload.error;
    if (code === "models_missing") {
      const items = (payload.missing || []).map((k) => t("err.modelItem." + k)).join(lang === "zh" ? "、" : ", ");
      showNotice("error", t("err.modelsMissing", { items }));
    } else if (code === "invalid_params") {
      applyBackendFieldErrors(payload.details);
      showNotice("error", t("err.invalidParams"));
    } else if (code === "unsupported_format") {
      const detail = (payload.details && payload.details[0]) || {};
      showNotice("error", detail.code === "decode" ? t("err.decode") : t("err.format"));
    } else if (code === "file_too_large") {
      showNotice("error", t("err.tooLarge", { mb: payload.max_upload_mb }));
    } else if (code === "busy") {
      showNotice("warn", t("err.busy"));
    } else if (code === "no_file") {
      showNotice("error", t("err.noFile"));
    } else if (code === "storage_failed") {
      showNotice("error", t("err.storage"));
    } else if (status === 0) {
      showNotice("error", t("err.network"));
    } else {
      showNotice("error", t("err.generic") + (code ? " (" + code + ")" : ""));
    }
  }

  /* ────────────────────────────── 提交 ────────────────────────────── */

  function collectParams() {
    return {
      width: Number(widthInput.value),
      height: Number(heightInput.value),
      inference_steps: Number(stepsInput.value),
      inference_steps_depth: Number(el("depthSteps").value),
      resolution_depth: depthResolutionParam(),
      mask_resolution: maskResolutionParam(),
      attention_backend: state.attnBackend,
      seed_mode: state.seedMode,
      seed: Number(seedInput.value),
      tblr_split: el("tblrSplit").checked,
      use_lama: el("useLama").checked,
      save_psd: el("savePsd").checked,
      ragdoll_split: el("ragdoll").checked,
      ragdoll_confidence: Number(el("ragdollConf").value),
      ragdoll_overlap: Number(el("ragdollOverlap").value),
      group_offload: el("groupOffload").checked,
      save_cut_images: el("cutImages").checked,
      layer_fix: el("layerFix").checked,
      parts: (() => {
        const picked = selectedParts();
        return picked.length === PART_TOTAL ? null : picked;
      })(),
      quantized: el("quantized").checked,
      layerdiff_model: selectedModel("layerdiff"),
      depth_model: selectedModel("depth"),
      keep_model: el("keepModel").checked,
    };
  }

  function submit() {
    if (state.running) return;
    if (!state.file) {
      showNotice("error", t("err.noFile"));
      return;
    }
    if (!validateInline()) return;

    hideNotice();
    resetResultArea();
    setRunning(true);
    el("factElapsed").textContent = "0s";
    state.jobId = null;
    state.lastSeq = 0;
    state.finalized = null;

    const form = new FormData();
    form.append("image", state.file.file);
    form.append("params", JSON.stringify(collectParams()));

    fetch("/api/run", { method: "POST", body: form })
      .then((response) =>
        response
          .json()
          .catch(() => ({}))
          .then((payload) => ({ ok: response.ok, status: response.status, payload }))
      )
      .then(({ ok, status, payload }) => {
        if (!ok || !payload.ok) {
          setRunning(false);
          setStage("empty");
          handleRunFailure(payload, status);
          return;
        }

        state.jobId = payload.job_id;
        state.activeParams = { ...payload.params, seed: payload.seed_used };
        if (state.seedMode === "random") {
          state.lastSeed = payload.seed_used;
          seedInput.value = String(payload.seed_used);
        }
        el("factSeed").textContent = String(payload.seed_used);
        el("factRes").textContent = `${payload.params.width} × ${payload.params.height}`;
        el("factSteps").textContent = String(payload.params.inference_steps);
        appendLog({
          ts: Date.now(),
          level: "sys",
          msg: t("log.seedUsed", { seed: payload.seed_used }) + " · " + payload.job_id,
        });
        connectStream(payload.job_id);
        startPolling(payload.job_id);
      })
      .catch(() => {
        setRunning(false);
        setStage("empty");
        showNotice("error", t("err.network"));
      });
  }

  el("runBtn").addEventListener("click", submit);

  /* ────────────────────────────── SSE ────────────────────────────── */

  function attachRunningJob() {
    // 刷新页面后如果后端还有任务在跑，重新接上日志与状态
    fetch("/api/current")
      .then((r) => r.json())
      .then((payload) => {
        const job = payload && payload.job;
        if (!job) {
          restoreLastJob();
          return;
        }
        if (state.running) return;
        resetResultArea();
        state.jobId = job.job_id;
        state.activeParams = { ...job.params, seed: job.seed_used };
        state.lastSeq = 0;
        state.finalized = null;
        setRunning(true);
        setStage("running");
        (job.log_tail || []).forEach(appendLog);
        updateFacts();
        if (job.elapsed != null) el("factElapsed").textContent = formatDuration(job.elapsed);
        connectStream(job.job_id);
        startPolling(job.job_id);
      })
      .catch(() => {
        /* 没有运行中的任务或网络异常时保持空闲界面 */
      });
  }

  function restoreLastJob() {
    // 没有在跑的任务时，回看 24 小时内最近一次成功结果（页面刷新后不至于空着）
    fetch("/api/last")
      .then((r) => r.json())
      .then((payload) => {
        const job = payload && payload.job;
        if (!job || state.running || state.file) return;
        const params = job.params || {};
        // 表单同步成这次任务的参数，避免"显示 1024、结果其实是 768"的错位
        if (params.width) widthInput.value = params.width;
        if (params.height) heightInput.value = params.height;
        if (params.inference_steps) {
          stepsInput.value = params.inference_steps;
          if (stepsRange) stepsRange.value = params.inference_steps;
        }
        if (job.seed_used != null) seedInput.value = job.seed_used;
        syncResolution(widthInput);
        updateFacts();
        validateInline();

        state.activeParams = { ...params, seed: job.seed_used };
        state.lastOutputCount = (job.outputs || []).filter((o) => o.kind === "image").length;
        finalize(job, job.job_id);
        const sub = el("stageSub");
        sub.textContent = `${t("stage.lastRun")} · ${sub.textContent}`;
      })
      .catch(() => {});
  }

  function startPolling(jobId) {
    stopPolling();
    state.pollTimer = setInterval(() => {
      if (state.jobId !== jobId) return;
      fetch("/api/status/" + jobId)
        .then((r) => (r.ok ? r.json() : Promise.reject(new Error("status"))))
        .then((payload) => {
          (payload.log_tail || []).forEach(appendLog);
          if (payload.elapsed != null && state.running) {
            el("factElapsed").textContent = formatDuration(payload.elapsed);
          }
          if (["completed", "completed_with_warning", "failed", "canceled"].includes(payload.status)) {
            finalize(payload, jobId);
          }
        })
        .catch(() => {
          /* 单次轮询失败不改变界面状态，下一轮再试 */
        });
    }, 1500);
  }

  function stopPolling() {
    if (state.pollTimer) {
      clearInterval(state.pollTimer);
      state.pollTimer = null;
    }
  }

  function connectStream(jobId) {
    if (state.es) {
      state.es.close();
      state.es = null;
    }

    const source = new EventSource("/api/stream/" + jobId);
    state.es = source;

    source.addEventListener("log", (event) => {
      try {
        appendLog(JSON.parse(event.data));
      } catch (_) {
        /* 忽略单行解析失败，不影响后续日志 */
      }
    });

    source.addEventListener("done", (event) => {
      source.close();
      state.es = null;
      let payload = {};
      try {
        payload = JSON.parse(event.data);
      } catch (_) {
        /* 空 payload 时按失败处理 */
      }
      finalize(payload, jobId);
    });

    source.onerror = () => {
      // 部分环境（杀软/代理）会缓冲 text/event-stream：这里不重连，交给轮询兜底
      source.close();
      state.es = null;
    };
  }

  function finalize(payload, jobId) {
    if (state.finalized === jobId) return;
    state.finalized = jobId;
    stopPolling();
    setRunning(false);

    const status = payload.status || "failed";
    const elapsed = payload.elapsed || 0;
    state.lastElapsed = elapsed;
    el("factElapsed").textContent = formatDuration(elapsed);

    if (payload.seed_used != null) {
      state.lastSeed = payload.seed_used;
      if (state.seedMode === "random") seedInput.value = String(payload.seed_used);
      updateFacts({ elapsed });
    }

    if (status === "failed") {
      setStage("failed");
      state.lastFailDetail = payload.error_detail || "";
      el("failedHint").textContent = t("err.inference") + (payload.error_detail ? ` (${payload.error_detail})` : "");
      setRunState("status.fail");
      showNotice("error", t("err.inference"));
      appendLog({ ts: Date.now(), level: "error", msg: t("err.inference") });
      return;
    }

    if (status === "canceled") {
      resetResultArea();
      setStage("empty");
      setRunState("status.canceled");
      showNotice("info", t("err.canceled"));
      return;
    }

    if (status === "completed_with_warning") {
      renderResults(payload);
      setRunState("status.warn");
      const message = payload.error === "no_outputs" ? t("err.noOutputs") : t("err.psd");
      showNotice("warn", message);
      appendLog({ ts: Date.now(), level: "warn", msg: message });
      return;
    }

    renderResults(payload);
    setRunState("status.done");
    appendLog({ ts: Date.now(), level: "ok", msg: t("log.elapsed", { sec: Math.round(elapsed) }) });
    hideNotice();
  }

  /* ────────────────────────────── 取消 ────────────────────────────── */

  el("cancelBtn").addEventListener("click", () => {
    if (!state.jobId || !state.running) return;
    el("cancelBtn").disabled = true;
    setRunState("status.canceling");
    appendLog({ ts: Date.now(), level: "warn", msg: t("log.cancelRequest") });
    fetch("/api/jobs/" + state.jobId + "/cancel", { method: "POST" })
      .catch(() => {})
      .then(() => {
        el("cancelBtn").disabled = false;
      });
  });

  /* ────────────────────────────── 语言 ────────────────────────────── */

  el("langBtn").addEventListener("click", () => {
    lang = lang === "zh" ? "en" : "zh";
    localStorage.setItem(LS_LANG, lang);
    applyI18n();
    if (el("depthResMode")) syncDepthResUi();
    if (el("maskResMode")) syncMaskResUi();
    if (el("attnBackend")) syncAttnAvailability();
    if (state.lastResults && !el("results").hidden) renderResults(state.lastResults, false);
    if (state.modelInfo) renderModelSelects();
    if (state.models) renderModels(state.models);
    if (state.file) renderFileSpec();
    // 日志级别标签随语言重绘
    document.querySelectorAll(".log-line").forEach((line) => {
      const level = line.dataset.level;
      const tag = line.querySelector(".log-level");
      if (tag && LEVEL_LABEL[level]) tag.textContent = t(LEVEL_LABEL[level]);
    });
  });

  /* ────────────────────────────── 可拖拽分隔条 ────────────────────────────── */

  function initSplitters() {
    const bench = document.querySelector(".bench");
    const stage = document.querySelector(".stage");
    const railSplit = el("splitterV");
    const logSplit = el("splitterH");
    if (!bench || !stage || !railSplit || !logSplit) return;

    const MIN_RAIL = 320;
    const MAX_RAIL = 640;
    const MIN_LOG = 120;
    const MIN_STAGE = 220;
    const clamp = (n, lo, hi) => Math.max(lo, Math.min(hi, n));

    const savedRail = Number(localStorage.getItem("stg.railW"));
    if (savedRail) bench.style.setProperty("--rail-w", clamp(savedRail, MIN_RAIL, MAX_RAIL) + "px");
    const savedLog = Number(localStorage.getItem("stg.consoleH"));
    if (savedLog) stage.style.setProperty("--console-h", savedLog + "px");

    let drag = null;

    function start(which, event) {
      if (window.innerWidth <= 1120) return;
      drag = which;
      event.currentTarget.classList.add("is-dragging");
      try {
        event.currentTarget.setPointerCapture(event.pointerId);
      } catch (_) {}
      document.body.style.cursor = which === "rail" ? "col-resize" : "row-resize";
      document.body.style.userSelect = "none";
    }

    function move(event) {
      if (!drag) return;
      if (drag === "rail") {
        const rect = bench.getBoundingClientRect();
        bench.style.setProperty("--rail-w", clamp(event.clientX - rect.left, MIN_RAIL, MAX_RAIL) + "px");
      } else {
        const rect = stage.getBoundingClientRect();
        const max = Math.max(MIN_LOG, rect.height - MIN_STAGE);
        stage.style.setProperty("--console-h", clamp(rect.bottom - event.clientY, MIN_LOG, max) + "px");
      }
    }

    function end(event) {
      if (!drag) return;
      event.currentTarget.classList.remove("is-dragging");
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
      if (drag === "rail") {
        localStorage.setItem("stg.railW", parseInt(bench.style.getPropertyValue("--rail-w"), 10) || 396);
      } else {
        localStorage.setItem("stg.consoleH", parseInt(stage.style.getPropertyValue("--console-h"), 10) || 236);
      }
      drag = null;
    }

    railSplit.addEventListener("pointerdown", (e) => start("rail", e));
    logSplit.addEventListener("pointerdown", (e) => start("log", e));
    [railSplit, logSplit].forEach((node) => {
      node.addEventListener("pointermove", move);
      node.addEventListener("pointerup", end);
      node.addEventListener("pointercancel", end);
    });

    railSplit.addEventListener("dblclick", () => {
      bench.style.removeProperty("--rail-w");
      localStorage.removeItem("stg.railW");
    });
    logSplit.addEventListener("dblclick", () => {
      stage.style.removeProperty("--console-h");
      localStorage.removeItem("stg.consoleH");
    });

    function keyStep(node, isRail) {
      return (event) => {
        if (!["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"].includes(event.key)) return;
        event.preventDefault();
        const step = event.shiftKey ? 48 : 16;
        if (isRail) {
          const current = parseInt(getComputedStyle(bench).getPropertyValue("--rail-w"), 10) || 396;
          const next = clamp(current + (event.key === "ArrowRight" ? step : -step), MIN_RAIL, MAX_RAIL);
          bench.style.setProperty("--rail-w", next + "px");
          localStorage.setItem("stg.railW", next);
        } else {
          const rect = stage.getBoundingClientRect();
          const current = parseInt(getComputedStyle(stage).getPropertyValue("--console-h"), 10) || Math.round(rect.height * 0.27);
          const max = Math.max(MIN_LOG, rect.height - MIN_STAGE);
          const next = clamp(current + (event.key === "ArrowUp" ? step : -step), MIN_LOG, max);
          stage.style.setProperty("--console-h", next + "px");
          localStorage.setItem("stg.consoleH", next);
        }
      };
    }
    railSplit.addEventListener("keydown", keyStep(railSplit, true));
    logSplit.addEventListener("keydown", keyStep(logSplit, false));
  }

  /* ────────────────────────────── 初始化 ────────────────────────────── */

  function introMotion() {
    if (reduceMotion || !window.gsap) return;
    // 一次编排好的入场：顶栏 → 轨道工序 → 成像台 → 控制台
    const tl = gsap.timeline({ defaults: { ease: "power2.out" } });
    tl.from(".masthead", { autoAlpha: 0, y: -8, duration: 0.34 })
      .from(".rail .step", { autoAlpha: 0, y: 10, duration: 0.4, stagger: 0.07 }, 0.08)
      .from(".stage", { autoAlpha: 0, y: 8, duration: 0.42 }, 0.16)
      .from(".console", { autoAlpha: 0, duration: 0.34 }, 0.3);
  }

  function boot() {
    applyI18n();
    buildParts();
    setSeedMode("fixed");
    clearLog();

    fetch("/api/config")
      .then((r) => r.json())
      .then((config) => {
        state.config = config;
        applyDefaults();
        applyI18n();
        validateInline();
      })
      .catch(() => {
        applyDefaults();
        validateInline();
        showNotice("error", t("err.network"));
      })
      .finally(() => {
        loadModelFiles().finally(() => {
          loadModels().finally(attachRunningJob);
        });
      });

    initSplitters();
    introMotion();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
