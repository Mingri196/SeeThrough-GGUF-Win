/* PSD 预览页：把图层按 offset 叠到 canvas 上，支持显隐、缩放、平移。
   数据来自 /api/psd/<job_id>/layers（服务端用 psd_tools 拆层）。 */
(() => {
  "use strict";

  const STR = {
    zh: {
      fit: "适应窗口",
      actual: "100%",
      bgChecker: "棋盘底",
      bgDark: "深色底",
      bgLight: "白色底",
      loading: "正在加载 PSD…",
      loadingLayer: "正在加载图层 {n}/{t}",
      hint: "滚轮缩放 · 按住中键拖动 · 双击适应窗口",
      layerWord: "层",
      allOn: "全显",
      allOff: "全隐",
      loadFail: "加载失败：{msg}",
      noLayer: "这个 PSD 里没有可显示的像素层",
    },
    en: {
      fit: "Fit",
      actual: "100%",
      bgChecker: "Checker",
      bgDark: "Dark",
      bgLight: "White",
      loading: "Loading PSD…",
      loadingLayer: "Loading layers {n}/{t}",
      hint: "Wheel to zoom · hold middle button to pan · double-click to fit",
      layerWord: "layers",
      allOn: "All on",
      allOff: "All off",
      loadFail: "Failed to load: {msg}",
      noLayer: "This PSD has no pixel layers to show",
    },
  };
  const L = STR[localStorage.getItem("stg.lang") === "en" ? "en" : "zh"];
  const MIN_SCALE = 0.02;
  const MAX_SCALE = 16;

  const el = (id) => document.getElementById(id);
  const viewport = el("viewport");
  const canvas = el("stage");
  const ctx = canvas.getContext("2d");
  const list = el("layerList");

  const state = {
    W: 0,
    H: 0,
    scale: 1,
    ox: 0,
    oy: 0,
    layers: [],
    bg: "checker",
    drag: null,
    ready: false,
  };

  const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

  /* ── 绘制 ──────────────────────────────────────────── */

  function draw() {
    if (!state.ready) {
      // 还没有画布尺寸时也要清理一次，避免上一帧残留
      const dpr0 = window.devicePixelRatio || 1;
      ctx.setTransform(dpr0, 0, 0, dpr0, 0, 0);
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      return;
    }
    const dpr = window.devicePixelRatio || 1;
    const vw = viewport.clientWidth;
    const vh = viewport.clientHeight;
    const cw = Math.round(vw * dpr);
    const ch = Math.round(vh * dpr);
    if (canvas.width !== cw || canvas.height !== ch) {
      canvas.width = cw;
      canvas.height = ch;
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, vw, vh);
    // 缩小时平滑（看得清轮廓），放大时关掉（看见像素边界，和 PS 一致）
    ctx.imageSmoothingEnabled = state.scale < 2;
    ctx.save();
    ctx.translate(state.ox, state.oy);
    ctx.scale(state.scale, state.scale);
    for (const layer of state.layers) {
      if (!layer.visible) continue;
      const img = layer.img;
      if (!img || !img.complete || !img.naturalWidth) continue;
      ctx.globalAlpha = (layer.opacity == null ? 255 : layer.opacity) / 255;
      ctx.drawImage(img, layer.left, layer.top);
    }
    ctx.restore();
    ctx.globalAlpha = 1;
    el("zoomLabel").textContent = Math.round(state.scale * 100) + "%";
  }

  let rafId = 0;
  function scheduleDraw() {
    if (rafId) return;
    rafId = requestAnimationFrame(() => {
      rafId = 0;
      draw();
    });
  }

  function fit() {
    if (!state.W || !state.H) return;
    const vw = viewport.clientWidth;
    const vh = viewport.clientHeight;
    const pad = 28;
    const s = Math.min((vw - pad * 2) / state.W, (vh - pad * 2) / state.H);
    state.scale = clamp(s, MIN_SCALE, MAX_SCALE);
    state.ox = (vw - state.W * state.scale) / 2;
    state.oy = (vh - state.H * state.scale) / 2;
    draw();
  }

  function zoomAt(px, py, factor) {
    const next = clamp(state.scale * factor, MIN_SCALE, MAX_SCALE);
    const k = next / state.scale;
    state.ox = px - (px - state.ox) * k;
    state.oy = py - (py - state.oy) * k;
    state.scale = next;
    draw();
  }

  /* ── 图层面板 ──────────────────────────────────────── */

  function eyeSvg(on) {
    return (
      '<svg class="eye" viewBox="0 0 16 16" fill="none" stroke="currentColor" ' +
      'stroke-width="1.2" stroke-linecap="round">' +
      '<path d="M1.4 8S3.9 3.8 8 3.8 14.6 8 14.6 8 12.1 12.2 8 12.2 1.4 8 1.4 8Z"/>' +
      '<circle cx="8" cy="8" r="1.8"/>' +
      (on ? "" : '<line x1="2.6" y1="13.4" x2="13.4" y2="2.6"/>') +
      "</svg>"
    );
  }

  function renderList() {
    list.innerHTML = "";
    // PS 图层面板习惯：最上层显示在最前，所以这里倒序遍历；绘制顺序不变
    for (let i = state.layers.length - 1; i >= 0; i--) {
      const layer = state.layers[i];
      const row = document.createElement("li");
      row.className = "layer-row" + (layer.visible ? "" : " is-off");
      row.innerHTML =
        eyeSvg(layer.visible) +
        '<img class="thumb" alt="" loading="lazy">' +
        '<span class="label"><span class="name"></span>' +
        `<span class="dim">${layer.width}×${layer.height}</span></span>`;
      row.querySelector(".thumb").src = layer.url;
      row.querySelector(".name").textContent = layer.name;
      row.addEventListener("click", () => {
        layer.visible = !layer.visible;
        row.classList.toggle("is-off", !layer.visible);
        row.querySelector(".eye").outerHTML = eyeSvg(layer.visible);
        scheduleDraw();
      });
      list.appendChild(row);
    }
    el("layerCount").textContent = state.layers.length + " " + L.layerWord;
  }

  /* ── 交互 ──────────────────────────────────────────── */

  viewport.addEventListener(
    "wheel",
    (event) => {
      if (!state.ready) return;
      event.preventDefault();
      const rect = viewport.getBoundingClientRect();
      // 指数缩放：滚动一格约 12%，快慢都与当前倍率成比例
      const factor = Math.exp(-event.deltaY * 0.0012);
      zoomAt(event.clientX - rect.left, event.clientY - rect.top, factor);
    },
    { passive: false }
  );

  // 中键拖动平移：preventDefault 是为了压掉 Chrome 的中键自动滚动模式
  viewport.addEventListener("mousedown", (event) => {
    if (event.button !== 1) return;
    event.preventDefault();
    state.drag = { x: event.clientX, y: event.clientY, ox: state.ox, oy: state.oy };
    viewport.classList.add("is-panning");
  });

  window.addEventListener("mousemove", (event) => {
    if (!state.drag) return;
    state.ox = state.drag.ox + (event.clientX - state.drag.x);
    state.oy = state.drag.oy + (event.clientY - state.drag.y);
    scheduleDraw();
  });

  window.addEventListener("mouseup", (event) => {
    if (event.button !== 1 || !state.drag) return;
    state.drag = null;
    viewport.classList.remove("is-panning");
  });

  viewport.addEventListener("auxclick", (event) => {
    if (event.button === 1) event.preventDefault();
  });
  viewport.addEventListener("dblclick", () => {
    if (state.ready) fit();
  });
  viewport.addEventListener("contextmenu", (event) => event.preventDefault());

  window.addEventListener("keydown", (event) => {
    if (!state.ready) return;
    if (event.key === "0") fit();
    if (event.key === "1") {
      state.scale = 1;
      state.ox = (viewport.clientWidth - state.W) / 2;
      state.oy = (viewport.clientHeight - state.H) / 2;
      draw();
    }
  });

  window.addEventListener("resize", () => {
    if (state.ready) scheduleDraw();
  });

  el("btnFit").addEventListener("click", fit);
  el("btnActual").addEventListener("click", () => {
    if (!state.ready) return;
    state.scale = 1;
    state.ox = (viewport.clientWidth - state.W) / 2;
    state.oy = (viewport.clientHeight - state.H) / 2;
    draw();
  });
  el("btnBg").addEventListener("click", () => {
    const order = ["checker", "dark", "light"];
    state.bg = order[(order.indexOf(state.bg) + 1) % order.length];
    viewport.dataset.bg = state.bg;
    const btn = el("btnBg");
    btn.textContent = L["bg" + state.bg[0].toUpperCase() + state.bg.slice(1)];
    btn.setAttribute("aria-pressed", state.bg === "dark" ? "true" : "false");
  });
  el("btnAllOn").addEventListener("click", () => toggleAll(true));
  el("btnAllOff").addEventListener("click", () => toggleAll(false));

  function toggleAll(on) {
    state.layers.forEach((layer) => {
      layer.visible = on;
    });
    list.querySelectorAll(".layer-row").forEach((row) => {
      row.classList.toggle("is-off", !on);
      row.querySelector(".eye").outerHTML = eyeSvg(on);
    });
    scheduleDraw();
  }

  /* ── 加载 ──────────────────────────────────────────── */

  function setOverlay(text, pct) {
    const overlay = el("overlay");
    if (text == null) {
      overlay.hidden = true;
      return;
    }
    overlay.hidden = false;
    el("overlayText").textContent = text;
    const bar = el("overlayBar");
    if (pct == null) {
      bar.hidden = true;
    } else {
      bar.hidden = false;
      el("overlayFill").style.width = Math.round(pct * 100) + "%";
    }
  }

  function showError(message) {
    const overlay = el("overlay");
    overlay.hidden = false;
    el("overlayBar").hidden = true;
    const p = el("overlayText");
    p.className = "err";
    p.textContent = L.loadFail.replace("{msg}", message);
  }

  async function boot() {
    el("hint").textContent = L.hint;
    el("btnFit").textContent = L.fit;
    el("btnActual").textContent = L.actual;
    el("btnBg").textContent = L.bgChecker;
    el("btnAllOn").textContent = L.allOn;
    el("btnAllOff").textContent = L.allOff;

    const parts = location.pathname.split("/").filter(Boolean);
    const jobId = parts[parts.length - 1] || "";
    const file = new URLSearchParams(location.search).get("file") || "";

    setOverlay(L.loading, null);
    let data;
    try {
      const res = await fetch(
        `/api/psd/${encodeURIComponent(jobId)}/layers?file=${encodeURIComponent(file)}`
      );
      data = await res.json().catch(() => ({}));
      if (!res.ok || !data.ok) throw new Error(data.error || "HTTP " + res.status);
    } catch (err) {
      showError(err.message || String(err));
      return;
    }

    state.W = data.canvas[0];
    state.H = data.canvas[1];
    el("fileName").textContent = data.file || file;
    el("fileName").title = data.file || file;
    el("fileFacts").textContent = `${state.W} × ${state.H}`;

    if (!data.layers.length) {
      showError(L.noLayer);
      return;
    }

    state.layers = data.layers.map((item) => ({
      name: item.name,
      left: item.left,
      top: item.top,
      width: item.width,
      height: item.height,
      opacity: item.opacity,
      visible: item.visible !== false,
      url: item.url,
      img: null,
    }));

    state.ready = true;
    fit();
    renderList();

    const total = state.layers.length;
    let done = 0;
    await Promise.all(
      state.layers.map(
        (layer) =>
          new Promise((resolve) => {
            const img = new Image();
            img.onload = () => {
              layer.img = img;
              done += 1;
              setOverlay(L.loadingLayer.replace("{n}", done).replace("{t}", total), done / total);
              scheduleDraw();
              resolve();
            };
            img.onerror = () => {
              done += 1;
              resolve();
            };
            img.src = layer.url;
          })
      )
    );
    setOverlay(null);
    draw();
  }

  boot();
})();
