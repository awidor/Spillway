// serve/web/app.js - Spillway's dashboard: /metrics once a second, drawn. No framework.
"use strict";

const $ = (id) => document.getElementById(id);
const fmt = (n, d = 0) => (n == null || !Number.isFinite(n) ? "–" : n.toLocaleString(undefined, {maximumFractionDigits: d, minimumFractionDigits: d}));
const kfmt = (n) => (n == null ? "–" : n >= 1000 ? `${fmt(n / 1000, n >= 10000 ? 0 : 1)}K` : fmt(n));
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]));
const GiB = 1073741824;
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();

// the API key, when the service has one: open the page as http://host:port/#key=... (a fragment is never sent)
let key = "";
try { key = localStorage.getItem("spillway.key") || ""; } catch (e) { /* private mode */ }
const hk = new URLSearchParams(location.hash.slice(1)).get("key");
if (hk) {
  key = hk;
  try { localStorage.setItem("spillway.key", hk); } catch (e) { /* private mode */ }
  history.replaceState(null, "", location.pathname);
}

// ------------------------------------------------------------------ the heatmap: layers x experts
let heat = null;   // {layers, experts, cells: [{e, r, h}] per layer, in expert order}
function hexColor(c) {
  const x = document.createElement("canvas").getContext("2d");
  x.fillStyle = c;
  const v = x.fillStyle;   // "#rrggbb"
  return [1, 3, 5].map((i) => parseInt(v.slice(i, i + 2), 16));
}
function drawHeat(ex) {
  const cv = $("heat"), L = ex.layers, E = ex.experts;
  if (!heat || heat.time !== ex.time) {
    const cells = [];
    for (let l = 0; l < L; l++) {
      const row = [];
      for (let e = 0; e < E; e++) {
        const i = l * E + e;
        row.push({e, r: ex.resident[i] === "1", h: parseInt(ex.heat[i], 16) / 15});
      }
      cells.push(row);
    }
    heat = {layers: L, experts: E, cells, time: ex.time};
  }
  const dpr = devicePixelRatio || 1, w = cv.clientWidth, h = cv.clientHeight;
  cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr);
  const g = cv.getContext("2d"), cw = cv.width / E, ch = cv.height / L;
  const paper = hexColor(css("--paper")), vram = hexColor(css("--vram")), cpu = hexColor(css("--cpu"));
  const mix = (a, b, t) => `rgb(${a.map((x, i) => Math.round(x + (b[i] - x) * t)).join(",")})`;
  for (let l = 0; l < L; l++)
    heat.cells[l].forEach((c, j) => {
      g.fillStyle = c.r ? mix(paper, vram, 0.34 + 0.66 * c.h) : mix(paper, cpu, 0.1 + 0.9 * c.h);
      g.fillRect(Math.floor(j * cw), Math.floor(l * ch), Math.ceil(cw), Math.ceil(ch));
    });
  const n = ex.resident.split("").filter((c) => c === "1").length;
  $("heat-count").innerHTML = `${fmt(n)}<small>of ${fmt(L * E)} in VRAM</small>`;
}
$("heat").addEventListener("mousemove", (ev) => {
  if (!heat) return;
  const cv = $("heat"), r = cv.getBoundingClientRect();
  const j = Math.floor(((ev.clientX - r.left) / r.width) * heat.experts), l = Math.floor(((ev.clientY - r.top) / r.height) * heat.layers);
  const c = heat.cells[l] && heat.cells[l][j], tip = $("tip");
  if (!c) { tip.hidden = true; return; }
  tip.hidden = false;
  tip.style.left = `${ev.clientX - r.left + 52 * (innerWidth > 560) + 34 * (innerWidth <= 560)}px`;
  tip.style.top = `${ev.clientY - r.top}px`;
  tip.textContent = `layer ${l + 1}, expert ${c.e}: ${c.r ? "in VRAM" : "in RAM"}, routing ${fmt(c.h * 100)}%`;
});
$("heat").addEventListener("mouseleave", () => { $("tip").hidden = true; });
addEventListener("resize", () => { if (last && last.experts) drawHeat(last.experts); });

// ------------------------------------------------------------------ readouts
const RO = [
  ["speed", "decode", "tok/s", "tok_s"], ["prefill", "prompt", "tok/s", "prefill_tok_s_mean"], ["gpu", "GPU", "%", "gpu_util"],
  ["pcie", "PCIe", "GB/s", "gpu_pcie_rx_mb"], ["cpu", "CPU", "%", "cpu"], ["vram", "VRAM", "GB", "gpu_mem_used"], ["temp", "GPU", "°C", "gpu_temp"]];
$("readouts").innerHTML = RO.map(([k, label, unit]) => `<div class="ro"><span class="ro__v" id="rv-${k}">–</span>
  <span class="ro__l">${label} ${unit === "°C" ? "temperature" : ""}</span><svg viewBox="0 0 100 22" preserveAspectRatio="none"><path id="rs-${k}"/></svg></div>`).join("");
const pathOf = (pts) => pts.map((p, i) => `${i ? "L" : "M"}${p[0].toFixed(1)},${p[1].toFixed(1)}`).join("");
function spark(id, v, max) {
  const x = (v || []).map((a) => a || 0);
  if (x.length < 2) return;
  const top = Math.max(max || 0, ...x, 1e-9);
  $(id).setAttribute("d", pathOf(x.map((a, i) => [(i / (x.length - 1)) * 100, 21 - (a / top) * 19])));
}

// ------------------------------------------------------------------ charts
const nice = (v) => { if (!(v > 0)) return 1; const p = 10 ** Math.floor(Math.log10(v)), m = v / p; return (m <= 1 ? 1 : m <= 2 ? 2 : m <= 5 ? 5 : 10) * p; };
const keys = (id, items) => { $(id).innerHTML = items.map(([c, t]) => `<span><i style="background:var(${c})"></i>${esc(t)}</span>`).join(""); };
function grid(W, H, m, ymax, yf, side = "l") {
  const sy = (y) => H - m.b - (y / ymax) * (H - m.t - m.b);
  let s = "";
  for (let i = 0; i <= 4; i++) {
    const y = (ymax * i) / 4;
    s += `<line class="g" x1="${m.l}" x2="${W - m.r}" y1="${sy(y)}" y2="${sy(y)}"/>`;
    s += side === "l" ? `<text class="t" x="${m.l - 8}" y="${sy(y) + 4}" text-anchor="end">${yf(y)}</text>`
                      : `<text class="t" x="${W - m.r + 8}" y="${sy(y) + 4}">${yf(y)}</text>`;
  }
  return {sy, s};
}
function sources(r) {   // a request's routed experts by where they ran (null: the engine did not say)
  if (r.hit_rate == null) return null;
  const p = r.pcie_share || 0;
  return {vram: (1 - p) * r.hit_rate, pcie: p, cpu: (1 - p) * (1 - r.hit_rate)};
}

function drawSources(reqs) {
  const rs = reqs.filter(sources).slice(0, 48).reverse(), W = 640, H = 220, m = {l: 40, r: 44, t: 6, b: 6};
  if (!rs.length) { $("c-src").innerHTML = `<text class="t" x="${m.l}" y="${H / 2}">Send a request to see where its experts ran.</text>`; return; }
  const tmax = nice(Math.max(...rs.map((r) => r.decode_tok_s || 0), 1));
  const a = grid(W, H, m, 1, (y) => `${fmt(y * 100)}%`), b = grid(W, H, m, tmax, (y) => fmt(y), "r");
  const bw = (W - m.l - m.r) / rs.length;
  let s = a.s + b.s.replace(/<line[^>]*>/g, "");
  const line = [];
  rs.forEach((r, i) => {
    const src = sources(r), x = m.l + i * bw;
    let y0 = 0;
    for (const k of ["vram", "pcie", "cpu"]) {
      const h = a.sy(y0) - a.sy(y0 + src[k]);
      s += `<rect x="${(x + 1).toFixed(1)}" width="${Math.max(bw - 2, 1).toFixed(1)}" y="${a.sy(y0 + src[k]).toFixed(1)}" height="${h.toFixed(1)}" fill="var(--${k})"><title>${k === "vram" ? "VRAM" : k === "pcie" ? "PCIe" : "CPU"} ${fmt(src[k] * 100, 1)}%</title></rect>`;
      y0 += src[k];
    }
    if (r.decode_tok_s) line.push([x + bw / 2, b.sy(r.decode_tok_s)]);
  });
  s += `<path class="speed" d="${pathOf(line)}"/>`;
  $("c-src").innerHTML = s;
}

function drawVram(hw, eng) {
  const total = hw.gpu_mem_total, used = hw.gpu_mem_used, cache = (eng.expert_cache_mib || 0) * 1048576;
  if (!total || used == null) { $("c-vram").innerHTML = `<p class="empty">No GPU reading.</p>`; return; }
  const parts = [["--vram", Math.min(cache, used), "experts"], ["--muted", Math.max(0, used - cache), "weights, K/V and buffers"], ["--faint", total - used, "free"]];
  $("c-vram").innerHTML = `<div class="split__bar">${parts.map(([c, b]) => `<span style="width:${(100 * b) / total}%;background:var(${c})"></span>`).join("")}</div>
    <div class="split__keys">${parts.map(([, b, t]) => `<span><b>${fmt(b / GiB, 1)}</b>GB ${t}</span>`).join("")}</div>`;
}

function drawTuning(t) {
  if (!t || !t.runs.length) { $("c-tune").innerHTML = `<p class="empty">Runs after spillway retune.</p>`; return; }
  const max = Math.max(...t.runs.map((r) => r.tok_s), 1);
  $("c-tune").innerHTML = t.runs.map((r) => {
    const best = t.best && r.tok_s === t.best.tok_s ? " best" : "";
    return `<span class="${best}">${esc(r.label)}</span><span class="${best}"><i style="width:${(100 * r.tok_s) / max}%"></i></span><span class="${best}">${r.tok_s ? fmt(r.tok_s, 1) : "failed"}</span>`;
  }).join("");
}

function drawContext(reqs, eng) {
  const W = 400, H = 220, m = {l: 40, r: 10, t: 6, b: 26};
  const pts = reqs.filter((r) => r.decode_tok_s).reverse().map((r) => [r.prompt_total || 0, r.decode_tok_s]);
  const xmax = eng.max_context || nice(Math.max(...pts.map((p) => p[0]), 1)), ymax = nice(Math.max(...pts.map((p) => p[1]), 1));
  const {sy, s} = grid(W, H, m, ymax, (y) => fmt(y));
  const sx = (x) => m.l + (x / xmax) * (W - m.l - m.r);
  let out = s;
  for (let i = 0; i <= 4; i++) out += `<text class="t" x="${sx((xmax * i) / 4)}" y="${H - 6}" text-anchor="middle">${kfmt((xmax * i) / 4)}</text>`;
  out += pts.map((p, i) => `<circle class="dot${i === pts.length - 1 ? " dot--last" : ""}" cx="${sx(p[0]).toFixed(1)}" cy="${sy(p[1]).toFixed(1)}" r="${i === pts.length - 1 ? 5 : 3.5}"><title>${kfmt(p[0])} tokens: ${fmt(p[1], 1)} tok/s</title></circle>`).join("");
  $("c-ctx").innerHTML = out;
}

function drawLive(h) {
  const S = [["--ink", "decode tok/s", h.tok_s, null, 0], ["--vram", "GPU %", h.gpu_util, 100, 0], ["--cpu", "CPU %", h.cpu, 100, 0], ["--pcie", "PCIe GB/s", (h.gpu_pcie_rx_mb || []).map((v) => (v || 0) / 1000), null, 1]];
  const W = 640, H = 220;
  let s = "";
  for (let i = 0; i <= 4; i++) s += `<line class="g" x1="0" x2="${W}" y1="${6 + i * 52}" y2="${6 + i * 52}"/>`;
  for (const [c, , v, max] of S) {
    const x = (v || []).map((a) => a || 0);
    if (x.length < 2) continue;
    const top = Math.max(max || 0, ...x, 1e-9);
    s += `<path d="${pathOf(x.map((a, i) => [(i / (x.length - 1)) * W, 214 - (a / top) * 208]))}" fill="none" stroke="var(${c})" stroke-width="2" vector-effect="non-scaling-stroke"/>`;
  }
  $("c-live").innerHTML = s;
  keys("k-live", S.map(([c, n, v, , d]) => [c, `${n} ${fmt((v || []).at(-1), d)}`]));
}

// ------------------------------------------------------------------ the loop
let last = null;
function setState(s, text) { $("state").dataset.s = s; $("state").textContent = text; }
function render(m) {
  last = m;
  const live = m.live || {}, hw = m.hardware || {}, eng = m.engine || {}, h = m.history || {}, reqs = m.requests || [];
  if (live.state === "generating") setState("generating", `Generating ${fmt(live.tok_s, 1)} tok/s`);
  else if (live.state === "reading") setState("reading", live.prompt_total ? `Reading the prompt, ${fmt((100 * live.prompt_read) / live.prompt_total)}%` : "Reading the prompt");
  else setState("idle", "Idle");
  const r0 = reqs[0] || {};
  const v = {speed: live.tok_s ?? r0.decode_tok_s, prefill: live.prefill_tok_s_mean ?? (r0.prompt_ms ? (r0.prompt_read || r0.prompt_tokens) / (r0.prompt_ms / 1000) : null),
             gpu: hw.gpu_util, pcie: hw.gpu_pcie_rx_mb == null ? null : hw.gpu_pcie_rx_mb / 1000, cpu: hw.cpu,
             vram: hw.gpu_mem_used == null ? null : hw.gpu_mem_used / GiB, temp: hw.gpu_temp};
  for (const [k, , unit, series] of RO) {
    const d = k === "vram" || k === "pcie" ? 1 : 0;
    $(`rv-${k}`).innerHTML = `${fmt(v[k], d)}<small>${unit}</small>`;
    spark(`rs-${k}`, h[series], k === "gpu" || k === "cpu" ? 100 : k === "vram" ? hw.gpu_mem_total : null);
  }
  if (m.experts) drawHeat(m.experts);
  else $("heat-count").innerHTML = `<small>appears after the first request</small>`;
  keys("k-src", [["--vram", "VRAM"], ["--pcie", "PCIe"], ["--cpu", "CPU"], ["--ink", "decode tok/s"]]);
  drawSources(reqs);
  drawVram(hw, eng);
  drawTuning(m.tuning);
  drawContext(reqs, eng);
  drawLive(h);
}

async function poll() {
  try {
    const r = await fetch("metrics?requests=all", {headers: key ? {Authorization: "Bearer " + key} : {}});
    if (r.ok) render(await r.json());
    else setState("error", r.status === 401 ? "Needs the API key: open this page as /#key=<key>" : `The server answered ${r.status}`);
  } catch (e) {
    setState("error", "Can't reach the server");
  }
  setTimeout(poll, 1000);
}
poll();
