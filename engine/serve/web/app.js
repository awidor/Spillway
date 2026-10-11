// serve/web/app.js - Spillway's dashboard: the server's /metrics, once a second, as charts. No framework.
"use strict";

const $ = (id) => document.getElementById(id);
const fmt = (n, d = 0) => (n == null || Number.isNaN(n) ? "–" : Number(n).toLocaleString(undefined, {maximumFractionDigits: d, minimumFractionDigits: d}));
const kfmt = (n) => (n == null ? "–" : n >= 1000 ? `${fmt(n / 1000, n >= 10000 ? 0 : 1)}k` : fmt(n));
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]));
const GiB = 1073741824, EXPERTS = 40 * 256;
const C = {vram: "var(--st-accent)", pcie: "var(--st-info)", cpu: "var(--st-warn)", tok: "var(--st-ink)", gpu: "var(--st-danger)"};

// ------------------------------------------------------------------ theme, key
function setTheme(t, save) {
  document.documentElement.dataset.theme = t;
  if (save) try { localStorage.setItem("strata.theme", t); } catch (e) { /* ignore */ }
  $("theme-icon").setAttribute("href", `web/sprite.svg#i-${t === "dark" ? "sun" : "moon"}`);
}
$("theme-btn").onclick = () => setTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark", true);
setTheme(document.documentElement.dataset.theme || "light", false);

// the API key, when the service has one: http://host:port/#key=... (the fragment never reaches the network)
let key = "";
try { key = localStorage.getItem("strata.apikey") || ""; } catch (e) { /* ignore */ }
const hk = new URLSearchParams(location.hash.slice(1)).get("key");
if (hk) { key = hk; try { localStorage.setItem("strata.apikey", hk); } catch (e) { /* ignore */ } history.replaceState(null, "", location.pathname); }

// ------------------------------------------------------------------ svg helpers
const NS = "http://www.w3.org/2000/svg";
function svg(el, html) { el.innerHTML = html; }
const nice = (v) => { if (!(v > 0)) return 1; const p = 10 ** Math.floor(Math.log10(v)), m = v / p; return (m <= 1 ? 1 : m <= 2 ? 2 : m <= 5 ? 5 : 10) * p; };
const path = (pts) => pts.map((p, i) => `${i ? "L" : "M"}${p[0].toFixed(1)},${p[1].toFixed(1)}`).join("");
const legend = (id, items) => { $(id).innerHTML = items.map(([c, t]) => `<i style="background:${c}"></i>${esc(t)}`).join(""); };

function axes(W, H, m, xmax, ymax, xf, yf) {   // gridlines and tick labels; returns the plot's scale functions
  const sx = (x) => m.l + (x / xmax) * (W - m.l - m.r), sy = (y) => H - m.b - (y / ymax) * (H - m.t - m.b);
  let g = "";
  for (let i = 0; i <= 4; i++) {
    const y = (ymax * i) / 4, x = (xmax * i) / 4;
    g += `<line class="grid" x1="${m.l}" x2="${W - m.r}" y1="${sy(y)}" y2="${sy(y)}"/>`;
    g += `<text class="tick" x="${m.l - 6}" y="${sy(y) + 4}" text-anchor="end">${yf(y)}</text>`;
    if (xf) g += `<text class="tick" x="${sx(x)}" y="${H - m.b + 15}" text-anchor="middle">${xf(x)}</text>`;
  }
  return {sx, sy, g};
}

function scatter(el, pts, xmax, xf) {   // pts: [x, tok/s, title]
  const W = 320, H = 200, m = {l: 36, r: 12, t: 10, b: 24};
  const ymax = nice(Math.max(1, ...pts.map((p) => p[1])));
  const {sx, sy, g} = axes(W, H, m, xmax, ymax, xf, (y) => fmt(y));
  svg(el, g + pts.map((p, i) => `<circle class="dot${i === pts.length - 1 ? " dot--last" : ""}" cx="${sx(p[0])}" cy="${sy(p[1])}" r="${i === pts.length - 1 ? 5 : 3.5}"><title>${esc(p[2])}</title></circle>`).join(""));
}

// ------------------------------------------------------------------ tiles
const TILES = [
  ["speed", "tok/s", "tok_s"], ["prefill", "prefill tok/s", "prefill_tok_s_mean"], ["gpu", "GPU %", "gpu_util"],
  ["vram", "VRAM GB", "gpu_mem_used"], ["pcie", "PCIe MB/s", "gpu_pcie_rx_mb"], ["cpu", "CPU %", "cpu"],
  ["ram", "RAM GB", "ram_used"], ["temp", "GPU °C", "gpu_temp"]];
$("tiles").innerHTML = TILES.map(([k, label]) => `<div class="st-card tile"><span class="tile__label">${label}</span>
  <span class="tile__value" id="tv-${k}">–</span><svg class="tile__spark" id="ts-${k}" viewBox="0 0 100 30" preserveAspectRatio="none">
  <path class="area"/><path class="line" vector-effect="non-scaling-stroke"/></svg></div>`).join("");
function spark(id, values, max) {
  const v = (values || []).map((x) => x || 0), el = $(id);
  if (v.length < 2) return;
  const top = Math.max(max || 0, ...v, 1e-9), line = path(v.map((x, i) => [(i / (v.length - 1)) * 100, 29 - (x / top) * 27]));
  el.querySelector(".line").setAttribute("d", line);
  el.querySelector(".area").setAttribute("d", `${line}L100,30L0,30Z`);
}

// ------------------------------------------------------------------ the charts
function sources(r) {   // a request's routed experts by where they were computed (null: the engine did not say)
  if (r.hit_rate == null) return null;
  const p = r.pcie_share || 0;
  return {vram: (1 - p) * r.hit_rate, pcie: p, cpu: (1 - p) * (1 - r.hit_rate)};
}

function renderSources(reqs) {
  const rs = reqs.filter((r) => sources(r)).slice(0, 40).reverse(), W = 600, H = 200, m = {l: 36, r: 40, t: 10, b: 10};
  const tmax = nice(Math.max(1, ...rs.map((r) => r.decode_tok_s || 0)));
  const {sy, g} = axes(W, H, m, 1, 1, null, (y) => `${fmt(y * 100)}%`);
  const bw = (W - m.l - m.r) / Math.max(rs.length, 1);
  let out = g, tok = [];
  rs.forEach((r, i) => {
    const s = sources(r), x = m.l + i * bw;
    let y0 = 0;
    for (const k of ["vram", "pcie", "cpu"]) {
      out += `<rect x="${x + 1}" width="${Math.max(bw - 2, 1)}" y="${sy(y0 + s[k])}" height="${sy(y0) - sy(y0 + s[k])}" fill="${C[k]}"><title>${k} ${fmt(s[k] * 100, 1)}%</title></rect>`;
      y0 += s[k];
    }
    if (r.decode_tok_s) tok.push([x + bw / 2, sy(r.decode_tok_s / tmax)]);
  });
  for (let i = 0; i <= 4; i++) out += `<text class="tick" x="${W - m.r + 6}" y="${sy(i / 4) + 4}">${fmt((tmax * i) / 4)}</text>`;
  out += `<path class="tokline" d="${path(tok)}"/>` + tok.map((p) => `<circle class="tokdot" cx="${p[0]}" cy="${p[1]}" r="3"/>`).join("");
  svg($("ch-src"), out);
}

function renderVram(hw, eng) {
  const total = hw.gpu_mem_total, used = hw.gpu_mem_used, cache = (eng.expert_cache_mib || 0) * 1048576;
  if (!total || used == null) return;
  const parts = [["cache", Math.min(cache, used), C.vram], ["other", Math.max(0, used - cache), C.cpu], ["free", total - used, "var(--st-line)"]];
  legend("lg-vram", parts.map(([n, b, c]) => [c, `${n} ${fmt(b / GiB, 1)} GB`]));
  // a ring: the GPU's memory, and inside it the expert slots against the model's 10,240 experts
  const R = 78, r2 = 56, cx = 150, cy = 100, L = 2 * Math.PI * R, L2 = 2 * Math.PI * r2;
  let off = 0, out = "";
  for (const [, b, c] of parts) {
    const len = (b / total) * L;
    out += `<circle cx="${cx}" cy="${cy}" r="${R}" fill="none" stroke="${c}" stroke-width="18" stroke-dasharray="${len} ${L - len}" stroke-dashoffset="${-off}" transform="rotate(-90 ${cx} ${cy})"/>`;
    off += len;
  }
  const f = Math.min(1, (eng.expert_slots || 0) / EXPERTS);
  out += `<circle cx="${cx}" cy="${cy}" r="${r2}" fill="none" stroke="var(--st-line)" stroke-width="8"/>
    <circle cx="${cx}" cy="${cy}" r="${r2}" fill="none" stroke="${C.vram}" stroke-width="8" stroke-dasharray="${f * L2} ${L2}" transform="rotate(-90 ${cx} ${cy})"/>
    <text class="big" x="${cx}" y="${cy + 4}" text-anchor="middle">${fmt(f * 100)}%</text>
    <text class="tick" x="${cx}" y="${cy + 22}" text-anchor="middle">${kfmt(eng.expert_slots)} / ${kfmt(EXPERTS)}</text>`;
  svg($("ch-vram"), out);
}

function renderLive(h) {
  const series = [["tok/s", h.tok_s, C.tok, null], ["GPU %", h.gpu_util, C.gpu, 100], ["CPU %", h.cpu, C.cpu, 100], ["PCIe MB/s", h.gpu_pcie_rx_mb, C.pcie, null]];
  const W = 600, H = 200;
  let out = "";
  for (let i = 0; i <= 4; i++) out += `<line class="grid" x1="0" x2="${W}" y1="${10 + i * 45}" y2="${10 + i * 45}"/>`;
  for (const [, v, c, max] of series) {
    const x = (v || []).map((a) => a || 0);
    if (x.length < 2) continue;
    const top = Math.max(max || 0, ...x, 1e-9);
    out += `<path d="${path(x.map((a, i) => [(i / (x.length - 1)) * W, 190 - (a / top) * 180]))}" fill="none" stroke="${c}" stroke-width="2" vector-effect="non-scaling-stroke"/>`;
  }
  svg($("ch-live"), out);
  legend("lg-live", series.map(([n, v, c]) => [c, `${n} ${fmt((v || []).at(-1), 0)}`]));
}

function renderTuning(t) {
  const el = $("ch-tune");
  if (!t || !t.runs.length) { el.innerHTML = ""; return; }
  const max = Math.max(...t.runs.map((r) => r.tok_s), 1);
  el.innerHTML = t.runs.map((r) => {
    const best = t.best && r.tok_s === t.best.tok_s;
    return `<div class="bar${best ? " bar--best" : ""}" title="${esc(r.label)}"><span class="bar__label">${esc(r.label)}</span>
      <span class="bar__track"><span class="bar__fill" style="width:${(100 * r.tok_s) / max}%"></span></span>
      <span class="bar__value">${fmt(r.tok_s, 1)}</span></div>`;
  }).join("");
}

function render(m) {
  const live = m.live || {}, hw = m.hardware || {}, eng = m.engine || {}, h = m.history || {}, reqs = m.requests || [];
  $("pill").dataset.state = live.state === "reading" || live.state === "generating" ? live.state : "idle";
  $("pill-text").textContent = live.state === "generating" ? `${fmt(live.tok_s, 1)} tok/s`
    : live.state === "reading" && live.prompt_total ? `${fmt((100 * live.prompt_read) / live.prompt_total)}%` : live.state || "idle";
  const last = reqs[0] || {};
  const tv = {speed: live.tok_s ?? last.decode_tok_s, prefill: live.prefill_tok_s_mean ?? (last.prompt_ms ? (last.prompt_read || last.prompt_tokens) / (last.prompt_ms / 1000) : null),
              gpu: hw.gpu_util, vram: hw.gpu_mem_used / GiB, pcie: hw.gpu_pcie_rx_mb, cpu: hw.cpu, ram: hw.ram_used / GiB, temp: hw.gpu_temp};
  for (const [k, , s] of TILES) {
    $(`tv-${k}`).textContent = fmt(tv[k], k === "vram" || k === "ram" ? 1 : 0);
    spark(`ts-${k}`, h[s], k === "gpu" || k === "cpu" ? 100 : k === "vram" ? hw.gpu_mem_total : null);
  }
  legend("lg-src", [[C.vram, "VRAM"], [C.pcie, "PCIe"], [C.cpu, "CPU"], [C.tok, "tok/s"]]);
  renderSources(reqs);
  renderVram(hw, eng);
  const done = reqs.filter((r) => r.decode_tok_s).reverse();
  const ctxMax = nice(Math.max(eng.max_context || 0, ...done.map((r) => r.prompt_total || 0)));
  scatter($("ch-ctx"), done.map((r) => [r.prompt_total || 0, r.decode_tok_s, `${kfmt(r.prompt_total)} tokens · ${fmt(r.decode_tok_s, 1)} tok/s`]), ctxMax, (x) => kfmt(x));
  scatter($("ch-hit"), done.filter(sources).map((r) => { const s = sources(r); return [s.vram * 100, r.decode_tok_s, `VRAM ${fmt(s.vram * 100, 1)}% · ${fmt(r.decode_tok_s, 1)} tok/s`]; }), 100, (x) => `${fmt(x)}%`);
  scatter($("ch-draft"), done.filter((r) => r.drafts_offered).map((r) => { const a = (100 * r.drafts_accepted) / r.drafts_offered; return [a, r.decode_tok_s, `${fmt(a, 1)}% · ${fmt(r.decode_tok_s, 1)} tok/s`]; }), 100, (x) => `${fmt(x)}%`);
  renderLive(h);
  renderTuning(m.tuning);
}

async function poll() {
  try {
    const r = await fetch("metrics?requests=all", {headers: key ? {Authorization: "Bearer " + key} : {}});
    if (r.ok) render(await r.json());
    else { $("pill").dataset.state = "queued"; $("pill-text").textContent = r.status === 401 ? "key" : `HTTP ${r.status}`; }
  } catch (e) {
    $("pill").dataset.state = "queued"; $("pill-text").textContent = "offline";
  }
  setTimeout(poll, 1000);
}
poll();
