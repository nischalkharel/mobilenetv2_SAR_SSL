// Shared helpers for the tracker pages: data loading, colours, SVG charts, tables, tooltip.
// Every chart is plain SVG built from strings; no library.

const FAM = {base: "var(--base)", dino: "var(--dino)", lejepa: "var(--lejepa)", spark: "var(--spark)", cond: "var(--cond)", region: "var(--region)"};
const POOL = {none: "ImageNet start, no SSL", atr: "SSL on ATRNet-STAR chips (in-domain)", sardet: "SSL on SARDet-100K scenes (cross-dataset)"};
const POOL_DET = {none: "ImageNet start, no SSL", sardet: "SSL on SARDet-100K scenes (in-domain for detection)"};
const BUDGETS = ["1pct", "3pct", "10pct"];
const bud = b => b.replace("pct", " %");
const esc = s => String(s ?? "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/"/g, "&quot;");
const mean = a => a.length ? a.reduce((x, y) => x + y, 0) / a.length : NaN;
const sd = a => { if (a.length < 2) return 0; const m = mean(a); return Math.sqrt(a.reduce((s, x) => s + (x - m) ** 2, 0) / (a.length - 1)); };
const f1 = x => (x >= 0 ? "+" : "−") + Math.abs(x).toFixed(1);
const fx = (x, n = 1) => (x == null || Number.isNaN(x)) ? "–" : x.toFixed(n);
const $ = id => document.getElementById(id);

async function load(name) {
  const r = await fetch(`/data/${name}.json`, {cache: "no-store"});
  if (!r.ok) throw new Error(`${name}.json: ${r.status}`);
  return r.json();
}

// ---- shell: nav + meta line ----
async function shell(active) {
  const pages = [["index.html", "Overview"], ["classification.html", "Classification"], ["detection.html", "Detection"], ["maps.html", "Response maps"], ["chips.html", "Chips"], ["runs.html", "Runs"], ["ideas.html", "Ideas"], ["glossary.html", "Glossary"]];
  const nav = $("nav");
  nav.innerHTML = pages.map(([h, t]) => `<a href="${h}" class="${h === active ? "active" : ""}">${t}</a>`).join("");
  try {
    const m = await load("meta");
    $("meta").innerHTML = `data collected ${esc(m.collected.replace("T", " "))} at ${esc(m.commit)} &middot; <a href="#" id="refresh">refresh</a>`;
    $("refresh").onclick = async e => { e.preventDefault(); e.target.textContent = "collecting…"; const r = await (await fetch("/api/collect", {method: "POST"})).json(); if (r.ok) location.reload(); else alert(r.output); };
    return m;
  } catch (e) { $("meta").textContent = "no data yet: run tracker/collect.py"; return null; }
}

// ---- tooltip ----
let tipEl;
function hover(el, html) {
  if (!tipEl) { tipEl = document.createElement("div"); tipEl.className = "tip"; document.body.appendChild(tipEl); }
  el.addEventListener("mousemove", e => { tipEl.innerHTML = html; tipEl.style.opacity = 1; tipEl.style.left = Math.min(e.clientX + 12, innerWidth - 330) + "px"; tipEl.style.top = (e.clientY + 14) + "px"; });
  el.addEventListener("mouseleave", () => tipEl.style.opacity = 0);
}
function bindTips(host) { host.querySelectorAll("[data-tip]").forEach(g => hover(g, g.dataset.tip)); }

// ---- cells of a method: values per split for one budget (null where missing) ----
function vals(method, budget, field = "acc", splits = [1, 2, 3]) {
  return splits.map(s => { const c = method.cells[`${budget}/${s}`]; return c ? c[field] : null; });
}
const present = a => a.filter(v => v != null);
function hatch(id, color) { return `<pattern id="${id}" patternUnits="userSpaceOnUse" width="6" height="6" patternTransform="rotate(45)"><rect width="6" height="6" fill="var(--surface)"/><rect width="2.4" height="6" fill="${color}"/></pattern>`; }
function headRow(y, h, W, title) { return `<rect x="0" y="${y}" width="${W}" height="${h}" fill="var(--surface2)"/><text x="8" y="${y + h - 7}" font-size="11" font-weight="600" letter-spacing="0.06em" fill="var(--fg2)">${esc(title.toUpperCase())}</text>`; }

// ---- grouped horizontal bars: groups = [{title, rows:[{label, family, control, hatched, v:{panelKey:[numbers]}}]}],
//      panels = [{key, title, xmax, ticks, ref, refLabel, fmt}] ----
function hbars(host, groups, panels, opt = {}) {
  const labelW = opt.labelW || 250, panelW = opt.panelW || 210, gap = 20, top = 30, rowH = 25, headH = 23;
  const rows = []; groups.forEach(g => { rows.push({head: g.title}); g.rows.forEach(r => rows.push({r})); });
  const W = labelW + panels.length * (panelW + gap) - gap, ys = []; let yy = top;
  rows.forEach(x => { ys.push(yy); yy += x.head ? headH : rowH; }); const H = yy + 26;
  let s = `<svg viewBox="0 0 ${W} ${H}" style="min-width:${Math.min(W, 980)}px" role="img"><defs>`;
  rows.forEach((x, i) => { if (x.r && x.r.hatched) s += hatch(`h${host.id}${i}`, FAM[x.r.family]); });
  s += `</defs>`;
  rows.forEach((x, i) => { if (x.r) s += `<text x="${labelW - 12}" y="${ys[i] + rowH / 2 + 4}" font-size="12.5" text-anchor="end" fill="var(--fg)">${esc(x.r.label)}</text>`; });
  panels.forEach((p, k) => {
    const x0 = labelW + k * (panelW + gap), right = 46, xmin = p.xmin || 0, sx = v => x0 + (v - xmin) / (p.xmax - xmin) * (panelW - right), fmt = p.fmt || (v => v.toFixed(1));
    s += `<text x="${x0}" y="14" font-size="13" font-weight="600" fill="var(--fg)">${esc(p.title)}</text>`;
    p.ticks.forEach(t => s += `<line x1="${sx(t)}" x2="${sx(t)}" y1="${top - 4}" y2="${H - 22}" stroke="var(--grid)"/><text x="${sx(t)}" y="${H - 7}" font-size="10" text-anchor="middle" fill="var(--fg3)">${t}</text>`);
    if (p.ref != null) s += `<line x1="${sx(p.ref)}" x2="${sx(p.ref)}" y1="${top - 4}" y2="${H - 22}" stroke="var(--ref)" stroke-dasharray="3 3"/>${p.refLabel ? `<text x="${sx(p.ref) + 3}" y="${top + 8}" font-size="9.5" fill="var(--fg3)">${esc(p.refLabel)}</text>` : ""}`;
    rows.forEach((x, i) => {
      if (!x.r) return; const r = x.r, v = present(r.v[p.key] || []); if (!v.length) { s += `<text x="${sx(xmin) + 4}" y="${ys[i] + rowH / 2 + 4}" font-size="11" fill="var(--fg3)">not run</text>`; return; }
      const m = mean(v), y = ys[i], fill = r.hatched ? `url(#h${host.id}${i})` : FAM[r.family];
      s += `<g class="bar" data-tip="${esc(r.label)}, ${esc(p.title)}: ${fmt(m)} (${v.map(fmt).join(" / ")})">`;
      s += `<rect x="${sx(xmin)}" y="${y + 4}" width="${Math.max(0, sx(m) - sx(xmin))}" height="${rowH - 9}" rx="3" fill="${fill}" stroke="${r.hatched ? FAM[r.family] : "none"}"${r.control ? ' opacity="0.5"' : ""}/>`;
      v.forEach(q => s += `<circle cx="${sx(q)}" cy="${y + rowH / 2 - 0.5}" r="3" fill="var(--fg)" stroke="var(--surface)" stroke-width="1.5"/>`);
      s += `<text x="${sx(m) + 6}" y="${y + rowH / 2 + 3.5}" font-size="11.5" font-family="IBM Plex Mono, Consolas, monospace" fill="var(--fg)">${fmt(m)}</text>`;
      s += `<rect x="${x0}" y="${y}" width="${panelW}" height="${rowH}" fill="transparent"/></g>`;
    });
  });
  rows.forEach((x, i) => { if (x.head) s += headRow(ys[i], headH, W, x.head); });
  host.innerHTML = s + `</svg>`; bindTips(host);
}

// ---- dot plot of paired differences: rows = [{label, group, sub, family, d:[numbers]}] ----
function dotplot(host, rows, xmin, xmax, unit, fmt = f1) {
  const W = 860, left = 250, right = 24, top = 22, rowH = 22, H = top + rowH * rows.length + 30;
  const sx = v => left + (v - xmin) / (xmax - xmin) * (W - left - right);
  let s = `<svg viewBox="0 0 ${W} ${H}" style="min-width:760px" role="img">`;
  const step = (xmax - xmin) > 12 ? 5 : (xmax - xmin) > 6 ? 2 : 1;
  for (let t = Math.ceil(xmin / step) * step; t <= xmax; t += step) s += `<line x1="${sx(t)}" x2="${sx(t)}" y1="${top - 6}" y2="${H - 24}" stroke="${t === 0 ? "var(--ref)" : "var(--grid)"}" stroke-width="${t === 0 ? 1.2 : 1}"/><text x="${sx(t)}" y="${H - 8}" font-size="10.5" text-anchor="middle" fill="var(--fg3)">${t > 0 ? "+" + t : t}</text>`;
  s += `<text x="${W - right}" y="12" font-size="11" text-anchor="end" fill="var(--fg3)">${esc(unit)}</text>`;
  let last = null;
  rows.forEach((r, i) => {
    const y = top + i * rowH, d = present(r.d); if (r.group !== last) { if (i > 0) s += `<line x1="0" x2="${W}" y1="${y}" y2="${y}" stroke="var(--line)"/>`; last = r.group; }
    if (!d.length) { s += `<text x="${left - 10}" y="${y + rowH / 2 + 4}" font-size="12.5" text-anchor="end" fill="var(--fg3)">${esc(r.label)}</text><text x="${sx(0) + 6}" y="${y + rowH / 2 + 4}" font-size="11" fill="var(--fg3)">not run</text>`; return; }
    const m = mean(d), pos = d.every(v => v > 0), neg = d.every(v => v < 0);
    s += `<g class="r" data-tip="${esc(r.label)} ${esc(r.sub || "")}: mean ${fmt(m)} (${d.map(fmt).join(" / ")})">`;
    s += `<text x="${left - 10}" y="${y + rowH / 2 + 4}" font-size="12.5" text-anchor="end" fill="var(--fg)">${esc(r.label)}</text>`;
    s += `<text x="8" y="${y + rowH / 2 + 4}" font-size="11.5" fill="var(--fg3)">${esc(r.sub || "")}</text>`;
    s += `<rect x="${Math.min(sx(0), sx(m))}" y="${y + rowH / 2 - 5}" width="${Math.abs(sx(m) - sx(0))}" height="10" rx="2" fill="${FAM[r.family]}" opacity="0.45"/>`;
    d.forEach(v => s += `<circle cx="${sx(v)}" cy="${y + rowH / 2}" r="4" fill="${FAM[r.family]}" stroke="var(--surface)" stroke-width="1.5"/>`);
    s += `<text x="${W - right}" y="${y + rowH / 2 + 4}" font-size="11.5" text-anchor="end" font-family="IBM Plex Mono, Consolas, monospace" fill="${pos ? "var(--good)" : neg ? "var(--bad)" : "var(--fg2)"}">${fmt(m)}${pos || neg ? "" : " ~"}</text>`;
    s += `<rect x="0" y="${y}" width="${W}" height="${rowH}" fill="transparent"/></g>`;
  });
  host.innerHTML = s + `</svg>`; bindTips(host);
}

// ---- sparkline of a curve ----
function spark(v, w = 96, h = 22, color = "var(--accent)") {
  const a = present(v || []); if (a.length < 2) return "";
  const lo = Math.min(...a), hi = Math.max(...a), sy = y => hi === lo ? h / 2 : h - 2 - (y - lo) / (hi - lo) * (h - 4);
  const pts = a.map((y, i) => `${(i / (a.length - 1) * (w - 2) + 1).toFixed(1)},${sy(y).toFixed(1)}`).join(" ");
  return `<svg class="spark" width="${w}" height="${h}" viewBox="0 0 ${w} ${h}"><polyline points="${pts}" fill="none" stroke="${color}" stroke-width="1.4"/></svg>`;
}

// ---- method display helpers ----
const famDot = f => `<span class="dot" style="background:${FAM[f]}"></span>`;
function label(m) { return `${famDot(m.family)}${esc(m.label)}${m.control ? ' <span class="pill">control</span>' : ""}`; }
function groupsByPool(methods, pools = POOL) {
  return Object.keys(pools).map(p => ({key: p, title: pools[p], rows: methods.filter(m => m.ssl === p)})).filter(g => g.rows.length);
}
function nDone(m) { return Object.values(m.cells).filter(c => c.status === "completed" || c.status === "from runs.csv").length; }
function drawer(host, m, field, fmt, unitName) {
  const cells = BUDGETS.flatMap(b => [1, 2, 3].map(s => [b, s, m.cells[`${b}/${s}`]]));
  host.innerHTML = `<div class="eyebrow">${esc(m.label)} &middot; ${esc(POOL[m.ssl] || "")}</div><p class="sub" style="margin:4px 0 10px">${esc(m.note || "")}</p>
    <div class="grid">${cells.map(([b, s, c]) => c ? `<div class="cell"><b>${bud(b)} labels, split ${s}</b> &middot; <span class="mono">${fmt(c[field])}</span> ${unitName}<br>
      <span class="k">run</span> <code>${esc(c.run)}</code><br><span class="k">best epoch</span> ${c.best_epoch ?? "?"} of ${c.epochs_run} &middot; <span class="k">time</span> ${fx(c.train_minutes, 0)} min &middot; <span class="k">commit</span> <code>${esc(c.commit)}</code><br>
      ${c.backbone ? `<span class="k">backbone</span> <code>${esc(c.backbone)}</code><br>` : ""}${c.ssl_run ? `<span class="k">SSL run</span> <code>${esc(String(c.ssl_run).split("/").pop())}</code><br>` : ""}
      <span class="k">${c.val_curve && c.val_curve.length ? "validation curve" : ""}</span> ${spark(c.val_curve)}</div>`
      : `<div class="cell empty">${bud(b)} labels, split ${s}: not run</div>`).join("")}</div>`;
  host.scrollIntoView({behavior: "smooth", block: "nearest"});
}
