// Precinct viewer: live (WebSocket) and replay (trace .jsonl) modes share one frame format.
"use strict";

const S = {
  header: null, frames: [], cur: -1, follow: true, live: false, ws: null,
  sel: null,           // {type:'cam', id} | {type:'gid', gid} | {type:'region', r}
  metrics: [], paused: false, terrKey: "", terr: null, dpr: window.devicePixelRatio || 1,
  aliases: new Map(), lastSeenEvent: -1, newCams: [],
};
const $ = (id) => document.getElementById(id);
const map = $("map"), ctx = map.getContext("2d");

// ------------------------------------------------------------------ colours
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
function regionColor(r, a = 1) {
  const h = (r * 137.508 + 20) % 360;
  return `hsla(${h}, 55%, 55%, ${a})`;
}
function gidColor(g) {
  let h = 0;
  for (let i = 0; i < g.length; i++) h = (h * 31 + g.charCodeAt(i)) >>> 0;
  return `hsl(${h % 360}, 70%, 45%)`;
}

// ------------------------------------------------------------------ geometry
let view = { s: 1, ox: 0, oy: 0 };
let base = { s: 1, ox: 0, oy: 0 };           // whole city fitted to the canvas
let zoom = { k: 1, x: 0, y: 0 };             // user zoom / pan on top of it (canvas pixels)
function resize() {
  S.dpr = window.devicePixelRatio || 1;          // changes with browser zoom
  const r = map.getBoundingClientRect();
  const w = Math.round(r.width * S.dpr), h = Math.round(r.height * S.dpr);
  if (w === map.width && h === map.height) return;
  map.width = w; map.height = h;
  fit(); draw();
}
function fit() {
  if (!S.header) return;
  const L = S.header.L, pad = 24 * S.dpr;
  const s = Math.min((map.width - 2 * pad) / L, (map.height - 2 * pad) / L);
  base = { s, ox: (map.width - s * L) / 2, oy: (map.height - s * L) / 2 };
  applyZoom();
}
function applyZoom() {
  view = { s: base.s * zoom.k, ox: base.ox * zoom.k + zoom.x, oy: base.oy * zoom.k + zoom.y };
  $("btnFit").hidden = zoom.k === 1 && zoom.x === 0 && zoom.y === 0;
}
function resetZoom() { zoom = { k: 1, x: 0, y: 0 }; applyZoom(); schedule(); }
const ZMAX = 60;
const X = (x) => view.ox + x * view.s;
const Y = (y) => view.oy + (S.header.L - y) * view.s;
const invX = (px) => (px - view.ox) / view.s;
const invY = (py) => S.header.L - (py - view.oy) / view.s;

// ------------------------------------------------------------------ data
function allCams() { return S.header ? S.header.cams.concat(S.newCams) : []; }
function onHeader(h) {
  S.header = h; S.frames = []; S.cur = -1; S.newCams = []; S.terr = null; S.terrKey = "";
  S.metrics = []; S.aliases = new Map(); S.lastSeenEvent = -1;
  $("events").innerHTML = "";
  zoom = { k: 1, x: 0, y: 0 };            // new city: show all of it
  fit();
}
function onFrame(f) {
  if (f.cams_new && f.cams_new.length) for (const c of f.cams_new) S.newCams[c[0] - S.header.cams.length] = [c[1], c[2], c[3]];
  f.vmap = new Map(f.view.map((v) => [v[0], v]));
  if (f.twin) f.tvmap = new Map(f.twin.view.map((v) => [v[0], v]));
  S.frames.push(f);
  if (S.live && S.frames.length > 1800) {
    // keep the last ~2 minutes in live mode so drawing stays fast
    const drop = 600;
    S.frames.splice(0, drop);
    S.cur = Math.max(0, S.cur - drop); S.lastSeenEvent -= drop;
  }
  const i = S.frames.length - 1;
  if (!S.loading && i > S.lastSeenEvent) { S.lastSeenEvent = i; for (const e of f.events) handleEvent(e, f); }
  if (S.follow) S.cur = i;
  $("scrub").max = S.frames.length - 1;
  if (S.follow) $("scrub").value = S.cur;
  schedule();
}

const EV = {
  crash_center: (i) => [`Region ${i.region}: center cam ${i.node} crashed`, "bad"],
  crash: (i) => [`Camera ${i.node} crashed`, "bad"],
  recover: (i) => [`Camera ${i.node} recovered`, "good"],
  center_up: (i) => [`Region ${i.region}: cam ${i.node} is center (term ${i.term})`, "good"],
  agg_crash: () => ["Aggregator crashed — regions keep tracking", "bad"],
  agg_recover: () => ["Aggregator restarting, requesting snapshots", ""],
  agg_rebuilt: (i) => [`Global view rebuilt in ${(i.duration * 1000).toFixed(0)} ms`, "good"],
  merge: (i) => [i.why === "reid" ? `Re-identified ${i.loser} as ${i.survivor} (crossed a gap)`
                                  : `Merged duplicate ${i.loser} → ${i.survivor}`, ""],
  member_failed: (i) => [`Region ${i.region}: detected failed cams ${i.cams.join(",")}`, "bad"],
  voter_change: (i) => [`Region ${i.region}: voters → [${i.voters.join(",")}]`, ""],
  loss: (i) => [`Message loss set to ${(i.p * 100).toFixed(0)}%`, ""],
  join: (i) => [`Camera ${i.node} joined region ${i.region}`, "good"],
  leave: (i) => [`Camera ${i.node} left`, ""],
  region_empty: (i) => [`Region ${i.region} empty — territory absorbed by neighbours`, "bad"],
  appoint: (i) => [`Region ${i.region}: directory appointed cam ${i.node} (no replication)`, ""],
};
const TOAST = new Set(["crash_center", "center_up", "agg_crash", "agg_rebuilt", "region_empty", "loss", "join"]);
function handleEvent(e, f) {
  const [t, kind, info] = e;
  if (kind === "merge" || kind === "alias") S.aliases.set(info.loser, info.survivor);
  if (kind === "merge" && S.sel && S.sel.type === "gid" && S.sel.gid === info.loser) S.sel.gid = info.survivor;
  let txt = null, cls = "";
  if (EV[kind]) [txt, cls] = EV[kind](info);
  else if (kind === "accept" && S.sel && S.sel.type === "gid" && info.gid === S.sel.gid)
    [txt, cls] = [`Handoff of ${info.gid}: region ${info.prev} → ${info.region} (epoch ${info.epoch})`, "good"];
  if (!txt) return;
  const li = document.createElement("li");
  li.innerHTML = `<span class="t">${t.toFixed(1)}s</span><span class="${cls}">${txt}</span>`;
  const ol = $("events"); ol.prepend(li);
  while (ol.children.length > 200) ol.lastChild.remove();
  if ((TOAST.has(kind) || kind === "accept") && (S.follow || !S.live) && !S.noToast) toast(txt, cls);
}
function toast(txt, cls) {
  const d = document.createElement("div"); d.textContent = txt; if (cls) d.className = cls;
  $("toast").append(d); setTimeout(() => d.remove(), 6000);
  while ($("toast").children.length > 4) $("toast").firstChild.remove();
}

// ------------------------------------------------------------------ active view (distributed / centralized twin)
const cent = () => S.mode === "cent";
const viewOf = (f) => cent() ? (f.twin ? f.twin.view : []) : f.view;
const vmapOf = (f) => cent() ? (f.tvmap || new Map()) : f.vmap;
const gtOf = (f) => cent() ? (f.twin ? f.twin.gt : []) : f.gt;
function setMode(m) {
  S.mode = m; S.sel = null;
  $("mDist").classList.toggle("on", m === "dist"); $("mCent").classList.toggle("on", m === "cent");
  $("hint").innerHTML = m === "cent"
    ? "Centralized twin: the same city fused by <b>one node</b> · gray dots are its vehicles, colored dots its published view · faults are not injected here · scroll to zoom, drag to pan, double-click to fit"
    : "Click a <b>★ center</b> to crash it · click a camera to select · click a track to follow its ID · shift-click empty ground to add a camera · scroll to zoom, drag to pan, double-click to fit";
  send({ cmd: "twin_view", on: m === "cent" });
  renderAccuracy();
  schedule();
}

// ------------------------------------------------------------------ drawing
let pending = false;
function schedule() { if (!pending) { pending = true; requestAnimationFrame(() => { pending = false; draw(); }); } }

function territory(f) {
  const cams = allCams(); const dead = new Set(f.dead);
  const key = cams.length + ":" + f.dead.join(",");
  if (key === S.terrKey && S.terr) return S.terr;
  const n = cams.length > 500 ? 400 : 160, L = S.header.L;   // finer for big cities (zoom)
  const c = document.createElement("canvas"); c.width = n; c.height = n;
  const g = c.getContext("2d"); const img = g.createImageData(n, n);
  const live = cams.map((c, i) => [c, i]).filter(([c, i]) => c && !dead.has(i));
  // bucket grid for nearest-camera search
  const B = 16, cell = L / B, grid = Array.from({ length: B * B }, () => []);
  for (const [c] of live) grid[Math.min(B - 1, (c[1] / cell) | 0) * B + Math.min(B - 1, (c[0] / cell) | 0)].push(c);
  const rgb = {};
  for (let py = 0; py < n; py++) for (let px = 0; px < n; px++) {
    const x = (px + 0.5) / n * L, y = (1 - (py + 0.5) / n) * L;
    const bx = Math.min(B - 1, (x / cell) | 0), by = Math.min(B - 1, (y / cell) | 0);
    let best = 1e18, reg = -1;
    for (let rad = 0; rad < B && (reg < 0 || rad < 2 + 1); rad++) {
      for (let j = Math.max(0, by - rad); j <= Math.min(B - 1, by + rad); j++)
        for (let i = Math.max(0, bx - rad); i <= Math.min(B - 1, bx + rad); i++) {
          if (Math.max(Math.abs(i - bx), Math.abs(j - by)) !== rad) continue;
          for (const c of grid[j * B + i]) { const d = (c[0] - x) ** 2 + (c[1] - y) ** 2; if (d < best) { best = d; reg = c[2]; } }
        }
    }
    if (!(reg in rgb)) { const h = (reg * 137.508 + 20) % 360; rgb[reg] = hsl2rgb(h / 360, 0.55, 0.6); }
    const o = (py * n + px) * 4, col = rgb[reg];
    img.data[o] = col[0]; img.data[o + 1] = col[1]; img.data[o + 2] = col[2]; img.data[o + 3] = reg < 0 ? 0 : 46;
  }
  g.putImageData(img, 0, 0);
  S.terr = c; S.terrKey = key;
  return c;
}
function hsl2rgb(h, s, l) {
  const k = (n) => (n + h * 12) % 12, a = s * Math.min(l, 1 - l);
  const f = (n) => l - a * Math.max(-1, Math.min(k(n) - 3, Math.min(9 - k(n), 1)));
  return [255 * f(0), 255 * f(8), 255 * f(4)];
}
function star(x, y, r, fill, stroke) {
  ctx.beginPath();
  for (let i = 0; i < 10; i++) {
    const a = -Math.PI / 2 + i * Math.PI / 5, rr = i % 2 ? r * 0.45 : r;
    ctx.lineTo(x + rr * Math.cos(a), y + rr * Math.sin(a));
  }
  ctx.closePath(); ctx.fillStyle = fill; ctx.fill(); ctx.lineWidth = 1.5 * S.dpr; ctx.strokeStyle = stroke; ctx.stroke();
}

function draw() {
  ctx.clearRect(0, 0, map.width, map.height);
  if (!S.header || S.cur < 0) return;
  const bx = Math.max(X(0), 0), by = Math.max(Y(S.header.L), 0);   // banners stay on screen when zoomed
  const f = S.frames[S.cur], d = S.dpr, L = S.header.L, r = S.header.r;
  const ink = css("--ink"), muted = css("--muted");
  $("simtime").textContent = cent() && f.twin ? `t = ${f.twin.now.toFixed(1)} s (centralized twin)` : `t = ${f.t.toFixed(1)} s`;
  // world border
  ctx.strokeStyle = css("--line"); ctx.lineWidth = 1 * d; ctx.strokeRect(X(0), Y(L), L * view.s, L * view.s);
  if ($("tTerr").checked && !cent()) {
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(territory(f), X(0), Y(L), L * view.s, L * view.s);
  }
  const cams = allCams(), dead = new Set(f.dead);
  const voters = new Set(Object.values(f.voters || {}).flat());
  const centers = cent() ? new Map(f.twin && f.twin.center != null ? [[f.twin.center, 0]] : [])
                         : new Map(Object.entries(f.centers).map(([reg, id]) => [id, +reg]));
  const camColor = (c, a) => cent() ? `rgba(120,120,120,${a})` : regionColor(c[2], a);
  if ($("tCams").checked) {
    for (let i = 0; i < cams.length; i++) {
      const c = cams[i]; if (!c) continue;
      ctx.beginPath(); ctx.arc(X(c[0]), Y(c[1]), r * view.s, 0, 2 * Math.PI);
      if (dead.has(i) && !cent()) { ctx.setLineDash([3 * d, 3 * d]); ctx.strokeStyle = muted; ctx.lineWidth = 1 * d; ctx.stroke(); ctx.setLineDash([]); }
      else { ctx.fillStyle = camColor(c, 0.07); ctx.fill(); ctx.strokeStyle = camColor(c, 0.35); ctx.lineWidth = 1 * d; ctx.stroke(); }
    }
  }
  for (let i = 0; i < cams.length; i++) {
    const c = cams[i]; if (!c) continue; const x = X(c[0]), y = Y(c[1]);
    if (dead.has(i) && !cent()) {
      ctx.strokeStyle = css("--danger"); ctx.lineWidth = 2 * d; const s = 4 * d;
      ctx.beginPath(); ctx.moveTo(x - s, y - s); ctx.lineTo(x + s, y + s); ctx.moveTo(x + s, y - s); ctx.lineTo(x - s, y + s); ctx.stroke();
    } else if (centers.has(i)) {
      star(x, y, cent() ? 13 * d : 9 * d, cent() ? "#eb6834" : regionColor(c[2], 1), ink);
    } else {
      ctx.fillStyle = camColor(c, 0.9); ctx.fillRect(x - 2.5 * d, y - 2.5 * d, 5 * d, 5 * d);
      if (voters.has(i) && !cent()) { ctx.beginPath(); ctx.arc(x, y, 5.5 * d, 0, 2 * Math.PI); ctx.strokeStyle = regionColor(c[2], 1); ctx.lineWidth = 1.5 * d; ctx.stroke(); }
    }
    if (S.sel && S.sel.type === "cam" && S.sel.id === i) {
      ctx.beginPath(); ctx.arc(x, y, 13 * d, 0, 2 * Math.PI); ctx.strokeStyle = css("--accent"); ctx.lineWidth = 2 * d; ctx.stroke();
    }
  }
  // selected region's own view (owned = solid square, shadow = dashed)
  if ($("tShadow").checked && !cent() && S.sel && (S.sel.type === "cam" || S.sel.type === "region")) {
    const reg = S.sel.type === "region" ? S.sel.r : (cams[S.sel.id] || [])[2];
    for (const t of (f.regions || {})[reg] || []) {
      const x = X(t[1]), y = Y(t[2]), s = 7 * d;
      ctx.setLineDash(t[3] ? [] : [2 * d, 2 * d]); ctx.strokeStyle = regionColor(reg, 1); ctx.lineWidth = 1.5 * d;
      ctx.strokeRect(x - s, y - s, 2 * s, 2 * s); ctx.setLineDash([]);
    }
  }
  if ($("tGt").checked || cent()) {          // truth is always shown for the twin so its lag is visible
    ctx.fillStyle = muted;
    for (const g of gtOf(f)) { ctx.beginPath(); ctx.arc(X(g[1]), Y(g[2]), 2 * d, 0, 2 * Math.PI); ctx.fill(); }
  }
  // trails
  if ($("tTrail").checked) {
    const n0 = Math.max(0, S.cur - 30);
    for (const v of viewOf(f)) {
      ctx.beginPath(); let started = false;
      for (let i = n0; i <= S.cur; i++) {
        const p = vmapOf(S.frames[i]).get(v[0]); if (!p) { started = false; continue; }
        if (!started) { ctx.moveTo(X(p[1]), Y(p[2])); started = true; } else ctx.lineTo(X(p[1]), Y(p[2]));
      }
      ctx.strokeStyle = gidColor(v[0]); ctx.globalAlpha = 0.45; ctx.lineWidth = 1.5 * d; ctx.stroke(); ctx.globalAlpha = 1;
    }
  }
  // published global view (distributed aggregator, or the centralized node)
  ctx.font = `${10.5 * d}px ui-monospace, monospace`;
  for (const v of viewOf(f)) {
    const x = X(v[1]), y = Y(v[2]), col = gidColor(v[0]);
    ctx.beginPath(); ctx.arc(x, y, 4.5 * d, 0, 2 * Math.PI);
    if (v[4]) { ctx.strokeStyle = col; ctx.lineWidth = 1.5 * d; ctx.setLineDash([2 * d, 2 * d]); ctx.stroke(); ctx.setLineDash([]); }
    else { ctx.fillStyle = col; ctx.fill(); }
    const selected = S.sel && S.sel.type === "gid" && S.sel.gid === v[0];
    if (selected) { ctx.beginPath(); ctx.arc(x, y, 11 * d, 0, 2 * Math.PI); ctx.strokeStyle = css("--accent"); ctx.lineWidth = 2.5 * d; ctx.stroke(); }
    if ($("tLabels").checked || selected) { ctx.fillStyle = selected ? css("--accent") : ink; ctx.fillText(v[0], x + 6 * d, y - 6 * d); }
  }
  if (cent()) {
    ctx.font = `600 ${13 * d}px system-ui`;
    if (!f.twin) { ctx.fillStyle = muted; ctx.fillText("Waiting for the centralized twin…", bx + 10 * d, by + 22 * d); }
    else {
      const lag = f.twin.view_t == null ? Infinity : f.twin.gt_t - f.twin.view_t;
      ctx.fillStyle = lag > 1 ? css("--danger") : ink;
      ctx.fillText(`Centralized: one node fuses all ${allCams().length} cameras · published view is ` +
        (isFinite(lag) ? `${Math.max(0, lag).toFixed(1)} s behind the vehicles (gray dots)` : "not yet available"), bx + 10 * d, by + 22 * d);
    }
  } else if (!f.agg_alive) {
    ctx.fillStyle = css("--danger"); ctx.font = `600 ${14 * d}px system-ui`;
    ctx.fillText("Aggregator down — global view frozen; regions keep tracking", bx + 10 * d, by + 22 * d);
  }
  $("btnAgg").textContent = f.agg_alive ? "Kill aggregator" : "Recover aggregator";
  $("btnAgg").classList.toggle("on", !f.agg_alive);
  drawSel(f); drawTimeline();
}

// ------------------------------------------------------------------ selection panel
function gidHistory(gid) {
  const out = []; let last = null;
  for (let i = 0; i <= S.cur; i++) {
    const v = vmapOf(S.frames[i]).get(gid);
    const reg = v ? v[3] : null;
    if (reg !== last) { out.push([S.frames[i].t, reg]); last = reg; }
  }
  return out;
}
// The panel is redrawn with every frame (~10 Hz live). Touch the DOM only when its content
// changes: replacing a button between mousedown and mouseup swallows the click.
function setHTML(el, html) { if (el._html !== html) { el._html = html; el.innerHTML = html; } }
$("sel").addEventListener("click", (ev) => {
  const b = ev.target.closest("button[data-cmd]");
  if (b) send({ cmd: b.dataset.cmd, node: +b.dataset.node });
});
function drawSel(f) {
  const el = $("sel");
  if (!S.sel) { setHTML(el, "Nothing selected."); return; }
  const cams = allCams();
  if (S.sel.type === "cam") {
    const i = S.sel.id, c = cams[i], dead = f.dead.includes(i), left = (f.left || []).includes(i);
    const isCenter = Object.values(f.centers).includes(i);
    const isVoter = Object.values(f.voters || {}).flat().includes(i);
    const role = left ? "left (retired)" : dead ? "down" : isCenter ? "center (Raft leader)" : isVoter ? "Raft voter" : "member";
    setHTML(el, `<div class="row"><b>Camera ${i}</b><span>region ${c[2]}</span></div>
      <div class="row"><span>role</span><span>${role}</span></div>
      <div class="row"><span>position</span><span>(${c[0].toFixed(0)}, ${c[1].toFixed(0)})</span></div>
      <div class="btns">${left ? "" : dead ? `<button data-cmd="recover" data-node="${i}">Recover</button>` :
        `<button class="danger" data-cmd="crash" data-node="${i}">Crash</button><button data-cmd="leave" data-node="${i}">Leave</button>`}</div>`);
  } else if (S.sel.type === "gid") {
    const g = S.sel.gid, v = vmapOf(f).get(g), hist = gidHistory(g);
    const owners = hist.filter((h) => h[1] !== null).map((h) => h[1]);
    const handoffs = owners.filter((r, j) => j > 0 && r !== owners[j - 1]).length;
    setHTML(el, `<div class="row"><b style="color:${gidColor(g)}">ID ${g}</b><span>${v ? (v[4] ? "stale" : "live") : "not in view"}</span></div>
      <div class="row"><span>owner region</span><span>${v ? v[3] : "–"}</span></div>
      <div class="row"><span>region changes</span><span>${handoffs}</span></div>
      <div class="hist">${hist.map((h) => `${h[0].toFixed(1)}s → ${h[1] === null ? "gap" : "region " + h[1]}`).join("<br>")}</div>`);
  }
}

function drawTimeline() {
  const c = $("timeline"), g = c.getContext("2d"), d = S.dpr;
  c.width = c.clientWidth * d; c.height = 18 * d;
  g.clearRect(0, 0, c.width, c.height);
  const n = S.frames.length; if (n < 2) return;
  const col = { crash_center: css("--danger"), crash: css("--danger"), agg_crash: css("--danger"),
                recover: css("--ok"), center_up: css("--ok"), agg_rebuilt: css("--ok"), loss: css("--accent"), join: css("--ok") };
  for (let i = 0; i < n; i++) for (const e of S.frames[i].events) if (col[e[1]]) {
    g.fillStyle = col[e[1]]; g.fillRect(i / (n - 1) * (c.width - 2), 2 * d, 2 * d, 14 * d);
  }
  g.fillStyle = css("--ink"); g.fillRect(S.cur / (n - 1) * (c.width - 2), 0, 2 * d, c.height);
}

// accuracy of whichever system the map shows: the distributed aggregator, or the centralized twin
const accOf = (m) => (cent() ? m.baseline : m) || {};
const num = (v) => typeof v === "number" && isFinite(v);
function drawSpark() {
  const c = $("spark"), g = c.getContext("2d"), d = S.dpr;
  c.width = c.clientWidth * d; c.height = 60 * d; g.clearRect(0, 0, c.width, c.height);
  const m = S.metrics.slice(-120).map(accOf); if (m.length < 2) return;
  const line = (key, color) => {
    g.beginPath(); let started = false;
    m.forEach((p, i) => {
      if (!num(p[key])) { started = false; return; }
      const x = i / (m.length - 1) * c.width;
      const y = (1 - Math.min(1, Math.max(0, (p[key] - 0.5) / 0.5))) * (c.height - 4 * d) + 2 * d;
      started ? g.lineTo(x, y) : g.moveTo(x, y); started = true;
    });
    g.strokeStyle = color; g.lineWidth = 1.5 * d; g.stroke();
  };
  line("mota", css("--accent")); line("idf1", css("--ok"));
  g.font = `${10 * d}px system-ui`;
  g.fillStyle = css("--accent"); g.fillText("MOTA", 4 * d, 12 * d); g.fillStyle = css("--ok"); g.fillText("IDF1", 40 * d, 12 * d);
  g.fillStyle = css("--muted"); g.fillText("0.5–1.0", c.width - 44 * d, 12 * d);
}
function renderAccuracy() {
  $("accWho").textContent = cent() ? "centralized" : "distributed";
  $("accWho").className = "who " + (cent() ? "c" : "d");
  const m = S.metrics[S.metrics.length - 1];
  const a = m ? accOf(m) : {};
  $("kMota").textContent = num(a.mota) ? a.mota.toFixed(3) : "–";
  $("kIdf1").textContent = num(a.idf1) ? a.idf1.toFixed(3) : "–";
  $("kSw").textContent = num(a.idsw) ? a.idsw : "–";
  $("kLat").textContent = num(a.lat_p50) ? (a.lat_p50 >= 1 ? a.lat_p50.toFixed(1) + " s" : (a.lat_p50 * 1000).toFixed(0) + " ms") : "–";
  drawSpark();
}
function onMetrics(m) {
  S.metrics.push(m);
  if (S.metrics.length > 600) S.metrics.splice(0, 300);
  renderAccuracy();
  drawLoad(m);
}

// ------------------------------------------------------------------ compute-load comparison
const pct = (x) => (x * 100).toFixed(x < 0.1 ? 1 : 0) + "%";
const kbs = (x) => x >= 1e6 ? (x / 1e6).toFixed(1) + " MB/s" : (x / 1e3).toFixed(0) + " KB/s";
const ms = (x) => isFinite(x) ? (x >= 1 ? x.toFixed(1) + " s" : (x * 1000).toFixed(0) + " ms") : "–";
function setBar(id, v) {
  const el = $(id); el.style.width = Math.min(100, v * 100) + "%"; el.classList.toggle("hot", v > 0.9);
}
function drawLoad(m) {
  const L = m.load, B = m.baseline;
  if (L) {
    setBar("bDist", L.busiest); $("vDist").textContent = pct(L.busiest);
    $("iDist").textContent = kbs(L.center_in_max);
    const regs = Object.entries(L.centers).sort((a, b) => +a[0] - +b[0]);
    const mx = Math.max(0.05, ...regs.map((r) => r[1]), L.agg);
    $("regBars").innerHTML = regs.map(([r, v]) =>
      `<div title="region ${r}: ${pct(v)}" style="height:${Math.max(2, v / mx * 100)}%;background:${regionColor(+r, 0.85)}"><span>${r}</span></div>`).join("") +
      `<div title="aggregator: ${pct(L.agg)}" style="height:${Math.max(2, L.agg / mx * 100)}%;background:var(--muted)"><span>agg</span></div>`;
  }
  $("lDist").textContent = ms(m.lat_p50); $("fDist").textContent = m.idf1.toFixed(3);
  if (B) {
    if (B.cpu !== undefined) { setBar("bCent", B.cpu); $("vCent").textContent = pct(B.cpu); }
    if (B.in !== undefined) $("iCent").textContent = kbs(B.in);
    $("lCent").textContent = ms(B.lat_p50); $("fCent").textContent = B.idf1 !== undefined ? B.idf1.toFixed(3) : "–";
    const lag = B.lag !== undefined ? B.lag : (B.saturated ? Infinity : 0);
    $("satNote").textContent = lag > 1 ? `Centralized node saturated: its view trails reality by ${isFinite(lag) ? lag.toFixed(1) + " s" : "the whole run"}` : "";
  }
}

// ------------------------------------------------------------------ interaction
function send(o) { if (S.ws && S.ws.readyState === 1) S.ws.send(JSON.stringify(o)); else toast("Commands need live mode", "bad"); }
// ---- zoom (wheel, around the cursor) and pan (drag); double-click or "Fit" resets
const toCanvas = (ev) => {
  const rect = map.getBoundingClientRect();
  return [(ev.clientX - rect.left) * map.width / rect.width, (ev.clientY - rect.top) * map.height / rect.height];
};
map.addEventListener("wheel", (ev) => {
  if (!S.header) return;
  ev.preventDefault();
  const [px, py] = toCanvas(ev);
  const step = ev.deltaMode === 1 ? ev.deltaY * 16 : ev.deltaY;          // lines -> pixels
  const k = Math.min(ZMAX, Math.max(1, zoom.k * Math.exp(-step * 0.0015)));
  const a = k / zoom.k;
  zoom = { k, x: px - (px - zoom.x) * a, y: py - (py - zoom.y) * a };
  if (k === 1) zoom = { k: 1, x: 0, y: 0 };
  applyZoom(); schedule();
}, { passive: false });
let drag = null, dragged = false;
map.addEventListener("pointerdown", (ev) => {
  if (ev.button !== 0) return;
  drag = { x: ev.clientX, y: ev.clientY, zx: zoom.x, zy: zoom.y }; dragged = false;
});
map.addEventListener("pointermove", (ev) => {
  if (!drag) return;
  const dx = ev.clientX - drag.x, dy = ev.clientY - drag.y;
  if (!dragged && Math.hypot(dx, dy) < 4) return;                       // still a click
  if (!dragged) { dragged = true; map.setPointerCapture(ev.pointerId); map.classList.add("panning"); }
  const rect = map.getBoundingClientRect(), sx = map.width / rect.width;
  zoom = { k: zoom.k, x: drag.zx + dx * sx, y: drag.zy + dy * sx };
  applyZoom(); schedule();
});
const endDrag = () => { drag = null; map.classList.remove("panning"); };
map.addEventListener("pointerup", endDrag);
map.addEventListener("pointercancel", endDrag);
map.addEventListener("dblclick", (ev) => { ev.preventDefault(); resetZoom(); });
$("btnFit").addEventListener("click", resetZoom);

map.addEventListener("click", (ev) => {
  if (dragged) { dragged = false; return; }                             // end of a pan, not a click
  if (!S.header || S.cur < 0) return;
  // map CSS pixels to canvas pixels with the canvas's actual scale (robust even if the buffer is stale)
  const rect = map.getBoundingClientRect();
  const px = (ev.clientX - rect.left) * map.width / rect.width, py = (ev.clientY - rect.top) * map.height / rect.height;
  const f = S.frames[S.cur], cams = allCams(), dead = new Set(f.dead);
  const near = (x, y, rad) => Math.hypot(X(x) - px, Y(y) - py) <= rad * S.dpr;
  if (cent()) {
    for (const v of viewOf(f)) if (near(v[1], v[2], 9)) { S.sel = { type: "gid", gid: v[0] }; schedule(); return; }
    S.sel = null; schedule(); return;
  }
  for (const [reg, id] of Object.entries(f.centers)) {
    const c = cams[id];
    if (c && near(c[0], c[1], 11)) {
      if (S.live && S.follow) { send({ cmd: "crash_center", region: +reg }); toast(`Crashing center of region ${reg}…`, "bad"); }
      S.sel = { type: "cam", id }; schedule(); return;
    }
  }
  for (const v of f.view) if (near(v[1], v[2], 9)) { S.sel = { type: "gid", gid: v[0] }; schedule(); return; }
  for (let i = 0; i < cams.length; i++) if (cams[i] && near(cams[i][0], cams[i][1], 8)) { S.sel = { type: "cam", id: i }; schedule(); return; }
  if (ev.shiftKey && S.live) { send({ cmd: "join", x: invX(px), y: invY(py) }); return; }
  S.sel = null; schedule();
});
function rebuildEvents(upto) {
  // replay: show only the events up to the cursor
  $("events").innerHTML = ""; S.aliases = new Map();
  S.noToast = true;
  for (let i = Math.max(0, upto - 900); i <= upto; i++) for (const e of S.frames[i].events) handleEvent(e, S.frames[i]);
  S.noToast = false; S.lastSeenEvent = upto;
}
$("scrub").addEventListener("input", (e) => {
  S.cur = +e.target.value; S.follow = S.cur === S.frames.length - 1;
  if (!S.live) rebuildEvents(S.cur);
  schedule();
});
$("btnLive").onclick = () => { S.follow = true; S.cur = S.frames.length - 1; $("scrub").value = S.cur; schedule(); };
$("btnPause").onclick = () => send({ cmd: S.paused ? "resume" : "pause" });
$("speed").onchange = (e) => send({ cmd: "speed", v: +e.target.value });
$("loss").oninput = (e) => { $("lossv").textContent = Math.round(e.target.value * 100) + "%"; };
$("loss").onchange = (e) => send({ cmd: "loss", p: +e.target.value });
$("btnAgg").onclick = () => { const f = S.frames[S.cur]; send({ cmd: f && !f.agg_alive ? "recover_agg" : "crash_agg" }); };
$("btnReset").onclick = () => send({ cmd: "reset" });
S.mode = "dist";
$("mDist").onclick = () => setMode("dist");
$("mCent").onclick = () => setMode("cent");
for (const id of ["tTerr", "tCams", "tGt", "tTrail", "tLabels", "tShadow"]) $(id).onchange = schedule;
$("tShadow").onchange = () => { send({ cmd: "regions", on: $("tShadow").checked }); schedule(); };
document.addEventListener("keydown", (e) => { if (e.code === "Space" && e.target === document.body) { e.preventDefault(); $("btnPause").click(); } });
$("file").onchange = async (e) => { const f = e.target.files[0]; if (f) loadTrace(await f.text()); };

function loadTrace(text) {
  if (S.ws) { S.ws.close(); S.ws = null; }
  S.live = false; $("mode").textContent = "replay";
  const lines = text.split("\n").filter((l) => l.trim());
  onHeader(JSON.parse(lines[0]));
  S.follow = false; S.loading = true;
  for (let i = 1; i < lines.length; i++) onFrame(JSON.parse(lines[i]));
  S.loading = false;
  $("loadCard").style.display = "none";
  $("modeSw").style.display = "none";
  document.querySelector(".kpis").innerHTML = "<div style='grid-column:1/-1;color:var(--muted)'>live mode only — run the server without --replay</div>";
  S.cur = 0; $("scrub").value = 0; rebuildEvents(0); schedule();
  let playing = true;
  $("btnPause").textContent = "Pause";
  $("btnPause").onclick = () => { playing = !playing; $("btnPause").textContent = playing ? "Pause" : "Play"; };
  const tick = () => {
    if (playing && S.cur < S.frames.length - 1) {
      S.cur++; $("scrub").value = S.cur;
      for (const e of S.frames[S.cur].events) handleEvent(e, S.frames[S.cur]);
      S.lastSeenEvent = S.cur; schedule();
    }
    setTimeout(tick, 100 / (+$("speed").value));
  };
  tick();
}

function connect() {
  const ws = new WebSocket((location.protocol === "https:" ? "wss://" : "ws://") + location.host + "/ws");
  S.ws = ws; S.live = true;
  ws.onmessage = (ev) => {
    const m = JSON.parse(ev.data);
    if (m.type === "header") {
      onHeader(m);
      if (S.mode === "cent") send({ cmd: "twin_view", on: true });
      if ($("tShadow").checked) send({ cmd: "regions", on: true });
    }
    else if (m.type === "frame") onFrame(m);
    else if (m.type === "metrics") onMetrics(m);
    else if (m.type === "status") {
      S.paused = m.paused; $("btnPause").textContent = m.paused ? "Resume" : "Pause";
      $("loss").value = m.loss; $("lossv").textContent = Math.round(m.loss * 100) + "%";
    } else if (m.type === "error") toast(m.msg, "bad");
  };
  ws.onclose = () => { if (S.live) { $("mode").textContent = "disconnected"; setTimeout(connect, 2000); } };
}

window.addEventListener("resize", resize);
// the map area also changes size without a window resize (header wrapping, side panel, fonts)
if (window.ResizeObserver) new ResizeObserver(resize).observe($("mapwrap"));
(async () => {
  resize();
  try {
    const mode = await (await fetch("/mode")).json();
    if (mode.mode === "replay") { loadTrace(await (await fetch("/trace")).text()); return; }
  } catch (e) { /* opened without server */ }
  connect();
})();
