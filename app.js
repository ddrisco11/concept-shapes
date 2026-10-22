import { analyze, decodeF16 } from "./analysis.js";

const $ = (id) => document.getElementById(id);
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const MAX_SENTENCES = 4000;

const state = { ds: null, layer: 12, result: null, profile: null, busy: false, stop: false, token: 0 };

// ------------------------------------------------------------------ worker
let worker = null, nextId = 1;
const pending = new Map();
function getWorker() {
  if (worker) return worker;
  worker = new Worker("worker.js?v=3", { type: "module" });
  worker.onmessage = ({ data }) => {
    const job = pending.get(data.id);
    if (data.type === "status") setStatus(data.text);
    else if (data.type === "progress") { setBar(data.done / data.total); setStatus(`Embedding ${data.done} of ${data.total} sentences`); }
    else if (data.type === "dims") Object.assign(job.extra, { L: data.L, D: data.D });
    else if (data.type === "embedded") { pending.delete(data.id); job.resolve({ data: data.data, ...job.extra }); }
    else if (data.type === "written") { pending.delete(data.id); job.resolve(data.text); }
    else if (data.type === "error") { pending.delete(data.id); job.reject(new Error(data.message)); }
  };
  worker.onerror = (e) => { for (const j of pending.values()) j.reject(new Error(e.message || "The model worker failed to start.")); pending.clear(); };
  return worker;
}
const call = (msg) => new Promise((resolve, reject) => {
  const id = nextId++;
  pending.set(id, { resolve, reject, extra: {} });
  getWorker().postMessage({ id, ...msg });
});

// ------------------------------------------------------------------ status
function setStatus(text, err = false) { $("status").textContent = text; $("status").classList.toggle("err", err); }
function setBar(frac) { $("bar").hidden = frac === null; if (frac !== null) $("bar").firstElementChild.style.width = `${Math.round(frac * 100)}%`; }

// ------------------------------------------------------------------ inputs
export function parseValues(text) {
  const out = new Set(), limit = Math.floor(MAX_SENTENCES / 4);
  const add = (value) => {
    out.add(value);
    if (out.size > limit) throw new Error(`That is too many values for one run (limit ${limit}).`);
  };
  for (const raw of text.split(/[\n,]+/)) {
    const s = raw.trim();
    if (!s) continue;
    const m = s.match(/^(-?\d+)\s*\.\.\s*(-?\d+)(?:\s*:\s*(\d+))?$/);
    if (m) {
      const a = +m[1], b = +m[2], step = Math.max(1, +(m[3] ?? 1)) * (b >= a ? 1 : -1);
      if (![a, b, step].every(Number.isSafeInteger)) throw new Error("Range endpoints and steps must be safe integers.");
      const count = Math.floor(Math.abs(b - a) / Math.abs(step)) + 1;
      if (count > limit) throw new Error(`That is too many values for one run (limit ${limit}).`);
      for (let i = 0, v = a; i < count; i++, v += step) add(String(v));
    } else add(s);
  }
  return [...out];
}

const esc = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
const wordRe = (s, flags = "g") => new RegExp(`(?<![\\w'-])${esc(s)}(?![\\w'’-])`, flags);

// Turn a model-written line into a template, or reject it.
export function toTemplate(line, anchor, values) {
  const s = line.trim().replace(/\*\*/g, "").replace(/^(?:[-*•]|\d+[.)])\s+/, "").trim().replace(/^"|"$/g, "");
  const words = s.split(/\s+/).length;
  if (words < 5 || words > 28 || !/[.!?]$/.test(s)) return null;
  if ((s.match(wordRe(anchor)) || []).length !== 1) return null;
  const rest = s.replace(wordRe(anchor), " ");
  if (rest.includes(anchor)) return null;
  if (values.every((v) => /^-?\d+$/.test(v)) && /\d/.test(rest)) return null;
  for (const v of values) if (v !== anchor && wordRe(v, "i").test(rest)) return null;
  return s.replace(wordRe(anchor), "{X}");
}

const BANK = [
  "They kept coming back to {X} for the rest of the evening.", "The first thing she wrote down was {X}.",
  "Nobody expected {X} to come up in the meeting.", "I had never thought much about {X} until then.",
  "The article was mostly about {X}.", "He circled {X} on the page and passed it back.",
  "Everything in the report pointed to {X}.", "Ask anyone in town and they will say {X}.",
  "The answer, it turned out, was {X}.", "We agreed on {X} after a long discussion.",
  "My grandmother always mentioned {X} when she told that story.", "The sign by the door simply read {X}.",
  "For some reason {X} stuck in my mind all week.", "The teacher asked the class what they knew about {X}.",
  "It was {X} that made the difference in the end.", "The label on the box said {X}.",
  "She searched the archive for anything related to {X}.", "The whole chapter is devoted to {X}.",
  "When the form asked, he filled in {X}.", "Their notes mention {X} more than once.",
  "The guide pointed at the map and said {X}.", "Of all the options, {X} seemed the most likely.",
  "The message contained a single word: {X}.", "You could tell from the way they said {X}.",
  "The record for that entry lists {X}.", "After a pause, the witness answered {X}.",
  "The survey result that surprised us most was {X}.", "On the back of the photograph someone had written {X}.",
  "The children were asked to draw something about {X}.", "In the final version, the editor changed it to {X}.",
  "The receipt was marked {X} in blue ink.", "Her favourite part of the lecture was about {X}.",
  "The password hint just said {X}.", "The old ledger had an entry for {X}.",
  "He remembered {X} long after he forgot the rest.", "The crowd started chanting {X}.",
  "The second column of the table shows {X}.", "It all started with {X}.",
  "The radio host kept returning to {X}.", "According to the note on the fridge, it was {X}.",
];

const TOPICS = ["a bridge", "a wedding", "a factory", "a newspaper article", "a garden", "a court trial", "a school", "a shop opening",
  "a voyage at sea", "a family moving house", "a museum exhibit", "a football match", "a storm", "a letter from a friend", "a railway station",
  "a hospital", "a village festival", "an old photograph", "a library", "a farm at harvest", "a city election", "a mountain hike", "a kitchen", "a concert"];

function prompts(topic, kind, anchor, i) {
  if (i % 2) return `Every sentence below contains "${anchor}".\n\nExamples:\nEveryone at the table agreed that ${anchor} was the right answer.\nShe wrote "${anchor}" at the top of the page and underlined it.\nIt took a while, but in the end ${anchor} turned out to matter.\n\nNow write 10 new short sentences about ${topic}. Each one must contain "${anchor}" exactly once. One sentence per line, no numbering.`;
  return `Write 10 different short sentences about ${topic}. Every sentence must mention the ${kind || "word"} ${anchor}. Use a different sentence structure each time. One sentence per line, no numbering.`;
}

async function gemmaTemplates(kind, values, want) {
  const anchor = values[Math.floor(values.length / 2)];
  const found = new Map(), maxCalls = Math.max(10, want * 3), start = Math.floor(Math.random() * TOPICS.length);
  for (let i = 0; i < maxCalls && found.size < want && !state.stop; i++) {
    setBar(found.size / want);
    setStatus(`Gemma is writing sentences around “${anchor}”: ${found.size} of ${want} kept (request ${i + 1})`);
    const text = await call({ type: "write", prompt: prompts(TOPICS[(start + i) % TOPICS.length], kind, anchor, i) });
    for (const line of text.split("\n")) {
      const t = toTemplate(line, anchor, values);
      if (t) found.set(t.toLowerCase().replace(/\W+/g, " "), t);
    }
  }
  const out = [...found.values()].slice(0, want);
  const fromGemma = out.length;
  for (const b of BANK) { if (out.length >= want) break; out.push(b); }
  return { templates: out, fromGemma };
}

// ------------------------------------------------------------------ build
async function build() {
  if (state.busy) return;
  let values;
  try { values = parseValues($("values").value); }
  catch (e) { return setStatus(e.message, true); }
  const kind = $("kind").value.trim();
  const source = document.querySelector('input[name="source"]:checked').value;
  if (values.length < 3) return setStatus("Enter at least three values.", true);
  let want = Math.max(4, Math.min(60, +$("count").value || 16));
  let capped = false;
  if (want * values.length > MAX_SENTENCES) { want = Math.max(4, Math.floor(MAX_SENTENCES / values.length)); capped = true; }
  if (want * values.length > MAX_SENTENCES) return setStatus(`That is too many values for one run (limit ${Math.floor(MAX_SENTENCES / 4)}).`, true);

  state.busy = true; state.stop = false;
  $("build").disabled = true; $("stop").hidden = false; setBar(0);
  try {
    let templates, note = "";
    if (source === "own") {
      templates = $("own").value.split("\n").map((s) => s.trim()).filter((s) => s.includes("{X}")).slice(0, want);
      if (templates.length < 2) throw new Error("Add at least two templates that contain {X}.");
    } else if (source === "bank") {
      templates = BANK.slice(0, want);
      note = "built-in sentences";
    } else {
      const g = await gemmaTemplates(kind, values, want);
      templates = g.templates;
      note = g.fromGemma === templates.length ? "sentences by Gemma 3 270M" : `${g.fromGemma} sentences by Gemma 3 270M, ${templates.length - g.fromGemma} built-in`;
    }
    if (source !== "own" && templates.length < want) capped = true;
    const sentences = templates.flatMap((t) => values.map((v) => t.replaceAll("{X}", v)));
    setBar(0); setStatus(`Embedding ${sentences.length} sentences…`);
    const { data, L, D } = await call({ type: "embed", sentences });
    setDataset({ name: kind ? `${kind[0].toUpperCase()}${kind.slice(1)}` : "Your concept", note, values, templates, T: templates.length, V: values.length, L, D, data }, null);
    setStatus(capped ? `Done, using ${templates.length} sentences per value to stay within ${MAX_SENTENCES} embeddings.` : "Done.");
  } catch (e) {
    setStatus(e.message || String(e), true);
  } finally {
    state.busy = false; $("build").disabled = false; $("stop").hidden = true; setBar(null);
  }
}

// ------------------------------------------------------------------ presets
async function loadPresets() {
  const list = await (await fetch("presets/index.json")).json();
  for (const p of list) {
    const b = document.createElement("button");
    b.textContent = p.name; b.dataset.id = p.id; b.setAttribute("aria-pressed", "false");
    b.onclick = () => loadPreset(p);
    $("presets").append(b);
  }
  if (list.length) loadPreset(list[0]);
}

async function loadPreset(p) {
  if (state.busy) return;
  const token = ++state.token;
  $("name").textContent = `${p.name}…`;
  const buf = await (await fetch(`presets/${p.file}`)).arrayBuffer();
  if (token !== state.token) return;
  setDataset({ ...p, data: decodeF16(buf) }, p.id);
  $("kind").value = p.kind; $("values").value = p.valuesText ?? p.values.join(", ");
}

// ------------------------------------------------------------------ render
function setDataset(ds, presetId) {
  state.ds = ds; state.profile = null; state.token++;
  for (const b of $("presets").children) b.setAttribute("aria-pressed", String(b.dataset.id === presetId));
  $("layer").max = ds.L - 1;
  state.layer = Math.min(state.layer, ds.L - 1);
  $("name").textContent = ds.name;
  $("sub").textContent = `${ds.V} values × ${ds.T} sentences${ds.note ? ` · ${ds.note}` : ""}`;
  $("tcount").textContent = ds.T;
  $("templates").replaceChildren(...ds.templates.map((t) => {
    const li = document.createElement("li");
    t.split("{X}").forEach((part, i) => { if (i) { const m = document.createElement("mark"); m.textContent = "{X}"; li.append(m); } li.append(part); });
    return li;
  }));
  $("loop").checked = false;
  render();
  computeProfile(state.token);
}

function render() {
  const ds = state.ds;
  if (!ds) return;
  state.layer = +$("layer").value;
  $("layerOut").textContent = state.layer;
  $("layerNote").textContent = state.layer === ds.L - 1 ? "final embedding" : state.layer === 0 ? "token embeddings, before any attention" : "";
  state.result = analyze(ds, state.layer);
  const { verdict } = state.result;
  $("verdict").hidden = false;
  $("verdict").firstElementChild.textContent = verdict.shape;
  $("verdict").lastElementChild.textContent = verdict.why;
  draw3d(); drawHeat(); drawScores(); drawProfile();
}

const baseLayout = () => ({
  paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
  font: { family: "ui-sans-serif, system-ui, -apple-system, sans-serif", size: 12, color: css("--ink-2") },
  margin: { l: 0, r: 0, t: 0, b: 0 }, showlegend: false,
  hoverlabel: { bgcolor: css("--surface"), bordercolor: css("--line"), font: { color: css("--ink"), size: 12 } },
});
const CONFIG = { displayModeBar: false, responsive: true };
const seqScale = () => [[0, css("--seq-lo")], [0.5, css("--seq-mid")], [1, css("--seq-hi")]];

function draw3d() {
  const ds = state.ds, { coords, sentences, kShow } = state.result;
  const off = Math.min(+$("axes").value, Math.max(0, kShow - 3));
  const ax = [0, 1, 2].map((i) => off + i);
  const col = (i) => coords.map((c) => c[ax[i]] ?? 0);
  const order = ds.values.map((_, i) => i);
  const every = Math.ceil(ds.V / 28);
  const ink = css("--ink"), grid = css("--line");
  const traces = [];
  if ($("showSent").checked) {
    const pick = (i) => Array.from({ length: ds.T * ds.V }, (_, r) => (ax[i] < kShow ? sentences[r * kShow + ax[i]] : 0));
    traces.push({
      type: "scatter3d", mode: "markers", x: pick(0), y: pick(1), z: pick(2),
      marker: { size: 2.2, opacity: 0.35, color: Array.from({ length: ds.T * ds.V }, (_, r) => r % ds.V), colorscale: seqScale(), cmin: 0, cmax: ds.V - 1 },
      text: Array.from({ length: ds.T * ds.V }, (_, r) => ds.templates[Math.floor(r / ds.V)].replaceAll("{X}", ds.values[r % ds.V])),
      hovertemplate: "%{text}<extra></extra>",
    });
  }
  if ($("loop").checked && ds.V > 2) {
    traces.push({ type: "scatter3d", mode: "lines", hoverinfo: "skip", x: [col(0)[ds.V - 1], col(0)[0]], y: [col(1)[ds.V - 1], col(1)[0]], z: [col(2)[ds.V - 1], col(2)[0]], line: { color: css("--ink-3"), width: 3, dash: "dot" } });
  }
  traces.push({
    type: "scatter3d", mode: $("join").checked ? "lines+markers+text" : "markers+text",
    x: col(0), y: col(1), z: col(2),
    text: ds.values.map((v, i) => (i % every === 0 || i === ds.V - 1 ? v : "")), textposition: "top center", textfont: { color: ink, size: 11 },
    customdata: ds.values, hovertemplate: "<b>%{customdata}</b><extra></extra>",
    line: { color: css("--ink-3"), width: 2 },
    marker: { size: ds.V > 80 ? 3.5 : 6, color: order, colorscale: seqScale(), cmin: 0, cmax: ds.V - 1, line: { color: css("--surface"), width: 1 },
      colorbar: { thickness: 10, len: 0.5, x: 1, tickvals: [0, ds.V - 1], ticktext: [ds.values[0], ds.values[ds.V - 1]], outlinewidth: 0, title: { text: "order", side: "right" } } },
  });
  const share = state.result.metrics.pcShare;
  const axis = (i) => ({ title: { text: `PC${ax[i] + 1} (${Math.round(100 * (share[ax[i]] ?? 0))}%)` }, gridcolor: grid, zerolinecolor: grid, showbackground: false, tickfont: { size: 10 }, color: css("--ink-3") });
  Plotly.react("plot3d", traces, { ...baseLayout(), uirevision: `${ds.name}-${ds.V}`, scene: { xaxis: axis(0), yaxis: axis(1), zaxis: axis(2), aspectmode: "data" } }, CONFIG);
}

function drawHeat() {
  const ds = state.ds, { sim } = state.result;
  const step = Math.ceil(ds.V / 16), tv = ds.values.map((_, i) => i).filter((i) => i % step === 0);
  const ticks = { tickmode: "array", tickvals: tv, ticktext: tv.map((i) => ds.values[i]), color: css("--ink-3"), showgrid: false, zeroline: false };
  Plotly.react("heat", [{
    type: "heatmap", z: sim, zmin: -1, zmax: 1, xgap: ds.V <= 40 ? 2 : 0, ygap: ds.V <= 40 ? 2 : 0,
    colorscale: [[0, css("--div-neg")], [0.5, css("--div-mid")], [1, css("--div-pos")]],
    customdata: ds.values.map((a) => ds.values.map((b) => [a, b])),
    hovertemplate: "%{customdata[0]} · %{customdata[1]}<br>cosine %{z:.2f}<extra></extra>",
    colorbar: { thickness: 10, len: 0.8, outlinewidth: 0, tickvals: [-1, 0, 1] },
  }], { ...baseLayout(), margin: { l: 70, r: 10, t: 6, b: 60 }, xaxis: ticks, yaxis: { ...ticks, autorange: "reversed", scaleanchor: "x" } }, CONFIG);
}

const pct = (x) => `${Math.round(100 * x)}%`;
const num = (x) => (x === null || x === undefined || Number.isNaN(x) ? "n/a" : x < -1 ? "< −1" : x.toFixed(2).replace("-", "−"));

function drawScores() {
  const m = state.result.metrics;
  const rows = [
    ["Variance in PC1 · PC2 · PC3", m.pcShare.slice(0, 3).map(pct).join(" · "), "A line puts nearly everything in PC1. A ring splits it between PC1 and PC2. A cloud spreads it thin."],
    ["Straightness", num(m.straightness), "Distance from first to last value, divided by the length of the path through every value. 1 is a straight line; near 0 is a tangle or a closed loop."],
    ["Neighbours vs the rest", `${num(m.neighbourSim)} vs ${num(m.otherSim)}`, "Average cosine between adjacent values, against all other pairs. A large gap means the order you typed is real to the model."],
    ["Last value to first", num(m.wrapSim), "Cosine between the two ends. If it is as high as the neighbour score, the shape closes into a ring."],
    ["Shared across sentences", pct(m.shared), "How much of the change caused by swapping values is the same in every sentence, rather than depending on the context."],
    ["Line fit in the top two directions", num(m.lineR2), "R² when position along a line is read from the two main directions, tested on sentences the probe never saw."],
    ["Ring fit in the top two directions", num(m.ringR2), "The same test, reading an angle around a circle instead."],
    ["Position of unseen values", num(m.unseenR2), "A 12-direction probe trained on every other value and tested on the ones in between. High only if position varies smoothly, not as a lookup table."],
  ];
  $("scores").replaceChildren(...rows.map(([k, v, why]) => {
    const d = document.createElement("div");
    d.innerHTML = "<dt></dt><dd></dd><p></p>";
    d.children[0].textContent = k; d.children[1].textContent = v; d.children[2].textContent = why;
    return d;
  }));
}

// Probe scores at every layer, computed in the background one layer per tick.
function computeProfile(token) {
  const ds = state.ds, prof = { line: [], ring: [], unseen: [] };
  let l = 0;
  const tick = () => {
    if (token !== state.token) return;
    const m = analyze(ds, l, { light: true }).metrics;
    prof.line.push(m.lineR2); prof.ring.push(m.ringR2); prof.unseen.push(m.unseenR2);
    state.profile = prof; drawProfile();
    if (++l < ds.L) setTimeout(tick, 0);
  };
  setTimeout(tick, 30);
}

function drawProfile() {
  const p = state.profile, grid = css("--line");
  const clip = (a) => a.map((v) => (v === null ? null : Math.max(-0.25, v)));
  const mk = (name, y, color, dash) => ({ type: "scatter", mode: "lines+markers", name, x: y.map((_, i) => i), y: clip(y), line: { color, width: 2, dash }, marker: { size: 6, color }, hovertemplate: `layer %{x}<br>${name}: %{y:.2f}<extra></extra>` });
  const traces = p ? [mk("line fit", p.line, css("--accent")), mk("ring fit", p.ring, css("--orange")), mk("unseen values", p.unseen, css("--ink-3"), "dot")] : [];
  Plotly.react("profile", traces, {
    ...baseLayout(), showlegend: true, legend: { orientation: "h", x: 0, y: 1.18, font: { color: css("--ink-2") } },
    margin: { l: 44, r: 10, t: 26, b: 36 },
    xaxis: { title: { text: "layer" }, dtick: 1, range: [-0.3, (state.ds?.L ?? 13) - 0.7], gridcolor: grid, zeroline: false, color: css("--ink-3") },
    yaxis: { title: { text: "R²" }, range: [-0.3, 1.05], gridcolor: grid, zerolinecolor: css("--ink-3"), color: css("--ink-3") },
    shapes: [{ type: "line", x0: state.layer, x1: state.layer, yref: "paper", y0: 0, y1: 1, line: { color: css("--ink-3"), width: 1, dash: "dot" } }],
  }, CONFIG);
  const el = $("profile");
  if (!el.dataset.bound) { el.dataset.bound = "1"; el.on("plotly_click", (ev) => { $("layer").value = ev.points[0].x; render(); }); }
}

// ------------------------------------------------------------------ wiring
function init() {
  $("build").onclick = build;
  $("stop").onclick = () => { state.stop = true; setStatus("Stopping after this request…"); };
  $("layer").oninput = render;
  for (const id of ["axes", "showSent", "join", "loop"]) $(id).onchange = draw3d;
  for (const r of document.querySelectorAll('input[name="source"]')) r.onchange = () => { $("own").hidden = r.value !== "own" || !r.checked; };
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => state.ds && render());
  loadPresets().catch((e) => { $("name").textContent = "Build a concept to begin"; setStatus(`Examples could not be loaded: ${e.message}`, true); });
}
if (typeof document !== "undefined") {
  if (window.Plotly) init(); else window.addEventListener("load", init);
}
