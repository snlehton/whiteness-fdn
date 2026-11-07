import { renderFDN, noiseReference, gammaFromT60, t60FromGamma, householder, hadamard, identity, orthogonalityError,
  powerSpectrum, echoDensity, rms, modalStats } from "./dsp.js?v=2";
import { Plot, Heatmap, fmtHz } from "./plots.js?v=2";
import { Engine, SOURCES, convolveLoop, audioContext } from "./audio.js?v=2";
// ?v=2: new URLs so browsers drop copies cached from plain http.server (scripts/serve.py now sends no-cache)

const FS = 48000;
const VARIANTS = [
  { key: "optimized", label: "DiffFDN" },
  { key: "init", label: "RO (init)" },
  { key: "householder", label: "Householder" },
  { key: "random", label: "Haar random" },
  { key: "hadamard", label: "Hadamard" },
  { key: "diagonal", label: "Diagonal (combs)" },
  { key: "custom", label: "Custom" },
  { key: "noise", label: "Noise ref" },
];
const SHORT = Object.fromEntries(VARIANTS.map((v) => [v.key, v.label]));
const LONG = { ...SHORT };
LONG.optimized = "DiffFDN (optimized)";
LONG.init = "RO (reference init, near identity)";
LONG.householder = "Householder, unit gains";
LONG.random = "Haar-random orthogonal, unit gains";
LONG.hadamard = "Hadamard, unit gains";
LONG.diagonal = "Diagonal = comb bank (anchor)";
LONG.noise = "Gaussian noise reference";

const $ = (id) => document.getElementById(id);
const css = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const color = (k) => css(`--v-${k}`);

const state = {
  index: [], run: null, fdns: {}, keys: [], sel: "optimized", cmp: "init", alt: "",
  decay: "expDE", t60: 2.5, source: "ir", irRange: 0.5, userAudio: null,
};
const cache = new Map();
const engine = new Engine();

// ----------------------------------------------------------------- plots
const plots = {
  heat: new Heatmap($("heat")),
  ir: new Plot($("ir"), { xlabel: "Time (s)", ylabel: "Amplitude", xfmt: (v) => `${+v.toFixed(3)}`, xfmtTip: (v) => `${(v * 1000).toFixed(2)} ms`, yfmt: (v) => `${+v.toPrecision(3)}`, height: 400 }),
  mag: new Plot($("mag"), { xlabel: "Frequency (Hz)", ylabel: "Magnitude (dB)", xlog: true, xfmt: fmtHz, xfmtTip: (v) => `${v.toFixed(1)} Hz`, yfmt: (v) => `${v.toFixed(1)}`, height: 240 }),
  zoom: new Plot($("magzoom"), { xlabel: "Frequency (Hz)", ylabel: "dB", xfmt: (v) => v.toFixed(0), xfmtTip: (v) => `${v.toFixed(2)} Hz`, yfmt: (v) => `${v.toFixed(1)}`, height: 180 }),
  hist: new Plot($("hist"), { xlabel: "20·log10|ρ| re mean (dB)", ylabel: "Modes / 0.5 dB", xfmt: (v) => `${v}`, xfmtTip: (v) => `${v.toFixed(2)} dB`, yfmt: (v) => v.toFixed(0), height: 220 }),
  scatter: new Plot($("scatter"), { xlabel: "Frequency (Hz)", ylabel: "dB re mean", xfmt: fmtHz, xfmtTip: (v) => `${v.toFixed(1)} Hz`, yfmt: (v) => `${v.toFixed(1)} dB`, height: 220, nearestPoint: true }),
  ned: new Plot($("ned"), { xlabel: "Time (s)", ylabel: "NED", xfmt: (v) => `${+v.toFixed(2)}`, xfmtTip: (v) => `${(v * 1000).toFixed(0)} ms`, yfmt: (v) => v.toFixed(2), height: 200 }),
  loss: new Plot($("loss"), { xlabel: "Epoch", ylabel: "Total loss", xfmt: (v) => `${v}`, xfmtTip: (v) => `epoch ${v}`, yfmt: (v) => v.toFixed(3), height: 170 }),
  losscomp: new Plot($("losscomp"), { xlabel: "Epoch (0 = init)", ylabel: "Component", xfmt: (v) => `${v}`, xfmtTip: (v) => `epoch ${v}`, yfmt: (v) => v.toFixed(3), height: 170 }),
};

function legend(el, items) {
  el.replaceChildren(...items.map(([k, text, col]) => {
    const s = document.createElement("span");
    const key = document.createElement("i");
    key.className = "key";
    key.style.background = col || color(k);
    s.append(key, document.createTextNode(text));
    return s;
  }));
}

// ------------------------------------------------------------ data model
function buildFdns(run) {
  const f = {};
  const delays = run.delays, N = delays.length;
  for (const [k, v] of Object.entries(run.variants)) f[k] = { delays, U: v.U, b: v.b, c: v.c };
  const ones = Array(N).fill(1);
  if (!f.householder) f.householder = { delays, U: householder(N), b: ones, c: ones };
  const H = hadamard(N);
  if (H) f.hadamard = { delays, U: H, b: ones, c: ones };
  f.diagonal = { delays, U: identity(N), b: ones, c: ones };
  return f;
}

function gamma() {
  if (state.decay === "LL") return 1;
  if (state.decay === "train") return state.run?.gamma_train ?? 0.9999;
  if (state.decay === "expDE") return gammaFromT60(2.5, FS);
  return gammaFromT60(state.t60, FS);
}
function irLength(g) {
  if (g >= 1) return 16 * FS;
  return Math.round(Math.min(12, 1.2 * t60FromGamma(g, FS) + 0.5) * FS);
}

function getIR(key, g = gamma(), len = irLength(g)) {
  const id = `${state.run.id}|${key}|${g}|${len}`;
  if (!cache.has(id)) {
    const h = key === "noise" ? noiseReference(g, len) : renderFDN(state.fdns[key], g, len);
    // equal energy (the paper normalized every IR to the same RMS)
    const r = rms(h);
    for (let i = 0; i < h.length; i++) h[i] /= r * 100;
    cache.set(id, h);
    if (cache.size > 40) cache.delete(cache.keys().next().value);
  }
  return cache.get(id);
}

function getSpectrum(key) {
  // decaying IR needed: lossless runs are analysed at the training gamma
  const g = gamma() >= 0.99999 ? state.run.gamma_train : gamma();
  const n = 1 << 19;
  const id = `${state.run.id}|${key}|spec|${g}`;
  if (!cache.has(id)) {
    const P = powerSpectrum(getIR(key, g, n), n);
    const cum = new Float64Array(P.length + 1);
    for (let k = 0; k < P.length; k++) cum[k + 1] = cum[k] + P[k];
    // normalise: mean power 100 Hz - 10 kHz = 0 dB
    const k0 = Math.round((100 / FS) * n), k1 = Math.round((10000 / FS) * n);
    const ref = (cum[k1] - cum[k0]) / (k1 - k0);
    const df = FS / n;
    const f = new Float64Array(P.length), db = new Float32Array(P.length);
    for (let k = 0; k < P.length; k++) { f[k] = k * df; db[k] = 10 * Math.log10(P[k] / ref + 1e-30); }
    // 1/3-octave smoothing on a log grid
    const sf = [], sdb = [];
    for (let x = Math.log(20); x <= Math.log(23000); x += 0.01) {
      const fc = Math.exp(x), a = Math.max(1, Math.floor((fc * 2 ** (-1 / 6)) / df)), b = Math.min(P.length - 1, Math.ceil((fc * 2 ** (1 / 6)) / df));
      sf.push(fc); sdb.push(10 * Math.log10((cum[b + 1] - cum[a]) / (b + 1 - a) / ref));
    }
    cache.set(id, { f, db, sf: Float64Array.from(sf), sdb: Float32Array.from(sdb), g });
  }
  return cache.get(id);
}

// ---------------------------------------------------------------- views
function renderVariantButtons() {
  const seg = $("variants");
  seg.replaceChildren();
  state.keys.forEach((k, i) => {
    const b = document.createElement("button");
    b.type = "button";
    b.setAttribute("role", "radio");
    b.setAttribute("aria-checked", String(k === state.sel));
    const key = document.createElement("i"); key.className = "key"; key.style.background = color(k);
    const num = document.createElement("span"); num.className = "num"; num.textContent = String(i + 1);
    b.append(key, document.createTextNode(SHORT[k]), num);
    b.title = LONG[k];
    b.onclick = () => select(k);
    seg.append(b);
  });
  const cmp = $("compare");
  cmp.replaceChildren(new Option("none", ""), ...state.keys.map((k) => new Option(LONG[k], k)));
  cmp.value = state.cmp || "";
}

function renderParams() {
  const k = state.sel, f = state.fdns[k];
  $("params-title").textContent = LONG[k];
  const kv = $("kv");
  const g = gamma();
  const rows = [
    ["Delays N", state.run.delays.length],
    ["Order M = Σm", state.run.delays.reduce((a, b) => a + b, 0)],
    ["Playback γ / T60", g >= 1 ? "1 (lossless) / ∞" : `${g.toFixed(6)} / ${t60FromGamma(g, FS).toFixed(2)} s`],
    ["Training γ / T60", `${state.run.gamma_train} / ${t60FromGamma(state.run.gamma_train, FS).toFixed(3)} s`],
  ];
  if (f) rows.push(["‖UUᵀ − I‖max", orthogonalityError(f.U).toExponential(1)]);
  kv.replaceChildren(...rows.flatMap(([a, b]) => { const dt = document.createElement("dt"), dd = document.createElement("dd"); dt.textContent = a; dd.textContent = String(b); return [dt, dd]; }));
  const gt = $("gains");
  gt.replaceChildren();
  if (!f) { plots.heat.set(null); $("heat").hidden = true; return; }
  $("heat").hidden = false;
  plots.heat.set(f.U, (i, j) => `U[${i + 1},${j + 1}] · line ${j + 1} → line ${i + 1}`);
  const head = gt.insertRow();
  for (const t of ["line", "m (samples)", "b", "", "c", ""]) { const th = document.createElement("th"); th.textContent = t; head.append(th); }
  const mx = Math.max(...f.b.map(Math.abs), ...f.c.map(Math.abs)) || 1;
  f.delays.forEach((m, i) => {
    const r = gt.insertRow();
    r.insertCell().textContent = String(i + 1);
    r.insertCell().textContent = String(m);
    for (const v of [f.b[i], f.c[i]]) {
      r.insertCell().textContent = v.toFixed(3);
      const cell = r.insertCell(); cell.className = "bar-cell";
      const bar = document.createElement("div"); bar.className = "gbar";
      const fill = document.createElement("i");
      const w = (Math.abs(v) / mx) * 50;
      fill.style.width = `${w}%`; fill.style.left = v >= 0 ? "50%" : `${50 - w}%`;
      fill.style.background = color(k);
      bar.append(fill); cell.append(bar);
    }
  });
}

function shown() { return [state.cmp, state.sel].filter((k, i, a) => k && a.indexOf(k) === i); }

const timeAxes = new Map();
function timeAxis(len) {
  if (!timeAxes.has(len)) timeAxes.set(len, Float64Array.from({ length: len }, (_, i) => i / FS));
  return timeAxes.get(len);
}

function renderIR() {
  const g = gamma();
  const keys = shown();
  const len = irLength(g);
  const range = state.irRange === "all" ? len : Math.min(len, Math.round(state.irRange * FS));
  let peak = 0;
  for (const k of keys) { const h = getIR(k); for (let i = 0; i < range; i++) peak = Math.max(peak, Math.abs(h[i])); }
  plots.ir.set(keys.map((k) => ({ name: LONG[k], color: color(k), x: timeAxis(len), y: getIR(k), width: 1, alpha: k === state.sel ? 1 : 0.6 })),
    { xlim: [0, range / FS], ylim: [-1.05 * peak, 1.05 * peak] });
  legend($("ir-legend"), keys.map((k) => [k, LONG[k]]).reverse());
}

function renderMag() {
  const keys = shown();
  const specs = keys.map((k) => [k, getSpectrum(k)]);
  const g = specs[0]?.[1].g;
  $("mag-hint").textContent = `· at γ = ${g} (T60 ${t60FromGamma(g, FS).toFixed(2)} s), thin = raw, thick = 1/3-octave`;
  const series = [];
  for (const [k, s] of specs) series.push({ name: `${LONG[k]} raw`, color: color(k), x: s.f, y: s.db, width: 1, alpha: 0.3, noTip: true });
  for (const [k, s] of specs) series.push({ name: LONG[k], color: color(k), x: s.sf, y: s.sdb, width: 2 });
  plots.mag.set(series, { xlim: [20, 24000], ylim: [-30, 25] });
  legend($("mag-legend"), keys.map((k) => [k, LONG[k]]).reverse());
  const f0 = Math.max(0, +$("zoomf").value || 0);
  plots.zoom.set(specs.map(([k, s]) => ({ name: LONG[k], color: color(k), x: s.f, y: s.db, width: k === state.sel ? 2 : 1.5, alpha: k === state.sel ? 1 : 0.75 })),
    { xlim: [f0, f0 + 50] });
}

function modalKeys() { return ["init", "optimized", "householder", "random", "custom"].filter((k) => state.run.modal?.[k]); }
const BASELINE_DASH = { householder: [5, 3], random: [2, 2] };   // secondary encoding for the baseline curves

function renderModal() {
  const keys = modalKeys();
  if (!keys.length) {
    plots.hist.set([]); plots.scatter.set([]);
    legend($("hist-legend"), [["", "No modal data in this file (run scripts/evaluate.py)", css("--text-muted")]]);
    legend($("scatter-legend"), []);
    return;
  }
  const edges = Float64Array.from({ length: 111 }, (_, i) => -25 + 0.5 * i);
  const hist = [], leg = [];
  for (const k of keys) {
    const st = modalStats(state.run.modal[k].db);
    const cnt = new Float64Array(edges.length - 1);
    for (const v of st.centered) { const i = Math.floor((v + 25) / 0.5); if (i >= 0 && i < cnt.length) cnt[i]++; }
    hist.push({ name: LONG[k], color: color(k), x: edges, y: cnt, kind: "stairs", width: BASELINE_DASH[k] ? 1.5 : 2, dash: BASELINE_DASH[k] || [] });
    leg.push([k, `${LONG[k]} · σ ${st.std.toFixed(1)} dB · p99 ${st.p99 >= 0 ? "+" : ""}${st.p99.toFixed(1)} dB`]);
  }
  plots.hist.set(hist, { xlim: [-25, 30] });
  legend($("hist-legend"), leg);
  // scatter: selected (and compare) if they have modal data
  const sk = shown().filter((k) => state.run.modal?.[k]);
  const sc = sk.map((k) => {
    const { angle, db } = state.run.modal[k];
    const valid = db.filter((v) => isFinite(v) && v > -250);
    const mean = valid.reduce((a, b) => a + b, 0) / valid.length;
    return { name: LONG[k], color: color(k), kind: "dots", radius: 1.3, alpha: 0.55,
      x: Float64Array.from(angle, (a) => (a / Math.PI) * FS / 2), y: Float32Array.from(db, (v) => v - mean) };
  });
  plots.scatter.set(sc, { xlim: [0, FS / 2], ylim: [-30, 25] });
  legend($("scatter-legend"), sk.map((k) => [k, LONG[k]]));
}

function renderNED() {
  const keys = [...new Set([...shown(), "noise"])];
  const series = keys.map((k) => {
    const id = `${state.run.id}|${k}|ned|${gamma()}`;
    if (!cache.has(id)) cache.set(id, echoDensity(getIR(k), FS, 1));
    const { t, d } = cache.get(id);
    return { name: LONG[k], color: color(k), x: t, y: d, width: k === "noise" ? 1.5 : 2, dash: k === "noise" ? [4, 3] : [] };
  });
  plots.ned.set(series, { ylim: [0, 1.3] });
  legend($("ned-legend"), keys.map((k) => [k, LONG[k]]));
}

function renderLoss() {
  const L = state.run.losses || {};
  if (L.train) {
    const ep = Float64Array.from(L.train, (_, i) => i + 1);
    plots.loss.set([
      { name: "train", color: css("--c-train"), x: ep, y: Float32Array.from(L.train), width: 2 },
      { name: "validation", color: css("--text-muted"), x: ep, y: Float32Array.from(L.valid), width: 2, dash: [4, 3] },
    ]);
    legend($("loss-legend"), [["", "train", css("--c-train")], ["", "validation", css("--text-muted")]]);
  } else {
    plots.loss.set([]);
    legend($("loss-legend"), [["", "No loss history in this file", css("--text-muted")]]);
  }
  const C = L.components;
  $("losscomp").hidden = !C;
  $("losscomp-legend").hidden = !C;
  if (C) {
    const ep = Float64Array.from(C.epoch, (e) => e + 1);
    plots.losscomp.set([
      { name: "L_spectral", color: css("--c-spectral"), x: ep, y: Float32Array.from(C.spectral), width: 2 },
      { name: `α·L_temporal (α=${C.alpha})`, color: css("--c-temporal"), x: ep, y: Float32Array.from(C.temporal, (v) => v * C.alpha), width: 2 },
    ]);
    legend($("losscomp-legend"), [["", "L_spectral", css("--c-spectral")], ["", `α·L_temporal (α=${C.alpha})`, css("--c-temporal")]]);
  }
}

function renderMetrics() {
  const t = $("metrics");
  t.replaceChildren();
  const M = state.run.metrics || {};
  const keys = Object.keys(M);
  if (!keys.length) { const r = t.insertRow(); r.insertCell().textContent = "No metrics in this file"; return; }
  const head = t.insertRow();
  const th0 = document.createElement("th"); th0.textContent = "Metric (at training γ)"; head.append(th0);
  for (const k of keys) {
    const th = document.createElement("th");
    const key = document.createElement("i"); key.className = "key"; key.style.background = color(k); key.style.marginRight = "6px";
    th.append(key, document.createTextNode(LONG[k] || k));
    head.append(th);
  }
  const rows = [
    ["Modes (upper half-plane)", (m) => m.n_modes],
    ["Modal excitation σ (dB)", (m) => m.std_db.toFixed(2)],
    ["99th percentile (dB re mean)", (m) => `${m.p99_db >= 0 ? "+" : ""}${m.p99_db.toFixed(2)}`],
    ["Loudest mode (dB re mean)", (m) => `${m.max_db >= 0 ? "+" : ""}${m.max_db.toFixed(2)}`],
    ["Modes > +6 dB", (m) => `${(100 * m.frac_above_6db).toFixed(1)} %`],
    ["‖h‖₂/‖h‖₁ × 10³ (temporal loss, lower = denser)", (m) => (1e3 * m.sparsity_ir).toFixed(2)],
    ["Diagonal energy ‖diag U‖²/N (1 = parallel combs)", (m) => (m.diag_energy == null ? "–" : `${m.diag_energy.toFixed(2)} (dense ≈ ${m.diag_energy_dense_ref.toFixed(2)})`)],
    ["Echo density 0–100 ms", (m) => m.ned_mean_0_100ms.toFixed(2)],
    ["Echo density 100–500 ms", (m) => m.ned_mean_100_500ms.toFixed(2)],
    ["Echo density 0.5–1 s", (m) => (m.ned_mean_500_1000ms == null ? "–" : m.ned_mean_500_1000ms.toFixed(2))],
    ["Time to NED ≥ 0.9", (m) => (m.t_ned_0p9_ms == null ? "–" : `${m.t_ned_0p9_ms.toFixed(0)} ms`)],
  ];
  for (const [name, fn] of rows) {
    const r = t.insertRow();
    r.insertCell().textContent = name;
    for (const k of keys) r.insertCell().textContent = String(fn(M[k]));
  }
}

function renderAll() {
  if (!state.run) return;
  renderParams(); renderIR(); renderMag(); renderModal(); renderNED(); renderLoss(); renderMetrics();
}

// ----------------------------------------------------------------- audio
async function buildAudio() {
  const g = gamma();
  const src = state.source;
  const buffers = new Map();
  let dry = null;
  if (src === "ir") {
    // the paper's stimuli: the IR itself (expDE) or the lossless tail from 6 s on (LL)
    const from = g >= 1 ? 6 * FS : 0;
    const to = g >= 1 ? 16 * FS : Math.min(irLength(g), Math.round((t60FromGamma(g, FS) + 0.6) * FS));
    for (const k of state.keys) buffers.set(k, getIR(k).slice(from, to));
  } else {
    let x;
    if (src === "file") {
      if (!state.userAudio) { $("audiofile").click(); throw new Error("choose an audio file first"); }
      x = state.userAudio;
    } else x = SOURCES[src].make(FS);
    if (g >= 1) throw new Error("convolution needs a decaying IR: pick expDE, training γ or a custom T60");
    for (const k of state.keys) {
      $("status").textContent = `Convolving ${LONG[k]}…`;
      buffers.set(k, await convolveLoop(x, getIR(k), FS));
    }
    dry = x;
  }
  // equal RMS across variants, one common peak limit
  const target = 0.05;
  let peak = 0;
  const gains = new Map();
  for (const [k, b] of buffers) {
    const s = target / (rms(b) || 1);
    gains.set(k, s);
    for (let i = 0; i < b.length; i++) peak = Math.max(peak, Math.abs(b[i]) * s);
  }
  const lim = Math.min(1, 0.95 / peak);
  for (const [k, b] of buffers) { const s = gains.get(k) * lim; for (let i = 0; i < b.length; i++) b[i] *= s; }
  if (dry) {
    const s = (target * lim) / (rms(dry) || 1);
    dry = Float32Array.from(dry, (v) => Math.max(-1, Math.min(1, v * s)));
  }
  engine.load(buffers, dry, FS);
}

let audioDirty = true;
async function play() {
  try {
    if (audioDirty) {
      $("status").textContent = "Rendering…";
      await buildAudio();
      audioDirty = false;
    }
    engine.start(state.sel, src_wet(), +$("vol").value);
    $("play").textContent = "Stop";
    $("status").textContent = `Playing ${LONG[state.sel]}`;
  } catch (e) {
    $("status").textContent = e.message;
  }
}
const src_wet = () => (state.source === "ir" ? 1 : +$("wet").value);
function stop() { engine.stop(); $("play").textContent = "Play"; $("status").textContent = ""; }
async function restartIfPlaying() {
  audioDirty = true;
  if (engine.playing) { stop(); await play(); } else $("status").textContent = "";
}

// ------------------------------------------------------------- controls
function select(k) {
  if (!state.keys.includes(k)) return;
  state.sel = k;
  if (state.cmp === k) state.cmp = "";
  renderVariantButtons();
  engine.select(k);
  if (engine.playing) $("status").textContent = `Playing ${LONG[k]}`;
  updateHash();
  renderParams(); renderIR(); renderMag(); renderModal(); renderNED();
}

function updateHash() {
  if (!state.run) return;
  const alt = state.run.alternatives?.length ? `&a=${state.alt}` : "";
  history.replaceState(null, "", `#run=${encodeURIComponent(state.run.id)}&v=${state.sel}&c=${state.cmp || ""}${alt}`);
}

// A run can offer several alternative training results for its custom slot (scripts/evaluate.py ALTERNATIVES);
// the chosen one is copied into the slot so every view treats it like any other variant.
function applyAlternative(run, id) {
  const a = run.alternatives.find((x) => x.id === id) || run.alternatives[0];
  run.variants.custom = { U: a.U, b: a.b, c: a.c };
  run.modal.custom = a.modal;
  run.metrics.custom = a.metrics;
  run.labels = { ...run.labels, custom: { short: a.short, long: a.long } };
  return a.id;
}

function setRun(run) {
  stop();
  state.run = run;
  const alts = run.alternatives || [];
  for (const id of [...cache.keys()]) if (id.includes("|custom|")) cache.delete(id);
  $("altwrap").hidden = !alts.length;
  if (alts.length) {
    state.alt = applyAlternative(run, state.alt);
    $("alt").replaceChildren(...alts.map((a) => new Option(a.long, a.id)));
    $("alt").value = state.alt;
    $("altlabel").textContent = `${alts[0].short} variant`;
  }
  // a run can name its custom slot (e.g. another training method on the same delays); a dropped file keeps "Custom"
  SHORT.custom = run.labels?.custom?.short ?? "Custom";
  LONG.custom = run.labels?.custom?.long ?? "Custom";
  state.fdns = buildFdns(run);
  state.keys = VARIANTS.map((v) => v.key).filter((k) => k === "noise" || state.fdns[k]);
  if (run.group === "file") state.sel = "custom";
  else if (!state.keys.includes(state.sel)) state.sel = state.keys[0];
  if (state.cmp && (!state.keys.includes(state.cmp) || state.cmp === state.sel)) state.cmp = "";
  audioDirty = true;
  renderVariantButtons();
  renderAll();
  updateHash();
}

async function loadRunById(id) {
  const item = state.index.find((r) => r.id === id) || state.index.find((r) => r.group === "reference") || state.index[0];
  if (!item) return;
  $("run").value = item.id;
  $("status").textContent = "Loading…";
  const run = await (await fetch(`data/${item.file}`, { cache: "no-cache" })).json();
  $("status").textContent = "";
  setRun(run);
}

function fromUserJson(d, name) {
  if (d.variants) return d;
  if (!d.delays || !(d.U || d.A) || !d.b || !d.c) throw new Error("expected a run export or {delays, U, b, c}");
  return { id: `file:${name}`, group: "file", label: name, delays: d.delays.map(Math.round), fs: 48000,
    gamma_train: d.gamma ?? 0.9999, variants: { custom: { U: d.U || d.A, b: d.b.flat(), c: d.c.flat() } }, modal: {}, metrics: {}, losses: {} };
}

async function init() {
  const params = new URLSearchParams(location.hash.slice(1));
  state.sel = params.get("v") || state.sel;
  state.cmp = params.get("c") ?? state.cmp;
  state.alt = params.get("a") || "";
  const src = $("source");
  for (const [k, v] of Object.entries(SOURCES)) src.append(new Option(v.label, k));
  try {
    state.index = await (await fetch("data/index.json", { cache: "no-cache" })).json();
  } catch {
    $("status").textContent = "No data/index.json - run scripts/evaluate.py, or load a JSON file.";
  }
  const sel = $("run");
  const groups = {};
  for (const r of state.index) (groups[r.group] ||= []).push(r);
  // our reproduction runs first: they carry every variant, including the alternative training methods
  for (const [g, items] of Object.entries(groups).sort(([a], [b]) => (b === "reference") - (a === "reference"))) {
    const og = document.createElement("optgroup");
    og.label = g;
    for (const r of items) og.append(new Option(r.label, r.id));
    sel.append(og);
  }
  sel.onchange = () => loadRunById(sel.value);
  $("compare").onchange = (e) => { state.cmp = e.target.value; updateHash(); renderIR(); renderMag(); renderModal(); renderNED(); };
  $("alt").onchange = (e) => {
    state.alt = applyAlternative(state.run, e.target.value);
    for (const id of [...cache.keys()]) if (id.includes("|custom|")) cache.delete(id);
    SHORT.custom = state.run.labels.custom.short;
    LONG.custom = state.run.labels.custom.long;
    state.fdns = buildFdns(state.run);
    renderVariantButtons(); renderAll(); updateHash(); restartIfPlaying();
  };
  $("decay").onchange = (e) => { state.decay = e.target.value; $("t60wrap").hidden = state.decay !== "custom"; renderAll(); restartIfPlaying(); };
  $("t60wrap").hidden = true;
  $("t60").oninput = (e) => { state.t60 = +e.target.value; $("t60out").textContent = `${state.t60.toFixed(1)} s`; };
  $("t60").onchange = () => { renderAll(); restartIfPlaying(); };
  $("t60out").textContent = "2.5 s";
  const wetVisible = () => { $("wet").parentElement.hidden = state.source === "ir"; };
  wetVisible();
  src.onchange = async (e) => {
    state.source = e.target.value;
    wetVisible();
    if (state.source === "file" && !state.userAudio) $("audiofile").click();
    restartIfPlaying();
  };
  $("audiofile").onchange = async (e) => {
    const f = e.target.files[0];
    if (!f) return;
    const buf = await audioContext().decodeAudioData(await f.arrayBuffer());
    const n = Math.min(buf.length, 30 * FS), x = new Float32Array(n);
    for (let ch = 0; ch < buf.numberOfChannels; ch++) { const d = buf.getChannelData(ch); for (let i = 0; i < n; i++) x[i] += d[i] / buf.numberOfChannels; }
    state.userAudio = x;
    state.source = "file"; src.value = "file";
    restartIfPlaying();
  };
  $("wet").oninput = (e) => { $("wetout").textContent = `${Math.round(e.target.value * 100)}%`; };
  $("wet").onchange = () => restartIfPlaying();
  $("wetout").textContent = "100%";
  $("vol").oninput = (e) => engine.setVolume(+e.target.value);
  $("play").onclick = () => (engine.playing ? stop() : play());
  $("zoomf").onchange = () => renderMag();
  for (const b of $("irrange").querySelectorAll("button")) {
    b.onclick = () => {
      state.irRange = b.dataset.v === "all" ? "all" : +b.dataset.v;
      $("irrange").querySelectorAll("button").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
      renderIR();
    };
  }
  $("load").onchange = async (e) => {
    const f = e.target.files[0];
    if (!f) return;
    try { setRun(fromUserJson(JSON.parse(await f.text()), f.name)); } catch (err) { $("status").textContent = err.message; }
  };
  document.addEventListener("dragover", (e) => e.preventDefault());
  document.addEventListener("drop", async (e) => {
    e.preventDefault();
    const f = e.dataTransfer.files[0];
    if (f) try { setRun(fromUserJson(JSON.parse(await f.text()), f.name)); } catch (err) { $("status").textContent = err.message; }
  });
  document.addEventListener("keydown", (e) => {
    if (e.target instanceof Element && e.target.matches("input, select, textarea")) return;
    if (e.code === "Space") { e.preventDefault(); engine.playing ? stop() : play(); }
    const n = parseInt(e.key, 10);
    if (n >= 1 && n <= state.keys.length) select(state.keys[n - 1]);
  });
  const themes = ["auto", "light", "dark"];
  $("theme").onclick = () => {
    const cur = document.documentElement.dataset.theme || "auto";
    const next = themes[(themes.indexOf(cur) + 1) % themes.length];
    if (next === "auto") delete document.documentElement.dataset.theme; else document.documentElement.dataset.theme = next;
    $("theme").textContent = `Theme: ${next}`;
    renderVariantButtons(); renderAll();
  };
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { renderVariantButtons(); renderAll(); });
  window.addEventListener("hashchange", () => {
    const p = new URLSearchParams(location.hash.slice(1));
    if (p.get("run") && p.get("run") !== state.run?.id) {
      // a link to another run carries its own selection
      state.sel = p.get("v") || state.sel;
      state.cmp = p.get("c") ?? state.cmp;
      state.alt = p.get("a") || state.alt;
      loadRunById(p.get("run"));
    } else if (p.get("v") && p.get("v") !== state.sel) select(p.get("v"));
  });
  if (state.index.length) await loadRunById(params.get("run"));
}

init();
