// Small canvas plotting layer: line / stairs / envelope / dots series, crosshair tooltip, heatmap.

const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

function niceTicks(lo, hi, count = 5) {
  const span = hi - lo;
  if (!(span > 0)) return [lo];
  const step0 = span / count, mag = Math.pow(10, Math.floor(Math.log10(step0)));
  const steps = [0.5, 1, 2, 2.5, 5, 10].map((m) => m * mag);
  const at = (step) => {
    const out = [];  // integer multiples of step, so 0 is exactly 0 (repeated addition drifted to 5.55e-17)
    for (let k = Math.ceil(lo / step - 1e-9); k * step <= hi + 1e-9 * span; k++) out.push(+(k * step).toPrecision(12));
    return out;
  };
  let i = steps.findIndex((s) => span / s <= count);
  if (i < 0) i = steps.length - 1;
  // a range just past a step leaves a lone tick (e.g. only 0 on a symmetric axis): step down until there are three
  let out = at(steps[i]);
  while (out.length < 3 && i > 0) out = at(steps[--i]);
  return out;
}
const LOG_TICKS = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000];
export const fmtHz = (f) => (f >= 1000 ? `${+(f / 1000).toFixed(f >= 10000 ? 0 : 1)}k` : `${+f.toFixed(0)}`);

function lowerBound(x, v) {
  let lo = 0, hi = x.length - 1;
  while (lo < hi) { const mid = (lo + hi) >> 1; if (x[mid] < v) lo = mid + 1; else hi = mid; }
  return lo;
}
function nearestIndex(x, v) {
  const i = lowerBound(x, v);
  return i > 0 && Math.abs(x[i - 1] - v) < Math.abs(x[i] - v) ? i - 1 : i;
}

export class Plot {
  /** opts: {xlabel, ylabel, xlog, xfmt, yfmt, height, crosshair} */
  constructor(container, opts = {}) {
    this.opts = { height: 220, crosshair: true, xfmt: (v) => `${+v.toPrecision(4)}`, yfmt: (v) => `${+v.toPrecision(4)}`, ...opts };
    this.wrap = document.createElement("div");
    this.wrap.className = "plot";
    this.canvas = document.createElement("canvas");
    this.canvas.style.height = `${this.opts.height}px`;
    this.tip = document.createElement("div");
    this.tip.className = "tooltip";
    this.tip.hidden = true;
    this.wrap.append(this.canvas, this.tip);
    container.append(this.wrap);
    this.series = [];
    this.hoverX = null;
    this.canvas.addEventListener("pointermove", (e) => this.onMove(e));
    this.canvas.addEventListener("pointerleave", () => { this.hoverX = null; this.tip.hidden = true; this.draw(); });
    new ResizeObserver(() => this.draw()).observe(this.wrap);
  }

  set(series, { xlim, ylim } = {}) {
    this.series = series.filter(Boolean);
    this.xlim = xlim;
    this.ylim = ylim;
    this.draw();
  }

  get pad() { return { l: 48, r: 12, t: 10, b: 34 }; }

  sx(v) {
    const { l, r } = this.pad, w = this.cw - l - r, [a, b] = this.xr;
    return this.opts.xlog ? l + ((Math.log(v) - Math.log(a)) / (Math.log(b) - Math.log(a))) * w : l + ((v - a) / (b - a)) * w;
  }
  invx(px) {
    const { l, r } = this.pad, w = this.cw - l - r, [a, b] = this.xr, u = (px - l) / w;
    return this.opts.xlog ? Math.exp(Math.log(a) + u * (Math.log(b) - Math.log(a))) : a + u * (b - a);
  }
  sy(v) {
    const { t, b: pb } = this.pad, h = this.ch - t - pb, [a, b] = this.yr;
    return t + (1 - (v - a) / (b - a)) * h;
  }

  autoRange() {
    let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
    for (const s of this.series) {
      const n = s.y.length;
      for (let i = 0; i < n; i++) {
        const xv = s.x ? s.x[i] : i, yv = s.y[i];
        if (!isFinite(yv)) continue;
        if (xv < x0) x0 = xv; if (xv > x1) x1 = xv;
        if (yv < y0) y0 = yv; if (yv > y1) y1 = yv;
        if (s.y2) { const v2 = s.y2[i]; if (v2 > y1) y1 = v2; if (v2 < y0) y0 = v2; }
      }
    }
    if (!isFinite(x0)) { x0 = 0; x1 = 1; y0 = 0; y1 = 1; }
    const yp = (y1 - y0) * 0.06 || 1;
    this.xr = this.xlim || [x0, x1];
    this.yr = this.ylim || [y0 - yp, y1 + yp];
  }

  draw() {
    const dpr = window.devicePixelRatio || 1;
    const rect = this.canvas.getBoundingClientRect();
    this.cw = rect.width; this.ch = rect.height;
    if (!this.cw) return;
    this.canvas.width = Math.round(this.cw * dpr);
    this.canvas.height = Math.round(this.ch * dpr);
    const g = this.canvas.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, this.cw, this.ch);
    this.autoRange();
    const { l, r, t, b } = this.pad;
    const ink2 = css("--text-secondary"), muted = css("--text-muted"), grid = css("--grid"), axis = css("--axis");
    g.font = "11px system-ui, -apple-system, 'Segoe UI', sans-serif";
    // grid + ticks
    g.strokeStyle = grid; g.lineWidth = 1; g.fillStyle = muted;
    const xt = this.opts.xlog ? LOG_TICKS.filter((v) => v >= this.xr[0] && v <= this.xr[1]) : niceTicks(this.xr[0], this.xr[1], Math.max(3, Math.floor((this.cw - l - r) / 80)));
    g.textAlign = "center"; g.textBaseline = "top";
    for (const v of xt) {
      const X = Math.round(this.sx(v)) + 0.5;
      g.beginPath(); g.moveTo(X, t); g.lineTo(X, this.ch - b); g.stroke();
      g.fillText(this.opts.xfmt(v), X, this.ch - b + 5);
    }
    g.textAlign = "right"; g.textBaseline = "middle";
    for (const v of niceTicks(this.yr[0], this.yr[1], Math.max(3, Math.floor((this.ch - t - b) / 40)))) {
      const Y = Math.round(this.sy(v)) + 0.5;
      g.beginPath(); g.moveTo(l, Y); g.lineTo(this.cw - r, Y); g.stroke();
      g.fillText(this.opts.yfmt(v), l - 6, Y);
    }
    g.strokeStyle = axis;
    g.beginPath(); g.moveTo(l, this.ch - b + 0.5); g.lineTo(this.cw - r, this.ch - b + 0.5); g.stroke();
    g.fillStyle = ink2; g.textAlign = "center"; g.textBaseline = "bottom";
    if (this.opts.xlabel) g.fillText(this.opts.xlabel, l + (this.cw - l - r) / 2, this.ch - 1);
    if (this.opts.ylabel) {
      g.save(); g.translate(11, t + (this.ch - t - b) / 2); g.rotate(-Math.PI / 2); g.textBaseline = "middle";
      g.fillText(this.opts.ylabel, 0, 0); g.restore();
    }
    // series
    g.save();
    g.beginPath(); g.rect(l, t, this.cw - l - r, this.ch - t - b); g.clip();
    for (const s of this.series) this.drawSeries(g, s);
    g.restore();
    // crosshair
    if (this.hoverX != null && this.opts.crosshair) {
      const X = Math.round(this.sx(this.hoverX)) + 0.5;
      g.strokeStyle = css("--text-muted"); g.lineWidth = 1;
      g.beginPath(); g.moveTo(X, t); g.lineTo(X, this.ch - b); g.stroke();
    }
  }

  drawSeries(g, s) {
    const n = s.y.length, xAt = (i) => (s.x ? s.x[i] : i);
    g.strokeStyle = s.color; g.fillStyle = s.color;
    g.globalAlpha = s.alpha ?? 1;
    g.lineWidth = s.width ?? 2; g.lineJoin = "round";
    g.setLineDash(s.dash || []);
    const i0 = Math.max(0, (s.x ? lowerBound(s.x, this.xr[0]) : Math.floor(this.xr[0])) - 1);
    const i1 = Math.min(n - 1, (s.x ? lowerBound(s.x, this.xr[1]) : Math.ceil(this.xr[1])) + 1);
    if (s.kind === "dots") {
      const rad = s.radius ?? 1.5;
      for (let i = i0; i <= i1; i++) { g.beginPath(); g.arc(this.sx(xAt(i)), this.sy(s.y[i]), rad, 0, 2 * Math.PI); g.fill(); }
    } else if (s.kind === "stairs") {   // x = bin edges (n+1), y = counts (n)
      g.beginPath();
      g.moveTo(this.sx(s.x[0]), this.sy(0));
      for (let i = 0; i < n; i++) { g.lineTo(this.sx(s.x[i]), this.sy(s.y[i])); g.lineTo(this.sx(s.x[i + 1]), this.sy(s.y[i])); }
      g.lineTo(this.sx(s.x[n]), this.sy(0));
      g.stroke();
    } else if (i1 - i0 > 3 * (this.cw - this.pad.l - this.pad.r)) {
      // decimate to a per-pixel min/max envelope
      let col = -1, mn = 0, mx = 0;
      g.beginPath();
      const flush = () => { if (col >= 0) { g.moveTo(col + 0.5, this.sy(mn)); g.lineTo(col + 0.5, this.sy(mx) + 0.01); } };
      for (let i = i0; i <= i1; i++) {
        const c = Math.floor(this.sx(xAt(i))), v = s.y[i], v2 = s.y2 ? s.y2[i] : v;
        if (c !== col) { flush(); col = c; mn = Math.min(v, v2); mx = Math.max(v, v2); }
        else { mn = Math.min(mn, v, v2); mx = Math.max(mx, v, v2); }
      }
      flush();
      g.lineWidth = 1;
      g.stroke();
    } else {
      g.beginPath();
      let pen = false;
      for (let i = i0; i <= i1; i++) {
        const v = s.y[i];
        if (!isFinite(v)) { pen = false; continue; }
        const X = this.sx(xAt(i)), Y = this.sy(v);
        pen ? g.lineTo(X, Y) : g.moveTo(X, Y);
        pen = true;
      }
      g.stroke();
    }
    g.globalAlpha = 1;
    g.setLineDash([]);
  }

  onMove(e) {
    if (!this.series.length) return;
    const rect = this.canvas.getBoundingClientRect();
    const px = e.clientX - rect.left;
    if (px < this.pad.l || px > this.cw - this.pad.r) { this.tip.hidden = true; return; }
    const xv = this.invx(px);
    const rows = [];
    let snapped = xv;
    if (this.opts.nearestPoint) {           // dense scatter: nearest point in screen space
      const py = e.clientY - rect.top;
      let best = null;
      for (const s of this.series) {
        if (!s.x) continue;
        const c = lowerBound(s.x, this.invx(px - 12)), c1 = lowerBound(s.x, this.invx(px + 12));
        for (let i = c; i <= Math.min(c1, s.x.length - 1); i++) {
          const d = Math.hypot(this.sx(s.x[i]) - px, this.sy(s.y[i]) - py);
          if (!best || d < best.d) best = { d, s, i };
        }
      }
      if (!best || best.d > 24) { this.tip.hidden = true; this.hoverX = null; this.draw(); return; }
      snapped = best.s.x[best.i];
      rows.push([best.s.color, this.opts.yfmt(best.s.y[best.i]), best.s.name]);
    } else {
      for (const s of this.series) {
        if (s.noTip) continue;
        let i, yv;
        if (s.kind === "stairs") {
          i = Math.min(s.y.length - 1, Math.max(0, lowerBound(s.x, xv) - 1));
          yv = s.y[i];
          snapped = 0.5 * (s.x[i] + s.x[i + 1]);
        } else if (s.x) {
          i = nearestIndex(s.x, xv); yv = s.y[i];
          if (s === this.series[0]) snapped = s.x[i];
        } else {
          i = Math.max(0, Math.min(s.y.length - 1, Math.round(xv))); yv = s.y[i];
          if (s === this.series[0]) snapped = i;
        }
        rows.push([s.color, this.opts.yfmt(yv), s.name]);
      }
    }
    this.hoverX = snapped;
    this.draw();
    this.tip.replaceChildren();
    const head = document.createElement("div");
    head.className = "tip-head";
    head.textContent = this.opts.xfmtTip ? this.opts.xfmtTip(snapped) : this.opts.xfmt(snapped);
    this.tip.append(head);
    for (const [color, val, name] of rows) {
      const row = document.createElement("div");
      row.className = "tip-row";
      const key = document.createElement("span"); key.className = "key"; key.style.background = color;
      const v = document.createElement("strong"); v.textContent = val;
      const nm = document.createElement("span"); nm.className = "tip-name"; nm.textContent = name;
      row.append(key, v, nm);
      this.tip.append(row);
    }
    this.tip.hidden = false;
    const X = this.sx(snapped);
    this.tip.style.left = `${X + 12 + this.tip.offsetWidth > this.cw ? X - 12 - this.tip.offsetWidth : X + 12}px`;
    this.tip.style.top = `${this.pad.t + 4}px`;
  }
}

// Diverging heatmap of a matrix (values in [-1, 1]); rows/cols labelled 1..N
export class Heatmap {
  constructor(container) {
    this.wrap = document.createElement("div");
    this.wrap.className = "plot heat";
    this.canvas = document.createElement("canvas");
    this.tip = document.createElement("div");
    this.tip.className = "tooltip"; this.tip.hidden = true;
    this.wrap.append(this.canvas, this.tip);
    container.append(this.wrap);
    this.canvas.addEventListener("pointermove", (e) => this.onMove(e));
    this.canvas.addEventListener("pointerleave", () => { this.hover = null; this.tip.hidden = true; this.draw(); });
    new ResizeObserver(() => this.draw()).observe(this.wrap);
  }
  set(M, labels) { this.M = M; this.labels = labels; this.draw(); }
  color(v) {
    const neg = css("--div-neg"), pos = css("--div-pos"), mid = css("--div-mid");
    const t = Math.min(1, Math.abs(v));
    return mix(mid, v < 0 ? neg : pos, Math.pow(t, 0.75));
  }
  draw() {
    if (!this.M) return;
    const dpr = window.devicePixelRatio || 1;
    const w = this.wrap.getBoundingClientRect().width;
    const rows = this.M.length, cols = this.M[0].length;
    const lab = 22, cell = Math.min(44, Math.floor((w - lab) / cols));
    this.geo = { lab, cell };
    const W = lab + cell * cols, H = lab + cell * rows;
    this.canvas.style.width = `${W}px`; this.canvas.style.height = `${H}px`;
    this.canvas.width = W * dpr; this.canvas.height = H * dpr;
    const g = this.canvas.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, W, H);
    g.font = "11px system-ui, -apple-system, 'Segoe UI', sans-serif";
    g.textAlign = "center"; g.textBaseline = "middle";
    const surface = css("--surface-1");
    for (let i = 0; i < rows; i++)
      for (let j = 0; j < cols; j++) {
        const v = this.M[i][j];
        g.fillStyle = this.color(v);
        g.fillRect(lab + j * cell + 1, lab + i * cell + 1, cell - 2, cell - 2);
        if (this.hover && this.hover[0] === i && this.hover[1] === j) {
          g.strokeStyle = css("--text-primary"); g.lineWidth = 2;
          g.strokeRect(lab + j * cell + 2, lab + i * cell + 2, cell - 4, cell - 4);
        }
        if (cell >= 34) {
          g.fillStyle = Math.abs(v) > 0.55 ? "#ffffff" : css("--text-primary");
          g.fillText(v.toFixed(2), lab + j * cell + cell / 2, lab + i * cell + cell / 2);
        }
      }
    g.fillStyle = css("--text-muted");
    for (let j = 0; j < cols; j++) g.fillText(String(j + 1), lab + j * cell + cell / 2, lab / 2);
    for (let i = 0; i < rows; i++) g.fillText(String(i + 1), lab / 2, lab + i * cell + cell / 2);
    void surface;
  }
  onMove(e) {
    if (!this.M) return;
    const r = this.canvas.getBoundingClientRect();
    const { lab, cell } = this.geo;
    const j = Math.floor((e.clientX - r.left - lab) / cell), i = Math.floor((e.clientY - r.top - lab) / cell);
    if (i < 0 || j < 0 || i >= this.M.length || j >= this.M[0].length) { this.tip.hidden = true; this.hover = null; this.draw(); return; }
    this.hover = [i, j];
    this.draw();
    this.tip.replaceChildren();
    const s = document.createElement("strong"); s.textContent = this.M[i][j].toFixed(4);
    const n = document.createElement("span"); n.className = "tip-name"; n.textContent = ` ${this.labels?.(i, j) ?? `[${i + 1},${j + 1}]`}`;
    this.tip.append(s, n);
    this.tip.hidden = false;
    this.tip.style.left = `${lab + (j + 1) * cell + 4}px`;
    this.tip.style.top = `${lab + i * cell}px`;
  }
}

function mix(a, b, t) {
  const pa = parse(a), pb = parse(b);
  return `rgb(${pa.map((v, k) => Math.round(v + (pb[k] - v) * t)).join(",")})`;
}
function parse(hex) {
  const h = hex.replace("#", "");
  return [0, 2, 4].map((k) => parseInt(h.slice(k, k + 2), 16));
}
