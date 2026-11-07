// FDN rendering and analysis in the browser (mirrors fdnopt/fdn.py and fdnopt/metrics.py).

export const gammaFromT60 = (t60, fs) => Math.pow(10, -3 / (fs * t60));
export const t60FromGamma = (g, fs) => -3 / (fs * Math.log10(g));

// Exact delay-line simulation: H(z) = c^T [D_m(z)^-1 - U diag(gamma^m)]^-1 b
export function renderFDN({ delays, U, b, c }, gamma, length) {
  const N = delays.length;
  const A = new Float64Array(N * N);
  for (let i = 0; i < N; i++)
    for (let j = 0; j < N; j++) A[i * N + j] = U[i][j] * Math.pow(gamma, delays[j]);
  const bufs = delays.map((m) => new Float64Array(m));
  const ptr = new Int32Array(N);
  const outs = new Float64Array(N);
  const h = new Float32Array(length);
  for (let n = 0; n < length; n++) {
    let y = 0;
    for (let i = 0; i < N; i++) {
      outs[i] = bufs[i][ptr[i]];
      y += c[i] * outs[i];
    }
    h[n] = y;
    const x = n === 0 ? 1 : 0;
    for (let i = 0; i < N; i++) {
      let v = b[i] * x;
      const row = i * N;
      for (let j = 0; j < N; j++) v += A[row + j] * outs[j];
      bufs[i][ptr[i]] = v;
      ptr[i] = ptr[i] + 1 === delays[i] ? 0 : ptr[i] + 1;
    }
  }
  return h;
}

// White Gaussian noise with the same homogeneous decay (the paper's MUSHRA reference)
export function noiseReference(gamma, length, seed = 1) {
  let s = seed >>> 0;
  const rand = () => ((s = (s * 1664525 + 1013904223) >>> 0) + 0.5) / 4294967296;
  const h = new Float32Array(length);
  let g = 1;
  for (let n = 0; n < length; n += 2) {
    const r = Math.sqrt(-2 * Math.log(rand())), t = 2 * Math.PI * rand();
    h[n] = r * Math.cos(t) * g;
    g *= gamma;
    if (n + 1 < length) h[n + 1] = r * Math.sin(t) * g;
    g *= gamma;
  }
  return h;
}

// ---------------------------------------------------------------- matrices
export const identity = (N) => Array.from({ length: N }, (_, i) => Array.from({ length: N }, (_, j) => +(i === j)));
export const householder = (N) => Array.from({ length: N }, (_, i) => Array.from({ length: N }, (_, j) => (i === j) - 2 / N));
export function hadamard(N) {
  if (N & (N - 1)) return null;
  let H = [[1]];
  while (H.length < N) H = [...H.map((r) => [...r, ...r]), ...H.map((r) => [...r, ...r.map((v) => -v)])];
  return H.map((r) => r.map((v) => v / Math.sqrt(N)));
}
export function orthogonalityError(U) {
  const N = U.length;
  let e = 0;
  for (let i = 0; i < N; i++)
    for (let j = 0; j < N; j++) {
      let s = 0;
      for (let k = 0; k < N; k++) s += U[i][k] * U[j][k];
      e = Math.max(e, Math.abs(s - (i === j)));
    }
  return e;
}

// ------------------------------------------------------------------- FFT
const twiddleCache = new Map();
export function fft(re, im) {
  const n = re.length;
  let tw = twiddleCache.get(n);
  if (!tw) {
    tw = { c: new Float64Array(n / 2), s: new Float64Array(n / 2) };
    for (let k = 0; k < n / 2; k++) { tw.c[k] = Math.cos((2 * Math.PI * k) / n); tw.s[k] = -Math.sin((2 * Math.PI * k) / n); }
    twiddleCache.set(n, tw);
  }
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) { [re[i], re[j]] = [re[j], re[i]]; [im[i], im[j]] = [im[j], im[i]]; }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const half = len >> 1, step = n / len;
    for (let i = 0; i < n; i += len)
      for (let k = 0; k < half; k++) {
        const wr = tw.c[k * step], wi = tw.s[k * step];
        const a = i + k, b = a + half;
        const vr = re[b] * wr - im[b] * wi, vi = re[b] * wi + im[b] * wr;
        re[b] = re[a] - vr; im[b] = im[a] - vi;
        re[a] += vr; im[a] += vi;
      }
  }
}

// Power spectrum |H(k)|^2, k = 0..n/2, of a (zero-padded) IR
export function powerSpectrum(h, n) {
  const re = new Float64Array(n), im = new Float64Array(n);
  re.set(h.subarray(0, Math.min(h.length, n)));
  fft(re, im);
  const P = new Float64Array(n / 2 + 1);
  for (let k = 0; k <= n / 2; k++) P[k] = re[k] * re[k] + im[k] * im[k];
  return P;
}

// Abel & Huang normalized echo density (1 = Gaussian), 20 ms Hann window
export function echoDensity(h, fs, durS = 1, winMs = 20, hop = 96) {
  const L = Math.round((winMs * 1e-3 * fs) / 2) * 2 + 1, half = (L - 1) / 2;
  const w = new Float64Array(L);
  let ws = 0;
  for (let i = 0; i < L; i++) ws += (w[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * (i + 1)) / (L + 1)));
  for (let i = 0; i < L; i++) w[i] /= ws;
  const erfc1 = 0.31731050786291415; // erfc(1/sqrt 2)
  const end = Math.min(h.length - half, Math.round(durS * fs));
  const t = [], d = [];
  for (let n = half; n < end; n += hop) {
    let p = 0;
    for (let i = 0; i < L; i++) { const v = h[n - half + i]; p += w[i] * v * v; }
    const sd = Math.sqrt(p);
    let cnt = 0;
    for (let i = 0; i < L; i++) if (Math.abs(h[n - half + i]) > sd) cnt += w[i];
    t.push(n / fs);
    d.push(cnt / erfc1);
  }
  return { t: Float64Array.from(t), d: Float64Array.from(d) };
}

export function rms(x, from = 0, to = x.length) {
  let s = 0;
  for (let i = from; i < to; i++) s += x[i] * x[i];
  return Math.sqrt(s / Math.max(1, to - from));
}

export function modalStats(db) {
  const floor = median(db) - 100; // drop unexcited modes (as in fdnopt.metrics)
  const v = db.filter((x) => x > floor);
  const mean = v.reduce((a, b) => a + b, 0) / v.length;
  const c = v.map((x) => x - mean).sort((a, b) => a - b);
  const std = Math.sqrt(c.reduce((a, b) => a + b * b, 0) / c.length);
  const q = (p) => c[Math.min(c.length - 1, Math.floor(p * (c.length - 1)))];
  return { centered: c, std, p99: q(0.99), max: c[c.length - 1], n: c.length };
}
function median(a) {
  const s = Array.from(a).sort((x, y) => x - y);
  return s[s.length >> 1];
}
