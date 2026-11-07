// Playback engine: every variant is rendered to a loop of identical length and all loops run in sync;
// switching variants is a 25 ms crossfade, so A/B comparisons are time-aligned.

let ctx = null;
export function audioContext() {
  if (!ctx) ctx = new AudioContext({ sampleRate: 48000 });
  return ctx;
}

function toBuffer(c, data, sr) {
  const b = c.createBuffer(1, data.length, sr);
  b.copyToChannel(data, 0);
  return b;
}

// Steady-state wet signal of a periodically repeated source: linear convolution folded modulo the loop length
export async function convolveLoop(src, ir, sr) {
  const L = src.length;
  const off = new OfflineAudioContext(1, L + ir.length, sr);
  const s = off.createBufferSource();
  s.buffer = toBuffer(off, src, sr);
  const conv = off.createConvolver();
  conv.normalize = false;
  conv.buffer = toBuffer(off, ir, sr);
  s.connect(conv).connect(off.destination);
  s.start();
  const out = (await off.startRendering()).getChannelData(0);
  const y = new Float32Array(L);
  for (let n = 0; n < out.length; n++) y[n % L] += out[n];
  return y;
}

// ----------------------------------------------------------- test sources
function rng(seed) {
  let s = seed >>> 0;
  return () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296) * 2 - 1;
}

export const SOURCES = {
  ir: { label: "Impulse response itself (paper's listening test)" },
  clicks: { label: "Clicks", make: (sr) => { const x = new Float32Array(2 * sr); x[0] = 1; x[sr] = 0.5; return x; } },
  burst: {
    label: "Noise bursts", make: (sr) => {
      const x = new Float32Array(2 * sr), r = rng(7), L = Math.round(0.02 * sr);
      for (let i = 0; i < L; i++) x[i] = 0.5 * r() * Math.sin((Math.PI * i) / L);
      return x;
    },
  },
  pluck: {
    label: "Plucked arpeggio", make: (sr) => {
      const x = new Float32Array(4 * sr), r = rng(3);
      [220, 261.63, 329.63, 392, 440, 392, 329.63, 261.63].forEach((f, k) => {
        const P = Math.round(sr / f), buf = Float32Array.from({ length: P }, () => r());
        const start = Math.round(k * 0.5 * sr), dur = Math.round(0.45 * sr);
        for (let n = 0, p = 0; n < dur; n++) {
          const nx = (p + 1) % P;
          const v = buf[p];
          buf[p] = 0.996 * 0.5 * (buf[p] + buf[nx]);
          p = nx;
          x[start + n] += 0.3 * v * Math.min(1, (dur - n) / 200);
        }
      });
      return x;
    },
  },
  drums: {
    label: "Drum loop", make: (sr) => {
      const x = new Float32Array(2 * sr), r = rng(11);
      const add = (t, fn, dur) => { const s = Math.round(t * sr); for (let n = 0; n < dur * sr && s + n < x.length; n++) x[s + n] += fn(n / sr); };
      for (const t of [0, 1]) add(t, (u) => 0.8 * Math.sin(2 * Math.PI * (50 * u + 70 * (1 - Math.exp(-u * 30)) / 30)) * Math.exp(-u * 9), 0.4);
      for (const t of [0.5, 1.5]) add(t, (u) => (0.35 * r() + 0.3 * Math.sin(2 * Math.PI * 185 * u)) * Math.exp(-u * 22), 0.25);
      for (let k = 0; k < 8; k++) add(k * 0.25 + 0.125 * (k % 2), (u) => 0.08 * r() * Math.exp(-u * 90), 0.06);
      return x;
    },
  },
  file: { label: "Your audio file…" },
};

// ------------------------------------------------------------------ engine
export class Engine {
  constructor() {
    this.voices = new Map();
    this.playing = false;
    this.selected = null;
    this.onEnded = null;
  }

  // buffers: Map(key -> Float32Array), all the same length; dry: optional Float32Array of that length
  load(buffers, dry, sr) {
    const c = audioContext();
    this.buffers = new Map([...buffers].map(([k, v]) => [k, toBuffer(c, v, sr)]));
    this.dry = dry ? toBuffer(c, dry, sr) : null;
  }

  start(selected, wet, volume) {
    const c = audioContext();
    c.resume();
    this.stop();
    this.master = c.createGain();
    this.master.gain.value = volume;
    this.master.connect(c.destination);
    const t0 = c.currentTime + 0.08;
    for (const [k, buf] of this.buffers) {
      const src = c.createBufferSource(), g = c.createGain();
      src.buffer = buf; src.loop = true;
      g.gain.value = k === selected ? wet : 0;
      src.connect(g).connect(this.master);
      src.start(t0);
      this.voices.set(k, { src, g });
    }
    if (this.dry) {
      const src = c.createBufferSource(), g = c.createGain();
      src.buffer = this.dry; src.loop = true;
      g.gain.value = 1 - wet;
      src.connect(g).connect(this.master);
      src.start(t0);
      this.voices.set("__dry", { src, g });
    }
    this.selected = selected;
    this.wet = wet;
    this.playing = true;
  }

  select(key) {
    this.selected = key;
    if (!this.playing) return;
    const t = audioContext().currentTime;
    for (const [k, v] of this.voices) {
      if (k === "__dry") continue;
      v.g.gain.cancelScheduledValues(t);
      v.g.gain.setValueAtTime(v.g.gain.value, t);
      v.g.gain.linearRampToValueAtTime(k === key ? this.wet : 0, t + 0.025);
    }
  }

  setVolume(v) { if (this.master) this.master.gain.setTargetAtTime(v, audioContext().currentTime, 0.02); }

  stop() {
    for (const v of this.voices.values()) { try { v.src.stop(); } catch { /* not started */ } v.src.disconnect(); }
    this.voices.clear();
    if (this.master) this.master.disconnect();
    this.playing = false;
  }
}
