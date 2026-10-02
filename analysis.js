// Geometry of a concept inside an embedding model.
//
// A dataset is a dense grid: T sentence templates x V ordered values, with a
// D-dim [CLS] vector at each of L layers. All analysis happens on one layer.
// The template's own mean is subtracted first, so what is left is the part of
// the representation that moves when only the value changes.

const dot = (a, ao, b, bo, n) => { let s = 0; for (let i = 0; i < n; i++) s += a[ao + i] * b[bo + i]; return s; };

// Unit-normalised rows of one layer: Float32Array (T*V*D)
export function layerRows(ds, layer) {
  const { T, V, L, D, data } = ds;
  const X = new Float32Array(T * V * D);
  for (let r = 0; r < T * V; r++) {
    const src = (r * L + layer) * D, dst = r * D;
    let n = 0;
    for (let i = 0; i < D; i++) n += data[src + i] * data[src + i];
    n = Math.sqrt(n) || 1;
    for (let i = 0; i < D; i++) X[dst + i] = data[src + i] / n;
  }
  return X;
}

// Subtract each template's mean over values (in place)
function centreTemplates(X, T, V, D) {
  const mu = new Float64Array(D);
  for (let t = 0; t < T; t++) {
    mu.fill(0);
    for (let v = 0; v < V; v++) for (let i = 0; i < D; i++) mu[i] += X[(t * V + v) * D + i];
    for (let v = 0; v < V; v++) for (let i = 0; i < D; i++) X[(t * V + v) * D + i] -= mu[i] / V;
  }
  return X;
}

// Mean over a subset of templates (and optionally a subset of values) -> (vals.length x D)
function valueMeans(C, V, D, temps, vals) {
  const M = new Float64Array(vals.length * D);
  vals.forEach((v, j) => {
    for (const t of temps) for (let i = 0; i < D; i++) M[j * D + i] += C[(t * V + v) * D + i];
    for (let i = 0; i < D; i++) M[j * D + i] /= temps.length;
  });
  return M;
}

// Top-k principal axes of the rows of M (n x D), via the n x n Gram matrix.
// Returns { W: k axes (k x D), share: variance share per axis, total }.
function principalAxes(M, n, D, k) {
  const mu = new Float64Array(D);
  for (let r = 0; r < n; r++) for (let i = 0; i < D; i++) mu[i] += M[r * D + i] / n;
  const Z = new Float64Array(n * D);
  for (let r = 0; r < n; r++) for (let i = 0; i < D; i++) Z[r * D + i] = M[r * D + i] - mu[i];
  const G = new Float64Array(n * n);
  let total = 0;
  for (let a = 0; a < n; a++) {
    for (let b = a; b < n; b++) G[a * n + b] = G[b * n + a] = dot(Z, a * D, Z, b * D, D);
    total += G[a * n + a];
  }
  k = Math.max(0, Math.min(k, n - 1));
  const W = new Float64Array(k * D), share = [], vecs = [];
  let seed = 1;
  const rand = () => ((seed = (seed * 16807) % 2147483647) / 2147483647) - 0.5;
  for (let c = 0; c < k; c++) {
    let u = Float64Array.from({ length: n }, rand), lam = 0;
    for (let it = 0; it < 200; it++) {
      for (const p of vecs) { const d = dot(u, 0, p, 0, n); for (let i = 0; i < n; i++) u[i] -= d * p[i]; }
      const w = new Float64Array(n);
      for (let a = 0; a < n; a++) w[a] = dot(G, a * n, u, 0, n);
      const nn = Math.sqrt(dot(w, 0, w, 0, n));
      if (nn < 1e-12) { lam = 0; break; }
      for (let i = 0; i < n; i++) w[i] /= nn;
      const delta = Math.abs(nn - lam);
      u = w; lam = nn;
      if (it > 20 && delta < 1e-9 * lam) break;
    }
    vecs.push(u);
    share.push(total > 0 ? lam / total : 0);
    const s = Math.sqrt(lam) || 1;
    for (let r = 0; r < n; r++) for (let i = 0; i < D; i++) W[c * D + i] += (u[r] * Z[r * D + i]) / s;
  }
  return { W, share, total, k };
}

// Least squares with a small ridge; F is rows x k (an intercept is added).
function fitLinear(F, rows, k, y) {
  const p = k + 1, A = new Float64Array(p * p), b = new Float64Array(p), x = new Float64Array(p);
  for (let r = 0; r < rows; r++) {
    for (let i = 0; i < p; i++) {
      const fi = i < k ? F[r * k + i] : 1;
      b[i] += fi * y[r];
      for (let j = i; j < p; j++) A[i * p + j] += fi * (j < k ? F[r * k + j] : 1);
    }
  }
  for (let i = 0; i < p; i++) { for (let j = 0; j < i; j++) A[i * p + j] = A[j * p + i]; if (i < k) A[i * p + i] *= 1 + 1e-6; A[i * p + i] += 1e-10; }
  for (let i = 0; i < p; i++) {
    let m = i;
    for (let r = i + 1; r < p; r++) if (Math.abs(A[r * p + i]) > Math.abs(A[m * p + i])) m = r;
    if (m !== i) { for (let j = 0; j < p; j++) [A[i * p + j], A[m * p + j]] = [A[m * p + j], A[i * p + j]]; [b[i], b[m]] = [b[m], b[i]]; }
    const d = A[i * p + i] || 1e-12;
    for (let r = i + 1; r < p; r++) {
      const f = A[r * p + i] / d;
      for (let j = i; j < p; j++) A[r * p + j] -= f * A[i * p + j];
      b[r] -= f * b[i];
    }
  }
  for (let i = p - 1; i >= 0; i--) {
    let s = b[i];
    for (let j = i + 1; j < p; j++) s -= A[i * p + j] * x[j];
    x[i] = s / (A[i * p + i] || 1e-12);
  }
  return x;
}

const predict = (F, rows, k, w) => Float64Array.from({ length: rows }, (_, r) => { let s = w[k]; for (let i = 0; i < k; i++) s += w[i] * F[r * k + i]; return s; });

function r2(y, p) {
  let m = 0; for (const v of y) m += v / y.length;
  let res = 0, tot = 0;
  for (let i = 0; i < y.length; i++) { res += (y[i] - p[i]) ** 2; tot += (y[i] - m) ** 2; }
  return tot > 0 ? 1 - res / tot : null;
}

// Probe: learn on (train templates x train values), test on (test templates x test values).
// Features are the projections onto the top-k axes of the training value means.
function probe(C, V, D, trT, teT, trV, teV, targets, k) {
  const M = valueMeans(C, V, D, trT, trV);
  const pa = principalAxes(M, trV.length, D, k);
  const feats = (temps, vals) => {
    const F = new Float64Array(temps.length * vals.length * pa.k);
    let r = 0;
    for (const t of temps) for (const v of vals) { for (let c = 0; c < pa.k; c++) F[r * pa.k + c] = dot(C, (t * V + v) * D, pa.W, c * D, D); r++; }
    return F;
  };
  const Ftr = feats(trT, trV), Fte = feats(teT, teV);
  const nTr = trT.length * trV.length, nTe = teT.length * teV.length;
  const scores = targets.map((tg) => {
    const ytr = Float64Array.from({ length: nTr }, (_, r) => tg[trV[r % trV.length]]);
    const yte = Float64Array.from({ length: nTe }, (_, r) => tg[teV[r % teV.length]]);
    return r2(yte, predict(Fte, nTe, pa.k, fitLinear(Ftr, nTr, pa.k, ytr)));
  });
  return scores;
}

export function analyze(ds, layer, opts = {}) {
  const { T, V, D } = ds;
  const light = !!opts.light;                       // skip per-sentence projections
  const C = centreTemplates(layerRows(ds, layer), T, V, D);
  const allT = [...Array(T).keys()], allV = [...Array(V).keys()];
  const M = valueMeans(C, V, D, allT, allV);
  const kShow = Math.min(6, V - 1);
  const pa = principalAxes(M, V, D, kShow);

  const coords = [];                                // value means in PC space: V x kShow
  for (let v = 0; v < V; v++) coords.push(Array.from({ length: kShow }, (_, c) => dot(M, v * D, pa.W, c * D, D)));
  let sentences = null;                             // every sentence: (T*V) x kShow
  if (!light) {
    sentences = new Float32Array(T * V * kShow);
    for (let r = 0; r < T * V; r++) for (let c = 0; c < kShow; c++) sentences[r * kShow + c] = dot(C, r * D, pa.W, c * D, D);
  }

  const norms = allV.map((v) => Math.sqrt(dot(M, v * D, M, v * D, D)) || 1);
  const sim = allV.map((a) => allV.map((b) => dot(M, a * D, M, b * D, D) / (norms[a] * norms[b])));
  const dist = (a, b) => { let s = 0; for (let i = 0; i < D; i++) s += (M[a * D + i] - M[b * D + i]) ** 2; return Math.sqrt(s); };

  let path = 0;
  for (let v = 0; v + 1 < V; v++) path += dist(v, v + 1);
  const adj = path / Math.max(1, V - 1);
  let simAdj = 0, simWrap = sim[V - 1][0], simAll = 0, nAll = 0;
  for (let a = 0; a < V; a++) for (let b = a + 1; b < V; b++) { simAll += sim[a][b]; nAll++; if (b === a + 1) simAdj += sim[a][b]; }
  let cNorm = 0;
  for (let i = 0; i < C.length; i++) cNorm += C[i] * C[i];

  // probes on held-out templates
  const teT = allT.filter((t) => t % 4 === 3), trT = allT.filter((t) => t % 4 !== 3);
  const idx = allV.map((v) => v / Math.max(1, V - 1));
  const cos = allV.map((v) => Math.cos((2 * Math.PI * v) / V)), sin = allV.map((v) => Math.sin((2 * Math.PI * v) / V));
  let lineR2 = null, ringR2 = null, unseenR2 = null;
  if (teT.length >= 1 && trT.length >= 2 && V >= 4) {
    // two dimensions for both, so neither probe can succeed just by telling the values apart
    const [a, b, c] = probe(C, V, D, trT, teT, allV, allV, [idx, cos, sin], 2);
    lineR2 = a; ringR2 = b === null || c === null ? null : (b + c) / 2;
    if (V >= 8) {
      const even = allV.filter((v) => v % 2 === 0), odd = allV.filter((v) => v % 2 === 1);
      [unseenR2] = probe(C, V, D, trT, teT, even, odd, [idx], Math.min(12, even.length - 1));
    }
  }

  const m = {
    pcShare: pa.share,
    straightness: path > 0 ? dist(0, V - 1) / path : 0,
    wrapRatio: adj > 0 ? dist(0, V - 1) / adj : 0,
    neighbourSim: simAdj / Math.max(1, V - 1),
    wrapSim: simWrap,
    otherSim: nAll > V - 1 ? (simAll - simAdj) / (nAll - (V - 1)) : 0,
    shared: cNorm > 0 ? (T * pa.total) / cNorm : 0,
    lineR2, ringR2, unseenR2,
  };
  return { coords, sentences, kShow, sim, metrics: m, verdict: verdict(m, V) };
}

// A plain-language reading of the numbers. Thresholds are rules of thumb.
export function verdict(m, V) {
  if (V < 4) return { shape: "Too few values", why: "Add at least four values to say anything about shape." };
  const pc = (i) => m.pcShare[i] ?? 0, pct = (x) => `${Math.round(100 * x)}%`;
  const gap = m.neighbourSim - m.otherSim;
  const ordered = gap > 0.1;
  const closes = m.wrapSim - m.otherSim > 0.5 * gap;
  const line2 = m.lineR2 ?? 0, ring2 = m.ringR2 ?? 0;
  if (ordered && closes && V >= 5 && ring2 > 0.6 && ring2 > line2)
    return { shape: "Ring", why: `Neighbours sit together and the last value sits next to the first. The loop lives in the two main directions, which hold ${pct(pc(0) + pc(1))} of the variation.` };
  if (line2 > 0.8 && pc(0) > 0.5 && m.straightness > 0.4)
    return { shape: "Line", why: `One direction holds ${pct(pc(0))} of the variation and the values march along it in order.` };
  if (ordered && pc(0) + pc(1) + pc(2) > 0.6)
    return { shape: "Curve", why: "The values follow a path in order, but it bends instead of running straight or closing into a loop." };
  if (ordered || (m.unseenR2 ?? 0) > 0.5)
    return { shape: "Ordered, but not a simple shape", why: `Nearby values are more alike and a probe can recover the order, yet the main direction holds only ${pct(pc(0))} of the variation. Mostly each value has its own direction, with a weak shared trend underneath.` };
  return { shape: "Cloud", why: "Each value sits in its own direction, like entries in a lookup table. The order you gave is not visible in the geometry." };
}

// IEEE half-precision -> Float32Array (presets are stored as float16)
export function decodeF16(buf) {
  const h = new Uint16Array(buf), out = new Float32Array(h.length);
  for (let i = 0; i < h.length; i++) {
    const x = h[i], e = (x >> 10) & 31, f = x & 1023, s = x & 0x8000 ? -1 : 1;
    out[i] = e === 0 ? s * f * 2 ** -24 : e === 31 ? (f ? NaN : s * Infinity) : s * (1 + f / 1024) * 2 ** (e - 15);
  }
  return out;
}
