import { readFileSync } from "node:fs";
import { analyze, decodeF16 } from "../analysis.js";
const idx = JSON.parse(readFileSync("presets/index.json"));
for (const p of idx) {
  const b = readFileSync(`presets/${p.file}`);
  const ds = { ...p, data: decodeF16(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength)) };
  const t0 = performance.now();
  const r = analyze(ds, 12);
  const m = r.metrics, f = (x) => (x == null ? "n/a" : x.toFixed(2));
  console.log(p.id.padEnd(9), r.verdict.shape.padEnd(14), `pc ${m.pcShare.slice(0, 3).map(f)}  straight ${f(m.straightness)}  nb ${f(m.neighbourSim)}/${f(m.otherSim)}  wrap ${f(m.wrapSim)}  shared ${f(m.shared)}  line ${f(m.lineR2)} ring ${f(m.ringR2)} unseen ${f(m.unseenR2)}  ${Math.round(performance.now() - t0)}ms`);
}
