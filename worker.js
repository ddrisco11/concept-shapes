// Runs both models off the main thread:
//   - bge-small-en-v1.5 (int8 ONNX served from this site) -> [CLS] vector at every layer
//   - Gemma 3 270M (fetched from the Hugging Face hub)    -> sentence contexts
import { AutoModel, AutoTokenizer, env, pipeline } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.0";

// The embedder is served by this site, Gemma by the Hugging Face hub; loads are sequential,
// so the hub address is swapped just for the embedder.
env.allowLocalModels = false;
const HUB = { host: env.remoteHost, path: env.remotePathTemplate };
const SITE = { host: new URL("./models/", self.location.href).href, path: "{model}/" };
const useHost = (h) => { env.remoteHost = h.host; env.remotePathTemplate = h.path; };

const EMBEDDER = "bge-small-layers";
const WRITER = "onnx-community/gemma-3-270m-it-ONNX";
const post = (type, body = {}) => self.postMessage({ type, ...body });
const progress = (label) => (p) => {
  if (p.status === "progress" && p.total) post("status", { text: `${label}: downloading ${Math.round((100 * p.loaded) / p.total)}%` });
};

let embedder = null, writer = null;

async function loadEmbedder() {
  if (!embedder) {
    post("status", { text: "Loading bge-small…" });
    useHost(SITE);
    const tok = await AutoTokenizer.from_pretrained(EMBEDDER);
    const model = await AutoModel.from_pretrained(EMBEDDER, { dtype: "q8", device: "wasm", progress_callback: progress("bge-small") });
    useHost(HUB);
    embedder = { tok, model };
  }
  return embedder;
}

async function embed(id, sentences) {
  const { tok, model } = await loadEmbedder();
  const BATCH = 16;
  let out = null, per = 0;
  for (let i = 0; i < sentences.length; i += BATCH) {
    const inputs = tok(sentences.slice(i, i + BATCH), { padding: true, truncation: true, max_length: 64 });
    const { cls_layers } = await model(inputs);
    if (!out) { per = cls_layers.dims[1] * cls_layers.dims[2]; out = new Float32Array(sentences.length * per); post("dims", { id, L: cls_layers.dims[1], D: cls_layers.dims[2] }); }
    out.set(cls_layers.data, i * per);
    post("progress", { id, done: Math.min(i + BATCH, sentences.length), total: sentences.length });
  }
  self.postMessage({ type: "embedded", id, data: out }, [out.buffer]);
}

async function loadWriter() {
  if (!writer) {
    useHost(HUB);
    const gpu = !!self.navigator.gpu && !!(await self.navigator.gpu.requestAdapter().catch(() => null));
    post("status", { text: `Loading Gemma 3 270M (${gpu ? "WebGPU" : "CPU, slower"})…` });
    writer = await pipeline("text-generation", WRITER, { device: gpu ? "webgpu" : "wasm", dtype: gpu ? "fp16" : "q4", progress_callback: progress("Gemma 3 270M") });
  }
  return writer;
}

async function write(id, prompt) {
  const gen = await loadWriter();
  const res = await gen([{ role: "user", content: prompt }], { max_new_tokens: 320, do_sample: true, temperature: 1.0, top_p: 0.95 });
  post("written", { id, text: res[0].generated_text.at(-1).content });
}

self.onmessage = async ({ data }) => {
  try {
    if (data.type === "embed") await embed(data.id, data.sentences);
    else if (data.type === "write") await write(data.id, data.prompt);
  } catch (e) {
    post("error", { id: data.id, message: String(e?.message ?? e) });
  }
};
