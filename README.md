# Concept Shapes

See the shape a small embedding model gives to a concept: a line, a ring, or a cloud.

**Live page:** https://ddrisco11.github.io/concept-shapes/

You give an ordered list of values (years, months, temperatures, anything). The page puts every value into the same set of sentences, embeds them with `bge-small-en-v1.5`, subtracts each sentence's own average, and shows what is left: the part of the representation that moves when only the value changes.

## What you see

- **3D view** of the value means in their first principal components, joined in order. Other components and every individual sentence can be switched on.
- **Similarity heatmap** between values. A line or ring shows as a band along the diagonal; a lookup table shows as a bare diagonal.
- **Scorecard**: variance per component, straightness, neighbour versus non-neighbour similarity, whether the ends meet, how much of the effect is shared across sentences, and probes tested on unseen sentences and unseen values.
- **Layer slider** over the embedding layer and all 12 transformer layers, with a chart of where order first becomes readable.

## How it runs

Everything runs in the browser; there is no server.

- `bge-small-en-v1.5` is exported to ONNX with one output, the `[CLS]` vector at every layer, and quantised to int8 (35 MB, in `models/`). It matches the original model at about 0.99 cosine on the final layer.
- Sentences come from Gemma 3 270M running in the page (WebGPU), from a built-in bank, or from templates you type. Gemma is asked for sentences containing one anchor value; the anchor is then swapped for a slot and filled with every value, so the value is the only thing that changes.
- The examples are precomputed with the same int8 model (`tools/build_presets.py`).

## Layout

| Path | What |
|---|---|
| `index.html`, `app.js`, `style.css` | the page |
| `analysis.js` | geometry and probes (plain JavaScript, also runs in Node) |
| `worker.js` | runs both models off the main thread |
| `tools/` | ONNX export, preset builder, a Node check of the analysis |
| `experiment/` | the original probing experiment on how bge-small represents time |

## Run locally

```
python3 -m http.server 8765
```

then open http://localhost:8765/.

## The experiment behind it

`experiment/` holds the scripts and results of a test of the hypothesis that bge-small represents calendar years along a line. It does not: each year mostly has its own direction, with a weak shared time axis underneath, while months and weekdays form rings. See `experiment/out/time_probe.png`.
