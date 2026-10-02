"""Fill template slots programmatically and embed with bge-small-en-v1.5.

Saves, for every (template, value), the [CLS] vector and the mean-pooled
vector at every layer (0 = embedding layer, 12 = final).
Output arrays: (n_templates, n_values, 13, 384) float16.
"""
import json

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

MODEL = "BAAI/bge-small-en-v1.5"
DEV = "mps" if torch.backends.mps.is_available() else "cpu"

VALUES = {
    "year": [str(y) for y in range(1800, 2031)],
    "month": ["January", "February", "March", "April", "May", "June", "July", "August",
              "September", "October", "November", "December"],
    "weekday": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
    "hour": ["12 AM"] + [f"{h} AM" for h in range(1, 12)] + ["12 PM"] + [f"{h} PM" for h in range(1, 12)],
}
SLOT = {"year": "{YEAR}", "month": "{MONTH}", "weekday": "{DAY}", "hour": "{HOUR}"}

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModel.from_pretrained(MODEL).to(DEV).eval()


@torch.no_grad()
def embed(sents, bs=256):
    cls, mean = [], []
    for i in range(0, len(sents), bs):
        b = tok(sents[i:i + bs], padding=True, truncation=True, max_length=64, return_tensors="pt").to(DEV)
        hs = torch.stack(model(**b, output_hidden_states=True).hidden_states, 1)  # B, L, T, D
        m = b["attention_mask"][:, None, :, None].float()
        cls.append(hs[:, :, 0].cpu().numpy().astype(np.float16))
        mean.append(((hs * m).sum(2) / m.sum(2)).cpu().numpy().astype(np.float16))
        if (i // bs) % 20 == 0:
            print(f"  {i}/{len(sents)}", flush=True)
    return np.concatenate(cls), np.concatenate(mean)


for kind, vals in VALUES.items():
    templates = [t["template"] for t in json.load(open(f"data/templates_{kind}.json"))]
    sents = [t.replace(SLOT[kind], v) for t in templates for v in vals]
    print(kind, len(templates), "templates x", len(vals), "values =", len(sents))
    cls, mean = embed(sents)
    shape = (len(templates), len(vals), cls.shape[1], cls.shape[2])
    np.savez(f"data/emb_{kind}.npz", cls=cls.reshape(shape), mean=mean.reshape(shape))

# how many wordpieces each year string becomes (single-token years are a lexical-lookup confound)
pieces = {y: len(tok.tokenize(y)) for y in VALUES["year"]}
json.dump(pieces, open("data/year_wordpieces.json", "w"))
print("multi-piece years:", [y for y, n in pieces.items() if n > 1])
