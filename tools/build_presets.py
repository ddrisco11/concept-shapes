"""Precompute the example datasets shown on the page.

Uses the same int8 ONNX model the browser loads, so examples and live builds
agree. Each preset is a float16 array (templates, values, 13 layers, 384).
"""
import json
import sys

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

SRC = sys.argv[1] if len(sys.argv) > 1 else "experiment/data"
MODEL = "models/bge-small-layers"
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
HOURS = ["12 AM"] + [f"{h} AM" for h in range(1, 12)] + ["12 PM"] + [f"{h} PM" for h in range(1, 12)]
PRESETS = [
    # id, name, kind, template file, slot, values, text shown in the values box, templates to keep
    ("years", "Years 1800–2030", "year", "year", "{YEAR}", [str(y) for y in range(1800, 2031)], "1800..2030", 12),
    ("decades", "Decades", "year", "year", "{YEAR}", [str(y) for y in range(1800, 2031, 10)], "1800..2030:10", 40),
    ("months", "Months", "month", "month", "{MONTH}", MONTHS, None, 40),
    ("weekdays", "Weekdays", "weekday", "weekday", "{DAY}", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], None, 40),
    ("hours", "Hours", "time", "hour", "{HOUR}", HOURS, None, 25),
    ("numbers", "Numbers 1–100", "number", "number", "{NUM}", [str(n) for n in range(1, 101)], "1..100", 20),
]

tok = AutoTokenizer.from_pretrained(MODEL)
sess = ort.InferenceSession(f"{MODEL}/onnx/model_quantized.onnx")
index = []
for pid, name, kind, tfile, slot, values, vtext, keep in PRESETS:
    templates = [t["template"].replace(slot, "{X}") for t in json.load(open(f"{SRC}/templates_{tfile}.json"))][:keep]
    sents = [t.replace("{X}", v) for t in templates for v in values]
    out = []
    for i in range(0, len(sents), 16):   # same batch size as the page
        b = tok(sents[i:i + 16], padding=True, truncation=True, max_length=64, return_tensors="np")
        out.append(sess.run(None, {"input_ids": b["input_ids"], "attention_mask": b["attention_mask"]})[0])
    arr = np.concatenate(out).astype("<f2")
    arr.tofile(f"presets/{pid}.bin")
    index.append({"id": pid, "name": name, "kind": kind, "note": "sentences by Gemma 3 270M", "values": values,
                  **({"valuesText": vtext} if vtext else {}), "templates": templates, "T": len(templates),
                  "V": len(values), "L": int(arr.shape[1]), "D": int(arr.shape[2]), "file": f"{pid}.bin"})
    print(pid, arr.shape, f"{arr.nbytes / 1e6:.1f} MB")
json.dump(index, open("presets/index.json", "w"))
