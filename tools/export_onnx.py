"""Export bge-small-en-v1.5 to ONNX with one output: the [CLS] vector at every layer.

Output "cls_layers" has shape (batch, 13, 384); index 0 is the embedding layer,
12 the final layer. The int8-quantised file is what the web page loads.
"""
import numpy as np
import onnxruntime as ort
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoModel, AutoTokenizer

SRC, OUT = "BAAI/bge-small-en-v1.5", "models/bge-small-layers"


class ClsLayers(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.m = AutoModel.from_pretrained(SRC).eval()

    def forward(self, input_ids, attention_mask):
        hs = self.m(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True).hidden_states
        return torch.stack([h[:, 0] for h in hs], 1)


tok = AutoTokenizer.from_pretrained(SRC)
tok.save_pretrained(OUT)
net = ClsLayers()
net.m.config.save_pretrained(OUT)
b = tok(["The bridge was finished in 1987.", "It rained on Tuesday, all day long."], padding=True, return_tensors="pt")
B, T = torch.export.Dim("b", min=1, max=512), torch.export.Dim("t", min=3, max=512)
torch.onnx.export(net, (b["input_ids"], b["attention_mask"]), f"{OUT}/onnx/model.onnx",
                  input_names=["input_ids", "attention_mask"], output_names=["cls_layers"],
                  dynamic_shapes={"input_ids": {0: B, 1: T}, "attention_mask": {0: B, 1: T}},
                  opset_version=18, dynamo=True, external_data=False)
quantize_dynamic(f"{OUT}/onnx/model.onnx", f"{OUT}/onnx/model_quantized.onnx", weight_type=QuantType.QUInt8)

sents = [f"The bridge was finished in {y}." for y in range(1800, 2031, 5)] + \
        [f"We met on {d}." for d in ["Monday", "Tuesday", "Friday", "Sunday"]]
b = tok(sents, padding=True, return_tensors="pt")
ref = net(b["input_ids"], b["attention_mask"]).detach().numpy()
feed = {"input_ids": b["input_ids"].numpy(), "attention_mask": b["attention_mask"].numpy()}
for f in ["model.onnx", "model_quantized.onnx"]:
    o = ort.InferenceSession(f"{OUT}/onnx/{f}").run(None, feed)[0]
    cos = (o * ref).sum(-1) / np.linalg.norm(o, axis=-1) / np.linalg.norm(ref, axis=-1)
    # does the geometry between sentences survive? compare centred final-layer similarity matrices
    c = lambda x: (lambda u: u @ u.T)((x[:, -1] - x[:, -1].mean(0)) / np.linalg.norm(x[:, -1] - x[:, -1].mean(0), axis=1, keepdims=True))
    print(f, "min cos vs torch per layer:", np.round(cos.min(0), 3), "sim-matrix corr:", round(float(np.corrcoef(c(o).ravel(), c(ref).ravel())[0, 1]), 4))
