"""Audit bge-small-en-v1.5's representation of time.

Hypothesis (strong form): E(t) ~= b + t*v  for calendar years, with one
context-independent direction v. Cyclic units (months, weekdays, hours)
are checked for ring geometry instead.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from sklearn.decomposition import PCA
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(0)
YEARS = np.arange(1800, 2031)
R = {}


def ridge():
    return make_pipeline(StandardScaler(), Ridge(alpha=10.0))


def flat(X, t_idx, y_idx=None):
    """X: (T, Y, D) -> rows for the chosen templates / year indices, plus targets."""
    y_idx = np.arange(X.shape[1]) if y_idx is None else y_idx
    sub = X[np.ix_(t_idx, y_idx)]
    return sub.reshape(-1, X.shape[-1]), np.tile(YEARS[y_idx], len(t_idx)).astype(np.float64)


def template_folds(T, k=5):
    g = np.arange(T)
    return [(tr, te) for tr, te in GroupKFold(k).split(g, groups=rng.permutation(T))]


# ---------------------------------------------------------------- years
Z = np.load("data/emb_year.npz")
T, Y, L, D = Z["cls"].shape
folds = template_folds(T)
print(f"year: {T} templates x {Y} years, {L} layers, d={D}")

# 1. linear decoding at every layer, split by template
R["layers"] = {}
for pool in ["cls", "mean"]:
    rows = []
    for l in range(L):
        X = Z[pool][:, :, l].astype(np.float32)
        r2, mae = [], []
        for tr, te in folds:
            Xtr, ytr = flat(X, tr)
            Xte, yte = flat(X, te)
            p = ridge().fit(Xtr, ytr).predict(Xte)
            r2.append(r2_score(yte, p)); mae.append(mean_absolute_error(yte, p))
        rows.append({"layer": l, "r2": float(np.mean(r2)), "mae_years": float(np.mean(mae))})
        print(f"  [{pool}] layer {l:2d}  R2={rows[-1]['r2']:.4f}  MAE={rows[-1]['mae_years']:.2f}y")
    R["layers"][pool] = rows

# final embedding = L2-normalised last-layer CLS (what bge actually outputs)
E = Z["cls"][:, :, -1].astype(np.float32)
E /= np.linalg.norm(E, axis=-1, keepdims=True)

# out-of-fold predictions for every (template, year)
pred = np.zeros((T, Y))
for tr, te in folds:
    Xtr, ytr = flat(E, tr)
    pred[te] = ridge().fit(Xtr, ytr).predict(flat(E, te)[0]).reshape(len(te), Y)
err = np.abs(pred - YEARS)
slopes = np.array([np.polyfit(YEARS, pred[i], 1)[0] for i in range(T)])
R["final_linear"] = {
    "r2": float(r2_score(np.tile(YEARS, T), pred.ravel())),
    "mae_years": float(err.mean()),
    "median_abs_err": float(np.median(err)),
    "per_template_slope_mean": float(slopes.mean()),
    "per_template_slope_sd": float(slopes.std()),
    "mae_by_era": {f"{a}-{a + 49}": float(err[:, (YEARS >= a) & (YEARS < a + 50)].mean())
                   for a in range(1800, 2001, 50)},
}
print("final linear:", json.dumps(R["final_linear"], indent=1))

# 4. linear vs nonlinear, plus held-out year blocks (unseen templates AND unseen years)
tr, te = folds[0]
idx = lambda m: np.where(m)[0]
splits = {
    "all years": (np.arange(Y), np.arange(Y)),
    "interpolate 1900-1949": (idx((YEARS < 1900) | (YEARS >= 1950)), idx((YEARS >= 1900) & (YEARS < 1950))),
    "extrapolate 1980-2030": (idx(YEARS < 1980), idx(YEARS >= 1980)),
}
models = {
    "ridge": ridge,
    "mlp": lambda: make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(256, 256), early_stopping=True,
                                                               max_iter=200, random_state=0)),
    "kernel_rbf": lambda: make_pipeline(StandardScaler(), KernelRidge(kernel="rbf", alpha=1.0, gamma=1 / D)),
}
R["linear_vs_nonlinear"] = {}
for sname, (ytr_i, yte_i) in splits.items():
    Xtr, ytr = flat(E, tr, ytr_i)
    Xte, yte = flat(E, te, yte_i)
    mu, sd = ytr.mean(), ytr.std()
    out = {}
    for mname, mk in models.items():
        if mname == "kernel_rbf":
            s = rng.choice(len(Xtr), min(8000, len(Xtr)), replace=False)
            m = mk().fit(Xtr[s], (ytr[s] - mu) / sd)
        else:
            m = mk().fit(Xtr, (ytr - mu) / sd)
        p = m.predict(Xte) * sd + mu
        out[mname] = {"r2": float(r2_score(yte, p)), "mae_years": float(mean_absolute_error(yte, p))}
    R["linear_vs_nonlinear"][sname] = out
    print(sname, json.dumps(out))

# 2. direction consistency of +10-year displacements
dec = np.arange(0, Y - 10, 10)                       # 1800, 1810, ..., 2020
steps = E[:, dec + 10] - E[:, dec]                   # (T, S, D)
U = steps / np.linalg.norm(steps, axis=-1, keepdims=True)
s = U.sum(0)
across_templates = ((s ** 2).sum(-1) - T) / (T * (T - 1))   # mean pairwise cos, per decade
M = steps.mean(0)
Mu = M / np.linalg.norm(M, axis=-1, keepdims=True)
step_cos = Mu @ Mu.T                                 # decade step vs decade step
g = E[:, -1].mean(0) - E[:, 0].mean(0); g /= np.linalg.norm(g)
off = step_cos[~np.eye(len(dec), dtype=bool)]
R["direction"] = {
    "pairwise_cos_across_templates_same_decade": {"mean": float(across_templates.mean()),
                                                  "min": float(across_templates.min()),
                                                  "max": float(across_templates.max())},
    "cos_between_decade_steps": {"adjacent": float(np.diag(step_cos, 1).mean()),
                                 "5_decades_apart": float(np.diag(step_cos, 5).mean()),
                                 "all_offdiag_mean": float(off.mean())},
    "cos_decade_step_vs_global_1800_to_2030": {int(YEARS[d]): float(c) for d, c in zip(dec, Mu @ g)},
}
print("direction:", json.dumps({k: v for k, v in R["direction"].items() if "global" not in k}, indent=1))

# 3. geometry of the year trajectory (template offset removed)
C = E - E.mean(1, keepdims=True)
Mc = C.mean(0)                                        # (Y, D) mean trajectory
pca = PCA(10).fit(Mc)
traj = pca.transform(Mc)
speed = np.linalg.norm(np.diff(Mc, axis=0), axis=1)
R["geometry"] = {
    "pca_var_explained_of_mean_trajectory": [float(v) for v in pca.explained_variance_ratio_[:6]],
    "share_of_year_variance_common_to_all_templates": float((Mc ** 2).sum() * T / (C ** 2).sum()),
    "pc1_vs_year_r2": float(np.corrcoef(traj[:, 0], YEARS)[0, 1] ** 2),
    "end_to_end_over_path_length": float(np.linalg.norm(Mc[-1] - Mc[0]) / speed.sum()),
    "mean_1yr_step_norm_by_era": {f"{a}-{a + 49}": float(speed[(YEARS[:-1] >= a) & (YEARS[:-1] < a + 50)].mean())
                                  for a in range(1800, 2001, 50)},
}
print("geometry:", json.dumps(R["geometry"], indent=1))

# naive "which dimensions?" view
Xtr, ytr = flat(E, tr); Xte, yte = flat(E, te)
corr = np.array([np.corrcoef(C[:, :, d].ravel(), np.tile(YEARS, T))[0, 1] for d in range(D)])
top = np.argsort(-np.abs(corr))[:10]
R["coordinates"] = {"top10_by_abs_corr": {int(d): float(corr[d]) for d in top}, "lasso": []}
sc = StandardScaler().fit(Xtr)
for a in [0.3, 0.1, 0.03, 0.01]:
    m = Lasso(alpha=a, max_iter=5000).fit(sc.transform(Xtr), (ytr - ytr.mean()) / ytr.std())
    p = m.predict(sc.transform(Xte)) * ytr.std() + ytr.mean()
    R["coordinates"]["lasso"].append({"alpha": a, "nonzero_dims": int((m.coef_ != 0).sum()),
                                      "r2": float(r2_score(yte, p))})
print("coordinates:", json.dumps(R["coordinates"]))

pieces = json.load(open("data/year_wordpieces.json"))
multi = np.array([pieces[str(y)] > 1 for y in YEARS])
R["tokenization"] = {"n_multi_piece_years": int(multi.sum()),
                     "mae_single_piece": float(err[:, ~multi].mean()),
                     "mae_multi_piece": float(err[:, multi].mean()) if multi.any() else None}
print("tokenization:", R["tokenization"])

# ---------------------------------------------------------------- cyclic units
cyc = {}
R["cyclic"] = {}
for kind in ["month", "weekday", "hour"]:
    X = np.load(f"data/emb_{kind}.npz")["cls"][:, :, -1].astype(np.float32)
    X /= np.linalg.norm(X, axis=-1, keepdims=True)
    Tk, V, _ = X.shape
    Mk = (X - X.mean(1, keepdims=True)).mean(0)
    adj = np.linalg.norm(np.diff(Mk, axis=0), axis=1).mean()
    th = 2 * np.pi * np.arange(V) / V
    targets = {"scalar": np.arange(V, dtype=float)[:, None], "circular": np.c_[np.cos(th), np.sin(th)]}
    dec_r2 = {}
    for name, tg in targets.items():
        r2 = []
        for a, b in template_folds(Tk):
            m = ridge().fit(X[a].reshape(-1, D), np.tile(tg, (len(a), 1)))
            r2.append(r2_score(np.tile(tg, (len(b), 1)), m.predict(X[b].reshape(-1, D))))
        dec_r2[name] = float(np.mean(r2))
    R["cyclic"][kind] = {
        "templates": Tk,
        "wraparound_dist_over_adjacent_dist": float(np.linalg.norm(Mk[-1] - Mk[0]) / adj),
        "max_dist_over_adjacent_dist": float(max(np.linalg.norm(Mk[i] - Mk[j]) for i in range(V) for j in range(V)) / adj),
        "heldout_template_r2": dec_r2,
    }
    cyc[kind] = PCA(2).fit_transform(Mk)
    print(kind, json.dumps(R["cyclic"][kind]))
adj = speed.mean()
R["cyclic"]["year_for_reference"] = {"wraparound_dist_over_adjacent_dist": float(np.linalg.norm(Mc[-1] - Mc[0]) / adj)}

json.dump(R, open("out/results.json", "w"), indent=1)

# ---------------------------------------------------------------- figure
BLUE, ORANGE, INK, MUTED, GRID, SURF = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
seq = LinearSegmentedColormap.from_list("seq", ["#cde2fb", "#3987e5", "#184f95"])
div = LinearSegmentedColormap.from_list("div", ["#d64545", "#d9d8d3", "#2a78d6"])
plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "text.color": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left",
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF})
fig, ax = plt.subplots(2, 4, figsize=(19, 9.5))

a = ax[0, 0]
for pool, col, lab in [("cls", BLUE, "[CLS]"), ("mean", ORANGE, "mean-pool")]:
    v = [r["r2"] for r in R["layers"][pool]]
    a.plot(range(L), v, color=col, lw=2, marker="o", ms=4, label=lab)
a.set(title="Linear decodability of year, by layer", xlabel="layer (0 = embeddings)", ylabel="R², unseen templates")
a.legend(frameon=False, loc="lower right")

a = ax[0, 1]
a.plot(YEARS, YEARS, color=MUTED, lw=1, ls="--")
a.fill_between(YEARS, np.percentile(pred, 10, 0), np.percentile(pred, 90, 0), color=BLUE, alpha=0.2, lw=0)
a.plot(YEARS, pred.mean(0), color=BLUE, lw=2)
a.annotate("y = x", (1830, 1830), xytext=(8, -12), textcoords="offset points", color=MUTED)
a.set(title="Ridge probe on final embedding (unseen templates)", xlabel="true year",
      ylabel="predicted year (mean, 10–90% band)")

a = ax[0, 2]
a.plot(YEARS, err.mean(0), color=BLUE, lw=2)
a.set(title="Probe error by year", xlabel="true year", ylabel="mean abs error (years)", ylim=(0, None))

a = ax[0, 3]
sca = a.scatter(traj[:, 0], traj[:, 1], c=YEARS, cmap=seq, s=14)
for y in [1800, 1850, 1900, 1950, 2000, 2030]:
    a.annotate(str(y), traj[y - 1800, :2], xytext=(5, 5), textcoords="offset points", fontsize=8)
ev = pca.explained_variance_ratio_
a.set(title="Mean year trajectory (template offset removed)", xlabel=f"PC1 ({ev[0]:.0%})", ylabel=f"PC2 ({ev[1]:.0%})")
fig.colorbar(sca, ax=a, label="year", fraction=0.046)

a = ax[1, 0]
im = a.imshow(step_cos, cmap=div, vmin=-1, vmax=1, origin="lower",
              extent=[1800, 1800 + 10 * len(dec), 1800, 1800 + 10 * len(dec)])
a.grid(False)
a.set(title="Cosine between +10-year displacement vectors", xlabel="decade start", ylabel="decade start")
fig.colorbar(im, ax=a, label="cosine", fraction=0.046)

for a, kind, labels in [
    (ax[1, 1], "month", ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]),
    (ax[1, 2], "weekday", ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]),
    (ax[1, 3], "hour", ["12a"] + [f"{h}a" for h in range(1, 12)] + ["12p"] + [f"{h}p" for h in range(1, 12)]),
]:
    P = cyc[kind]
    loop = np.r_[P, P[:1]]
    a.plot(loop[:-1, 0], loop[:-1, 1], color=BLUE, lw=1.5)
    a.plot(loop[-2:, 0], loop[-2:, 1], color=BLUE, lw=1.5, ls=":")
    a.scatter(P[:, 0], P[:, 1], color=BLUE, s=28, edgecolor=SURF, linewidth=1.5, zorder=3)
    for (x, y), lab in zip(P, labels):
        a.annotate(lab, (x, y), xytext=(5, 5), textcoords="offset points", fontsize=8)
    a.set(title=f"{kind.capitalize()}s: PCA of class means (dotted = wraparound)", xlabel="PC1", ylabel="PC2")
    a.set_aspect("equal", adjustable="datalim")

fig.suptitle("bge-small-en-v1.5: how is time represented?", x=0.01, ha="left", fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig("out/time_probe.png", dpi=130)
print("wrote out/results.json, out/time_probe.png")
