#!/usr/bin/env python3
"""
CSEN3011 Artificial Neural Networks - Project 28
Predictive-Maintenance Feature Learning with Autoencoder (AI4I 2020, UCI #601)

Assigned architecture : Autoencoder + MLP
Assigned optimisation : Mini-Batch Gradient Descent
Evaluation required   : Reconstruction loss, F1, ROC-AUC, latent-dimension study

Everything (forward pass, backpropagation, mini-batch gradient descent, early
stopping) is implemented from scratch in NumPy, so the code shows the ANN
concepts directly.  Only numpy / pandas / scikit-learn / matplotlib are needed.

HOW TO RUN
    pip install numpy pandas scikit-learn matplotlib
    python project28_ai4i_autoencoder_mlp.py                 # auto-downloads dataset
    python project28_ai4i_autoencoder_mlp.py --csv ai4i2020.csv   # use a local copy
    python project28_ai4i_autoencoder_mlp.py --quick         # fast smoke test

Dataset (original source):
    https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset
If auto-download fails, download the zip from that page, extract ai4i2020.csv
and pass it with --csv.

Outputs go to ./results_project28/ (figures/, tables/, run_info.json,
results_summary.md).  Use ONLY numbers produced by a run on the real dataset in
your report.  The --synthetic flag exists only to test that the code runs; it
writes to a different folder and its numbers are meaningless.
"""
import argparse
import io
import json
import os
import platform
import sys
import time
import urllib.request
import zipfile

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

plt.rcParams.update({"font.family": "serif", "font.size": 11, "figure.dpi": 120})

# --------------------------------------------------------------------------
# 1. DATA
# --------------------------------------------------------------------------
UCI_URL = ("https://archive.ics.uci.edu/static/public/601/"
           "ai4i+2020+predictive+maintenance+dataset.zip")
NUMERIC = ["Air temperature [K]", "Process temperature [K]",
           "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]
LEAKY = ["TWF", "HDF", "PWF", "OSF", "RNF"]       # failure-mode flags (define the target)
IDS = ["UDI", "Product ID"]


def load_ai4i(csv=None):
    for c in [csv, "ai4i2020.csv", os.path.join("data", "ai4i2020.csv")]:
        if c and os.path.exists(c):
            print(f"[data] loading local file {c}")
            return pd.read_csv(c)
    print("[data] downloading from UCI ...")
    try:
        req = urllib.request.Request(UCI_URL, headers={"User-Agent": "Mozilla/5.0"})
        raw = urllib.request.urlopen(req, timeout=90).read()
        z = zipfile.ZipFile(io.BytesIO(raw))
        name = [n for n in z.namelist() if n.lower().endswith(".csv")][0]
        df = pd.read_csv(z.open(name))
        df.to_csv("ai4i2020.csv", index=False)
        return df
    except Exception as e:  # noqa
        sys.exit("Could not download the dataset (" + str(e) + ").\n"
                 "Download it manually from\n  https://archive.ics.uci.edu/dataset/601/"
                 "ai4i+2020+predictive+maintenance+dataset\nextract ai4i2020.csv and run:\n"
                 "  python project28_ai4i_autoencoder_mlp.py --csv ai4i2020.csv")


def synthetic_ai4i(n=10000, seed=0):
    """Rough imitation of AI4I - ONLY for testing that the code runs."""
    r = np.random.default_rng(seed)
    typ = r.choice(["L", "M", "H"], n, p=[.6, .3, .1])
    air = r.normal(300, 2, n)
    proc = air + 10 + r.normal(0, 1, n)
    rpm = 1538 + r.exponential(170, n) - 120 + r.normal(0, 40, n)
    torque = np.clip(40 - 0.03 * (rpm - 1538) + r.normal(0, 6, n), 3, None)
    wear = r.integers(0, 254, n).astype(float)
    power = torque * rpm * 2 * np.pi / 60
    lim = pd.Series(typ).map({"L": 11000, "M": 12000, "H": 13000}).values
    hdf = ((proc - air) < 8.6) & (rpm < 1380)
    pwf = (power < 3500) | (power > 9000)
    osf = wear * torque > lim
    twf = (wear > 200) & (r.random(n) < 0.08)
    rnf = r.random(n) < 0.001
    fail = (hdf | pwf | osf | twf | rnf).astype(int)
    return pd.DataFrame({"UDI": np.arange(1, n + 1), "Product ID": [f"X{i}" for i in range(n)],
                         "Type": typ, NUMERIC[0]: air, NUMERIC[1]: proc, NUMERIC[2]: rpm,
                         NUMERIC[3]: torque, NUMERIC[4]: wear, "Machine failure": fail,
                         "TWF": twf.astype(int), "HDF": hdf.astype(int), "PWF": pwf.astype(int),
                         "OSF": osf.astype(int), "RNF": rnf.astype(int)})


def audit_and_prepare(df):
    """Clean the raw table; return feature DataFrame X, target y and an audit dict."""
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    # tolerate small naming differences between CSV versions (units, case)
    canon = {"udi": "UDI", "product id": "Product ID", "type": "Type", "machine failure": "Machine failure",
             "air temperature": NUMERIC[0], "process temperature": NUMERIC[1], "rotational speed": NUMERIC[2],
             "torque": NUMERIC[3], "tool wear": NUMERIC[4], "twf": "TWF", "hdf": "HDF", "pwf": "PWF",
             "osf": "OSF", "rnf": "RNF"}
    ren = {}
    for c in df.columns:
        key = c.lower().split("[")[0].strip()
        if key in canon:
            ren[c] = canon[key]
    df = df.rename(columns=ren)
    audit = {"raw_rows": int(len(df)), "raw_columns": list(df.columns)}
    audit["missing_values_total"] = int(df.isna().sum().sum())
    feat_cols = ["Type"] + NUMERIC
    audit["duplicate_rows_on_features_and_target"] = int(df.duplicated(subset=feat_cols + ["Machine failure"]).sum())
    audit["duplicate_ids"] = int(df["UDI"].duplicated().sum()) if "UDI" in df else None
    df = df.dropna(subset=feat_cols + ["Machine failure"]).drop_duplicates(subset=["UDI"] if "UDI" in df else None)
    q1, q3 = df[NUMERIC].quantile(.25), df[NUMERIC].quantile(.75)
    iqr = q3 - q1
    out = ((df[NUMERIC] < q1 - 1.5 * iqr) | (df[NUMERIC] > q3 + 1.5 * iqr)).sum()
    audit["iqr_outliers_per_feature (kept)"] = {k: int(v) for k, v in out.items()}
    audit["dropped_identifier_columns"] = [c for c in IDS if c in df.columns]
    audit["dropped_leaky_columns"] = [c for c in LEAKY if c in df.columns]
    y = df["Machine failure"].astype(int).values
    X = pd.get_dummies(df[feat_cols], columns=["Type"], dtype=float)
    X = X[NUMERIC + ["Type_L", "Type_M", "Type_H"]]
    audit["final_rows"] = int(len(X))
    audit["final_feature_names"] = list(X.columns)
    audit["class_counts"] = {"no_failure(0)": int((y == 0).sum()), "failure(1)": int((y == 1).sum())}
    audit["failure_rate_percent"] = round(100 * y.mean(), 3)
    return df, X, y, audit


_SPLITS = {}


def get_split(X, y, seed):
    """Stratified 70/15/15 split; scaler fitted on TRAIN only (no leakage)."""
    if seed in _SPLITS:
        return _SPLITS[seed]
    Xa, Xte, ya, yte = train_test_split(X.values, y, test_size=0.15, stratify=y, random_state=seed)
    Xtr, Xva, ytr, yva = train_test_split(Xa, ya, test_size=0.15 / 0.85, stratify=ya, random_state=seed)
    sc = StandardScaler().fit(Xtr)
    out = tuple(sc.transform(a) for a in (Xtr, Xva, Xte)) + (ytr, yva, yte)
    _SPLITS[seed] = out
    return out


# --------------------------------------------------------------------------
# 2. NEURAL NETWORK FROM SCRATCH (forward pass + backpropagation)
# --------------------------------------------------------------------------
def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))


ACTS = {
    "relu": (lambda z: np.maximum(0.0, z), lambda z, a: (z > 0).astype(z.dtype)),
    "tanh": (np.tanh, lambda z, a: 1.0 - a ** 2),
    "sigmoid": (_sigmoid, lambda z, a: a * (1.0 - a)),
    "linear": (lambda z: z, lambda z, a: np.ones_like(z)),
}


class Dense:
    def __init__(self, n_in, n_out, act, rng):
        self.n_in, self.n_out, self.act = n_in, n_out, act
        self.f, self.df = ACTS[act]
        if act == "relu":                       # He initialisation
            self.W = rng.normal(0.0, np.sqrt(2.0 / n_in), (n_in, n_out))
        else:                                   # Glorot / Xavier initialisation
            lim = np.sqrt(6.0 / (n_in + n_out))
            self.W = rng.uniform(-lim, lim, (n_in, n_out))
        self.b = np.zeros(n_out)

    def forward(self, x):
        self.x = x
        self.z = x @ self.W + self.b
        self.a = self.f(self.z)
        return self.a

    def backward(self, g, from_z=False):
        gz = g if from_z else g * self.df(self.z, self.a)
        self.gW = self.x.T @ gz
        self.gb = gz.sum(0)
        return gz @ self.W.T

    def copy(self):
        c = Dense.__new__(Dense)
        c.__dict__.update({k: v for k, v in self.__dict__.items() if k in ("n_in", "n_out", "act", "f", "df")})
        c.W, c.b = self.W.copy(), self.b.copy()
        return c


class Net:
    def __init__(self, layers):
        self.layers = layers
        self.vel = None

    @staticmethod
    def build(sizes, acts, rng):
        return Net([Dense(sizes[i], sizes[i + 1], acts[i], rng) for i in range(len(acts))])

    def forward(self, x):
        for l in self.layers:
            x = l.forward(x)
        return x

    def backward(self, g, from_z_last=False):
        for i, l in enumerate(reversed(self.layers)):
            g = l.backward(g, from_z=(from_z_last and i == 0))

    def step(self, lr, momentum=0.0, l2=0.0):
        """One (mini-batch) gradient-descent update.  momentum=0 -> plain GD."""
        if momentum > 0 and self.vel is None:
            self.vel = [(np.zeros_like(l.W), np.zeros_like(l.b)) for l in self.layers]
        for i, l in enumerate(self.layers):
            gW = l.gW + l2 * l.W
            if momentum > 0:
                vW, vb = self.vel[i]
                vW = momentum * vW - lr * gW
                vb = momentum * vb - lr * l.gb
                self.vel[i] = (vW, vb)
                l.W += vW
                l.b += vb
            else:
                l.W -= lr * gW
                l.b -= lr * l.gb

    def n_params(self):
        return int(sum(l.W.size + l.b.size for l in self.layers))

    def get(self):
        return [(l.W.copy(), l.b.copy()) for l in self.layers]

    def set(self, w):
        for l, (W, b) in zip(self.layers, w):
            l.W, l.b = W.copy(), b.copy()

    def copy(self):
        return Net([l.copy() for l in self.layers])


def loss_and_grad(task, out, y, cw=None):
    """Returns (loss, dLoss/dOut, grad_is_wrt_preactivation)."""
    N = out.shape[0]
    if task == "mse":                                   # autoencoder reconstruction
        d = out - y
        return float(np.mean(d ** 2)), 2.0 * d / d.size, False
    p = np.clip(out, 1e-7, 1 - 1e-7)                    # weighted binary cross-entropy
    w = np.ones_like(y) if cw is None else np.where(y == 1, cw[1], cw[0])
    loss = float(np.mean(w * -(y * np.log(p) + (1 - y) * np.log(1 - p))))
    return loss, w * (out - y) / N, True               # sigmoid+BCE: dL/dz = w(p-y)/N


def fit(net, Xtr, Ytr, Xva, Yva, task, lr, bs, epochs, rng, patience=20,
        momentum=0.0, l2=0.0, cw=None, min_delta=1e-6):
    """Mini-batch gradient descent with backpropagation and early stopping."""
    n = len(Xtr)
    bs = n if (bs is None or bs >= n) else bs          # bs == n  ->  batch gradient descent
    hist = {"train": [], "val": [], "train_acc": [], "val_acc": []}
    best, best_w, best_ep, wait = np.inf, net.get(), 0, 0
    for ep in range(epochs):
        idx = rng.permutation(n) if bs < n else np.arange(n)
        for s in range(0, n, bs):
            b = idx[s:s + bs]
            out = net.forward(Xtr[b])
            _, g, fz = loss_and_grad(task, out, Ytr[b], cw)
            net.backward(g, fz)
            net.step(lr, momentum, l2)
        otr, ova = net.forward(Xtr), net.forward(Xva)
        ltr = loss_and_grad(task, otr, Ytr, cw)[0]
        lva = loss_and_grad(task, ova, Yva, cw)[0]
        hist["train"].append(ltr)
        hist["val"].append(lva)
        if task == "bce":
            hist["train_acc"].append(float(np.mean((otr > .5) == (Ytr > .5))))
            hist["val_acc"].append(float(np.mean((ova > .5) == (Yva > .5))))
        if lva < best - min_delta:
            best, best_w, best_ep, wait = lva, net.get(), ep, 0
        else:
            wait += 1
            if wait >= patience:
                break
    net.set(best_w)
    hist["best_epoch"] = best_ep + 1
    hist["epochs_run"] = len(hist["train"])
    return hist


# --------------------------------------------------------------------------
# 3. MODELS
# --------------------------------------------------------------------------
def build_autoencoder(d, k, rng, h=6):
    enc = Net.build([d, h, k], ["tanh", "linear"], rng)
    dec = Net.build([k, h, d], ["tanh", "linear"], rng)
    return enc, dec


def build_head(d_in, hidden, rng):
    sizes = [d_in] + list(hidden) + [1]
    acts = ["relu"] * len(hidden) + ["sigmoid"]
    return Net.build(sizes, acts, rng)


def best_threshold(y, p):
    ts = np.linspace(0.02, 0.98, 97)
    f1s = [f1_score(y, (p >= t).astype(int), zero_division=0) for t in ts]
    return float(ts[int(np.argmax(f1s))])


def clf_metrics(y, p, thr):
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"accuracy": float((tn + tp) / len(y)),
            "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "f1": float(f1_score(y, pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y, p)),
            "pr_auc": float(average_precision_score(y, p)),
            "threshold": thr, "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


BASE = dict(mode="frozen", k=4, ae_lr=0.1, ae_bs=64, ae_epochs=150, head_hidden=(16, 8),
            head_lr=0.05, head_bs=64, head_epochs=150, pos_weight=True, momentum=0.0)
_AE_CACHE = {}


def get_ae(cfg, seed, split):
    key = (seed, cfg["k"], cfg["ae_lr"], cfg["ae_bs"], cfg["ae_epochs"], cfg["momentum"])
    if key in _AE_CACHE:
        return _AE_CACHE[key]
    Xtr, Xva, _, _, _, _ = split
    rng = np.random.default_rng(seed)
    enc, dec = build_autoencoder(Xtr.shape[1], cfg["k"], rng)
    ae = Net(enc.layers + dec.layers)
    # unsupervised: the autoencoder is trained to reproduce its own input
    hist = fit(ae, Xtr, Xtr, Xva, Xva, "mse", cfg["ae_lr"], cfg["ae_bs"], cfg["ae_epochs"], rng,
               momentum=cfg["momentum"])
    _AE_CACHE[key] = (ae, len(enc.layers), hist)
    return _AE_CACHE[key]


def run(cfg, seed, X, y):
    t0 = time.time()
    split = get_split(X, y, seed)
    Xtr, Xva, Xte, ytr, yva, yte = split
    rng = np.random.default_rng(seed + 1000)
    res = {"seed": seed, "mode": cfg["mode"], "k": cfg["k"] if cfg["mode"] != "raw" else np.nan}
    cw = None
    if cfg["pos_weight"]:                                  # "balanced" class weights, mean weight = 1
        n, npos = len(ytr), ytr.sum()
        cw = (n / (2.0 * (n - npos)), n / (2.0 * npos))
    Ytr, Yva = ytr.reshape(-1, 1).astype(float), yva.reshape(-1, 1).astype(float)

    if cfg["mode"] == "raw":
        head = build_head(Xtr.shape[1], cfg["head_hidden"], rng)
        h = fit(head, Xtr, Ytr, Xva, Yva, "bce", cfg["head_lr"], cfg["head_bs"], cfg["head_epochs"], rng,
                momentum=cfg["momentum"], cw=cw)
        pva, pte = head.forward(Xva).ravel(), head.forward(Xte).ravel()
        res.update(recon_mse_test=np.nan, ae_params=0, head_params=head.n_params(), ae_best_epoch=np.nan)
        hist_ae = None
    else:
        ae, n_enc, hist_ae = get_ae(cfg, seed, split)
        enc = Net(ae.layers[:n_enc])
        rec_te = ae.forward(Xte)
        res["recon_mse_test"] = float(np.mean((rec_te - Xte) ** 2))
        res["recon_mse_train"] = float(np.mean((ae.forward(Xtr) - Xtr) ** 2))
        err_te = np.mean((rec_te - Xte) ** 2, axis=1)
        res["recon_err_auc"] = float(roc_auc_score(yte, err_te))
        res["ae_params"] = ae.n_params()
        res["ae_best_epoch"] = hist_ae["best_epoch"]
        if cfg["mode"] == "frozen":                        # encoder fixed, classifier on latent code
            Ztr, Zva, Zte = enc.forward(Xtr), enc.forward(Xva), enc.forward(Xte)
            head = build_head(cfg["k"], cfg["head_hidden"], rng)
            h = fit(head, Ztr, Ytr, Zva, Yva, "bce", cfg["head_lr"], cfg["head_bs"], cfg["head_epochs"], rng,
                    momentum=cfg["momentum"], cw=cw)
            pva, pte = head.forward(Zva).ravel(), head.forward(Zte).ravel()
            res["head_params"] = head.n_params()
        else:                                              # "finetune": encoder initialised from AE, trained end-to-end
            head = build_head(cfg["k"], cfg["head_hidden"], rng)
            joint = Net([l.copy() for l in enc.layers] + head.layers)
            h = fit(joint, Xtr, Ytr, Xva, Yva, "bce", cfg["head_lr"], cfg["head_bs"], cfg["head_epochs"], rng,
                    momentum=cfg["momentum"], cw=cw)
            pva, pte = joint.forward(Xva).ravel(), joint.forward(Xte).ravel()
            res["head_params"] = joint.n_params()
        res["_z_te"] = enc.forward(Xte)
        res["_err_te"] = err_te
    thr = best_threshold(yva, pva)                         # threshold picked on VALIDATION data only
    res.update(clf_metrics(yte, pte, thr))
    res.update(head_best_epoch=h["best_epoch"], head_epochs_run=h["epochs_run"], time_s=round(time.time() - t0, 1))
    res.update(_hist_head=h, _hist_ae=hist_ae, _y_te=yte, _p_te=pte)
    return res


# --------------------------------------------------------------------------
# 4. PLOTTING HELPERS
# --------------------------------------------------------------------------
def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def draw_net(ax, sizes, names, color, title):
    maxn = max(sizes)
    xs = np.linspace(0, 1, len(sizes))
    pos = [[(xs[i], (j - (n - 1) / 2) / max(maxn - 1, 1) * 0.9) for j in range(n)] for i, n in enumerate(sizes)]
    for i in range(len(sizes) - 1):
        for p in pos[i]:
            for q in pos[i + 1]:
                ax.plot([p[0], q[0]], [p[1], q[1]], lw=0.3, color="0.75", zorder=1)
    for i, col in enumerate(pos):
        x, yv = zip(*col)
        ax.scatter(x, yv, s=170, zorder=2, ec="k", fc=color[i])
        ax.text(xs[i], -0.62, f"{names[i]}\n({sizes[i]})", ha="center", va="top", fontsize=9)
    ax.set_title(title, fontsize=12)
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.95, 0.6)
    ax.axis("off")


def fig_architecture(d, k, hidden, path):
    fig, axs = plt.subplots(2, 1, figsize=(10, 8.5))
    draw_net(axs[0], [d, 6, k, 6, d],
             ["Input\n(scaled\nfeatures)", "Hidden 1\ntanh", f"Latent z\nlinear", "Hidden 2\ntanh", "Output\nlinear\n(reconstruction)"],
             ["#cfe8ff", "#ffe9b3", "#ffb3b3", "#ffe9b3", "#cfe8ff"],
             f"(a) Autoencoder: {d}-6-{k}-6-{d}   (encoder | decoder), trained with MSE loss")
    sizes = [k] + list(hidden) + [1]
    names = ["Latent z\n(input)"] + [f"Hidden {i + 1}\nReLU" for i in range(len(hidden))] + ["Output\nsigmoid\nP(failure)"]
    cols = ["#ffb3b3"] + ["#ffe9b3"] * len(hidden) + ["#c8f0c8"]
    draw_net(axs[1], sizes, names, cols, f"(b) Classification MLP on the latent code: {'-'.join(map(str, sizes))}, weighted BCE loss")
    save(fig, path)


def agg_table(df, keys):
    cols = ["recon_mse_test", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    g = df.groupby(keys, sort=False)[cols]
    m, s = g.mean(), g.std(ddof=0)
    out = pd.DataFrame(index=m.index)
    for c in cols:
        out[c] = [("-" if np.isnan(a) else f"{a:.4f} ± {b:.4f}") for a, b in zip(m[c], s[c])]
    return out.reset_index(), m.reset_index(), s.reset_index()


# --------------------------------------------------------------------------
# 5. MAIN
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None)
    ap.add_argument("--outdir", default="results_project28")
    ap.add_argument("--seeds", type=int, default=3, help="number of random splits/initialisations")
    ap.add_argument("--base-seed", type=int, default=42)
    ap.add_argument("--quick", action="store_true", help="few epochs, 1 seed (smoke test)")
    ap.add_argument("--synthetic", action="store_true", help="fake data to test the code ONLY")
    a = ap.parse_args()

    if a.synthetic:
        a.outdir += "_SYNTHETIC_TEST_ONLY"
        print("!" * 70, "\nSYNTHETIC DATA: numbers are meaningless, do NOT use them in the report\n" + "!" * 70)
    if a.quick:
        a.seeds = 1
        BASE.update(ae_epochs=15, head_epochs=15)
    fig_dir, tab_dir = os.path.join(a.outdir, "figures"), os.path.join(a.outdir, "tables")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(tab_dir, exist_ok=True)
    seeds = [a.base_seed + i for i in range(a.seeds)]

    raw = synthetic_ai4i() if a.synthetic else load_ai4i(a.csv)
    df, X, y, audit = audit_and_prepare(raw)
    print(json.dumps(audit, indent=2))
    json.dump(audit, open(os.path.join(a.outdir, "preprocessing_audit.json"), "w"), indent=2)
    df.head(10).to_csv(os.path.join(tab_dir, "dataset_sample.csv"), index=False)
    df[NUMERIC].describe().T.to_csv(os.path.join(tab_dir, "feature_statistics.csv"))

    Xtr, Xva, Xte, ytr, yva, yte = get_split(X, y, a.base_seed)
    split_info = {"train": int(len(ytr)), "val": int(len(yva)), "test": int(len(yte)),
                  "train_failures": int(ytr.sum()), "val_failures": int(yva.sum()), "test_failures": int(yte.sum())}
    print("split:", split_info)

    info = {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__,
            "sklearn": sklearn.__version__, "matplotlib": matplotlib.__version__,
            "platform": platform.platform(), "processor": platform.processor(),
            "seeds_used": seeds, "base_config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in BASE.items()},
            "split": split_info, "early_stopping_patience": 20, "ae_architecture": "d-6-k-6-d (tanh, linear latent, linear output)",
            "weight_init": "Glorot uniform (tanh/linear/sigmoid), He normal (ReLU)", "optimizer": "mini-batch gradient descent (momentum 0)",
            "synthetic": bool(a.synthetic)}
    json.dump(info, open(os.path.join(a.outdir, "run_info.json"), "w"), indent=2)

    # ---- Fig: class balance + feature histograms --------------------------------
    fig, axs = plt.subplots(2, 3, figsize=(11, 6))
    axs[0, 0].bar(["No failure", "Failure"], [(y == 0).sum(), (y == 1).sum()], color=["#7fb3d5", "#e57373"])
    axs[0, 0].set_title("Class distribution")
    for ax, c in zip(axs.ravel()[1:], NUMERIC):
        ax.hist(df.loc[y == 0, c], bins=40, alpha=.6, density=True, label="No failure")
        ax.hist(df.loc[y == 1, c], bins=40, alpha=.6, density=True, label="Failure")
        ax.set_title(c, fontsize=10)
    axs[0, 1].legend(fontsize=8)
    save(fig, os.path.join(fig_dir, "fig01_dataset_overview.png"))

    # ---- Fig: architecture ---------------------------------------------------------
    fig_architecture(X.shape[1], BASE["k"], BASE["head_hidden"], os.path.join(fig_dir, "fig02_architecture.png"))

    all_rows, store = [], {}

    def go(exp, name, cfg_over, seeds_to_use, keep=False):
        cfg = {**BASE, **cfg_over}
        for s in seeds_to_use:
            r = run(cfg, s, X, y)
            row = {k: v for k, v in r.items() if not k.startswith("_")}
            row.update(experiment=exp, name=name)
            all_rows.append(row)
            if keep and s == seeds_to_use[0]:
                store[name] = r
        last = all_rows[-1]
        print(f"[{exp}] {name:42s} F1={last['f1']:.3f} AUC={last['roc_auc']:.3f} "
              f"recon={last['recon_mse_test']:.4f} ({last['time_s']}s)")

    # ---- E1: main comparison (baseline vs proposed vs ablation) --------------------
    for s in seeds:  # reference logistic regression (NOT a neural network, only a yardstick)
        Xtr_, Xva_, Xte_, ytr_, yva_, yte_ = get_split(X, y, s)
        lr_ = LogisticRegression(class_weight="balanced", max_iter=2000).fit(Xtr_, ytr_)
        pva_, pte_ = lr_.predict_proba(Xva_)[:, 1], lr_.predict_proba(Xte_)[:, 1]
        m = clf_metrics(yte_, pte_, best_threshold(yva_, pva_))
        all_rows.append({"experiment": "E1 main comparison", "name": "Logistic regression (reference only)", "seed": s,
                         "mode": "logreg", "k": np.nan, "recon_mse_test": np.nan, **m})
    go("E1 main comparison", "Baseline: raw-feature MLP", dict(mode="raw"), seeds, keep=True)
    go("E1 main comparison", "Proposed: AE(k=4) + MLP (frozen encoder)", dict(mode="frozen"), seeds, keep=True)
    go("E1 main comparison", "Ablation: AE + MLP fine-tuned end-to-end", dict(mode="finetune"), seeds, keep=True)

    # ---- E2: latent-dimension study ---------------------------------------------------
    for k in [2, 3, 4, 5, 6]:
        go("E2 latent dimension", f"k={k}", dict(k=k), seeds, keep=True)
    # ---- E3: autoencoder learning rate -----------------------------------------------
    for lr in [0.01, 0.05, 0.1, 0.3]:
        go("E3 AE learning rate", f"lr={lr}", dict(ae_lr=lr), seeds, keep=True)
    # ---- E4: mini-batch size (full batch = batch gradient descent) ------------------------
    for bs in [16, 64, 256, 0]:
        go("E4 AE batch size", "full batch (BGD)" if bs == 0 else f"batch={bs}", dict(ae_bs=(None if bs == 0 else bs)), seeds, keep=True)
    # ---- E5: classifier head size / depth ---------------------------------------------
    for hid in [(8,), (16, 8), (32, 16), (32, 16, 8)]:
        go("E5 MLP head architecture", "hidden=" + "-".join(map(str, hid)), dict(head_hidden=hid), seeds)
    # ---- E6: classifier learning rate -------------------------------------------------
    for lr in [0.01, 0.05, 0.2]:
        go("E6 MLP learning rate", f"lr={lr}", dict(head_lr=lr), seeds)
    # ---- E7: other training settings -------------------------------------------------
    go("E7 training settings", "Proposed without class weighting", dict(pos_weight=False), seeds)
    go("E7 training settings", "Baseline without class weighting", dict(mode="raw", pos_weight=False), seeds)
    go("E7 training settings", "Proposed + momentum 0.9 (extension)", dict(momentum=0.9, ae_lr=0.01, head_lr=0.01), seeds)

    R = pd.DataFrame(all_rows)
    R.to_csv(os.path.join(tab_dir, "results_all_runs.csv"), index=False)

    # ---- summary tables ---------------------------------------------------------------------
    md = ["# Project 28 - results summary (mean ± std over seeds " + str(seeds) + ")\n",
          "Run info: see run_info.json.  Test split sizes: " + json.dumps(split_info) + "\n"]
    for exp in R["experiment"].unique():
        sub = R[R.experiment == exp]
        t, _, _ = agg_table(sub, ["name"])
        t.to_csv(os.path.join(tab_dir, "summary_" + exp.split()[0] + ".csv"), index=False)
        md += [f"\n## {exp}\n", "| " + " | ".join(t.columns) + " |", "|" + "---|" * len(t.columns)]
        md += ["| " + " | ".join(map(str, r)) + " |" for r in t.values]
    cm = pd.DataFrame([{"model": n, "TN": store[n]["tn"], "FP": store[n]["fp"], "FN": store[n]["fn"], "TP": store[n]["tp"],
                        "threshold": store[n]["threshold"]} for n in list(store)[:3]])
    cm.to_csv(os.path.join(tab_dir, "confusion_matrices_seed0.csv"), index=False)
    params = pd.DataFrame([{"model": n, "autoencoder_params": store[n]["ae_params"], "head_params": store[n]["head_params"],
                            "total_trainable": store[n]["ae_params"] + store[n]["head_params"]} for n in list(store)[:3]])
    params.to_csv(os.path.join(tab_dir, "parameter_counts.csv"), index=False)
    md += ["\n## Confusion matrices (first seed, test set)\n", cm.to_string(index=False),
           "\n## Parameter counts\n", params.to_string(index=False)]
    open(os.path.join(a.outdir, "results_summary.md"), "w").write("\n".join(md))

    # ---- FIGURES ----------------------------------------------------------------------------------
    P = store["Proposed: AE(k=4) + MLP (frozen encoder)"]
    B = store["Baseline: raw-feature MLP"]
    F = store["Ablation: AE + MLP fine-tuned end-to-end"]

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    ha = P["_hist_ae"]
    axs[0].plot(ha["train"], label="Train"); axs[0].plot(ha["val"], label="Validation")
    axs[0].axvline(ha["best_epoch"] - 1, ls="--", c="gray", lw=1, label="Best epoch")
    axs[0].set(title="Autoencoder: reconstruction loss (MSE)", xlabel="Epoch", ylabel="MSE"); axs[0].set_yscale("log"); axs[0].legend()
    hh = P["_hist_head"]
    axs[1].plot(hh["train"], label="Train"); axs[1].plot(hh["val"], label="Validation")
    axs[1].axvline(hh["best_epoch"] - 1, ls="--", c="gray", lw=1, label="Best epoch")
    axs[1].set(title="MLP on latent code: weighted BCE loss", xlabel="Epoch", ylabel="Loss"); axs[1].legend()
    save(fig, os.path.join(fig_dir, "fig03_training_curves_loss.png"))

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (nm, r) in zip(axs, [("Proposed (AE+MLP)", P), ("Baseline (raw MLP)", B)]):
        ax.plot(r["_hist_head"]["train_acc"], label="Train accuracy")
        ax.plot(r["_hist_head"]["val_acc"], label="Validation accuracy")
        ax.set(title=f"{nm}: accuracy @ threshold 0.5", xlabel="Epoch", ylabel="Accuracy"); ax.legend()
    save(fig, os.path.join(fig_dir, "fig04_training_curves_accuracy.png"))

    fig, axs = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (nm, r) in zip(axs, [("Baseline: raw MLP", B), ("Proposed: AE+MLP", P), ("Fine-tuned AE+MLP", F)]):
        m = np.array([[r["tn"], r["fp"]], [r["fn"], r["tp"]]])
        ax.imshow(m, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, m[i, j], ha="center", va="center", fontsize=14, color="white" if m[i, j] > m.max() / 2 else "black")
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["No failure", "Failure"], yticklabels=["No failure", "Failure"],
               xlabel="Predicted", ylabel="Actual", title=f"{nm}\n(threshold {r['threshold']:.2f})")
    save(fig, os.path.join(fig_dir, "fig05_confusion_matrices.png"))

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for nm, r in [("Baseline raw MLP", B), ("Proposed AE+MLP", P), ("Fine-tuned AE+MLP", F)]:
        fpr, tpr, _ = roc_curve(r["_y_te"], r["_p_te"])
        axs[0].plot(fpr, tpr, label=f"{nm} (AUC={r['roc_auc']:.3f})")
        pr, rc, _ = precision_recall_curve(r["_y_te"], r["_p_te"])
        axs[1].plot(rc, pr, label=f"{nm} (AP={r['pr_auc']:.3f})")
    axs[0].plot([0, 1], [0, 1], "k--", lw=.8)
    axs[0].set(title="ROC curve (test set)", xlabel="False positive rate", ylabel="True positive rate"); axs[0].legend(fontsize=8)
    axs[1].set(title="Precision-recall curve (test set)", xlabel="Recall", ylabel="Precision"); axs[1].legend(fontsize=8)
    save(fig, os.path.join(fig_dir, "fig06_roc_pr_curves.png"))

    sub = R[R.experiment == "E2 latent dimension"]
    g = sub.groupby("k")
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
    axs[0].errorbar(g.recon_mse_test.mean().index, g.recon_mse_test.mean(), g.recon_mse_test.std(ddof=0), marker="o", capsize=3)
    axs[0].set(title="Reconstruction MSE (test)", xlabel="Latent dimension k", ylabel="MSE")
    for m_, lab in [("f1", "F1"), ("roc_auc", "ROC-AUC"), ("pr_auc", "PR-AUC")]:
        axs[1].errorbar(g[m_].mean().index, g[m_].mean(), g[m_].std(ddof=0), marker="o", capsize=3, label=lab)
    axs[1].set(title="Downstream classification (test)", xlabel="Latent dimension k", ylabel="Score"); axs[1].legend()
    bl = R[R.name == "Baseline: raw-feature MLP"]
    axs[2].bar(["Raw MLP"] + [f"k={k}" for k in g.f1.mean().index], [bl.f1.mean()] + list(g.f1.mean()),
               color=["#999999"] + ["#5b9bd5"] * len(g))
    axs[2].set(title="F1 vs raw-feature baseline", ylabel="F1")
    save(fig, os.path.join(fig_dir, "fig07_latent_dimension_study.png"))

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for exp_, ax, ttl in [("E3 AE learning rate", axs[0], "Autoencoder learning rate"), ("E4 AE batch size", axs[1], "Autoencoder batch size")]:
        names = R[R.experiment == exp_]["name"].unique()
        for n in names:
            r = store[n]
            ax.plot(r["_hist_ae"]["val"], label=f"{n} (test MSE {r['recon_mse_test']:.3f})")
        ax.set(title=f"{ttl}: validation MSE", xlabel="Epoch", ylabel="Validation MSE"); ax.set_yscale("log"); ax.legend(fontsize=8)
    save(fig, os.path.join(fig_dir, "fig08_lr_and_batchsize_curves.png"))

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    err, yy = P["_err_te"], P["_y_te"]
    bins = np.logspace(np.log10(max(err.min(), 1e-4)), np.log10(err.max()), 50)
    axs[0].hist(err[yy == 0], bins=bins, alpha=.6, density=True, label="No failure")
    axs[0].hist(err[yy == 1], bins=bins, alpha=.6, density=True, label="Failure")
    axs[0].set_xscale("log"); axs[0].legend()
    axs[0].set(title=f"Reconstruction-error distribution\n(error as anomaly score: AUC={P['recon_err_auc']:.3f})", xlabel="Per-sample MSE")
    z = store["k=2"]["_z_te"]; y2 = store["k=2"]["_y_te"]
    axs[1].scatter(z[y2 == 0, 0], z[y2 == 0, 1], s=4, alpha=.4, label="No failure")
    axs[1].scatter(z[y2 == 1, 0], z[y2 == 1, 1], s=14, c="red", alpha=.8, label="Failure")
    axs[1].set(title="Latent space (k=2), test set", xlabel="z1", ylabel="z2"); axs[1].legend()
    save(fig, os.path.join(fig_dir, "fig09_error_distribution_and_latent_space.png"))

    # ---- console recap -----------------------------------------------------------------------
    print("\n" + "=" * 70 + "\nDONE.  Files written to:", os.path.abspath(a.outdir))
    print(open(os.path.join(a.outdir, "results_summary.md")).read())


if __name__ == "__main__":
    main()
