"""Participation-ratio figure for the nine degree-distribution ensembles.

Extracted from the notebook `Exploratory adaptation in large random networks.ipynb`
(cells 2, 15, 21, 22, 24). Two variants were computed there:

  * cell 15: PR of X during an EA trial (run_trial, learning on), together with
    CF; saved as PR_results.pickle / CF_results.txt, n = 100 per ensemble.
  * cell 21: PR of the free dynamics (run_dynamics, no learning, t_max = 500,
    all 5000 steps).

Ensemble keys are (out_dist, in_dist): the notebook loops
`for (out_dist, in_dist) in all_combinations` and passes
`deg_sampler=lambda: (degree_rv(in_dist), degree_rv(out_dist))` to graph_tool,
whose sampler returns (in-degree, out-degree). So ('sf', 'binom') means
scale-free OUT-degree with binomial in-degree.

Usage
  python experiments/pr_figure.py                 # plot from saved pickle/json
  python experiments/pr_figure.py --recompute --n-trials 20 --N 500
      # recompute PR of free dynamics (cell 21) with graph_tool if available,
      # otherwise with the configuration-model builder in hub_mechanism.py
"""

import argparse
import itertools
import json
import pickle
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DISTS = ["sf", "binom", "exp"]
COMBOS = list(itertools.product(DISTS, DISTS))          # (out_dist, in_dist)
GAMMA, A_PARAM, MEAN_K, BETA = 2.4, 1.0, 3.5, 3.5


# ---- notebook cell 2 --------------------------------------------------------

def comp_d(X, d_sim, start_idx=0):
    """Participation ratio of the activity covariance over X[start_idx:d_sim]."""
    X = X[start_idx:d_sim]
    X_ = X - X.mean(0)
    C = X_.T.dot(X_) / (d_sim - start_idx + 1)
    return np.trace(C) ** 2 / np.trace(C.dot(C))


# ---- notebook cells 6, 10, 22 (free dynamics, no learning) -----------------

def degree_rv(dist, N):
    from scipy.stats import genpareto
    if dist == "sf":
        return genpareto.rvs(c=1 / (GAMMA - 1), scale=A_PARAM / (GAMMA - 1), loc=A_PARAM)
    if dist == "binom":
        return np.random.binomial(n=N, p=MEAN_K / N)
    if dist == "exp":
        return np.random.exponential(scale=BETA)
    raise ValueError(dist)


def topology(out_dist, in_dist, N):
    """Adjacency T with T[i, j] = 1 for edge j -> i (in-degree = row sums)."""
    try:
        from graph_tool.generation import random_graph
        from graph_tool.spectral import adjacency
        g = random_graph(N, deg_sampler=lambda: (degree_rv(in_dist, N), degree_rv(out_dist, N)),
                         directed=True, verbose=False)
        return np.array(adjacency(g).todense()).astype(np.uint8)
    except ImportError:
        # fallback: exact degrees on one side, random partners on the other
        T = np.zeros((N, N), dtype=np.uint8)
        heavy_out = out_dist != "binom"
        dist = out_dist if heavy_out else in_dist
        ks = np.minimum(np.round([degree_rv(dist, N) for _ in range(N)]).astype(int), N - 1)
        for i in range(N):
            partners = np.random.choice(N - 1, size=ks[i], replace=False)
            partners = partners + (partners >= i)
            if heavy_out:
                T[partners, i] = 1
            else:
                T[i, partners] = 1
        return T


def init_J(T, g_0=10.0):
    active = np.where(T != 0)
    avg_k = T.sum(1).mean()
    J = np.zeros(T.shape)
    J[active] = (g_0 / np.sqrt(avg_k)) * np.random.randn(active[0].size)
    return J


def run_dynamics(T, t_max=500.0, dt=0.1, g_0=10.0):
    N = T.shape[0]
    W = init_J(T, g_0)
    T_sim = round(t_max / dt) + 1
    X = np.zeros((T_sim, N))
    X[0] = 10 * np.random.randn(N)
    for i in range(T_sim - 1):
        X[i + 1] = X[i] + dt * (W @ np.tanh(X[i]) - X[i])
    return X, W


def recompute(N, n_trials, t_max, seed):
    np.random.seed(seed)
    PR = {}
    for out_dist, in_dist in COMBOS:
        d = np.zeros(n_trials)
        for i in range(n_trials):
            T = topology(out_dist, in_dist, N)
            X, _ = run_dynamics(T, t_max=t_max)
            d[i] = comp_d(X, X.shape[0])
        PR[(out_dist, in_dist)] = d
        print(f"out={out_dist:5s} in={in_dist:5s}  PR mean={d.mean():6.2f}  median={np.median(d):6.2f}")
    return PR


# ---- figure (notebook cell 24, plus CF vs PR) --------------------------------

def plot(PR, CF, out_path, title):
    keys = list(PR.keys())
    labels = [f"out={o}\nin={i}" for o, i in keys]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={"width_ratios": [2, 1]})
    ax = axes[0]
    ax.boxplot([PR[k] for k in keys], tick_labels=labels)
    ax.set_ylabel("participation ratio d")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax = axes[1]
    for k in keys:
        if CF is not None and k in CF:
            ax.scatter(np.median(PR[k]), CF[k], s=60)
            ax.annotate(f"{k[0]}/{k[1]}", (np.median(PR[k]), CF[k]), fontsize=8,
                        xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("median participation ratio")
    ax.set_ylabel("CF")
    ax.set_title("CF vs PR (label = out/in)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recompute", action="store_true")
    ap.add_argument("--N", type=int, default=1500)
    ap.add_argument("--n-trials", type=int, default=50)
    ap.add_argument("--t-max", type=float, default=500.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    CF = None
    cf_path = ROOT / "CF_results.txt"
    if cf_path.exists():
        CF = {eval(k): v for k, v in json.load(open(cf_path)).items()}

    if args.recompute:
        PR = recompute(args.N, args.n_trials, args.t_max, args.seed)
        title = f"PR of free dynamics (no learning), N={args.N}, n={args.n_trials}"
        out = args.out or ROOT / "experiments" / "results" / f"pr_free_N{args.N}_n{args.n_trials}.png"
        with open(ROOT / "experiments" / "results" / f"pr_free_N{args.N}_n{args.n_trials}.pickle", "wb") as f:
            pickle.dump(PR, f)
    else:
        with open(ROOT / "PR_results.pickle", "rb") as f:
            PR = pickle.load(f)
        title = "PR during EA trials, N=1500, n=100 (PR_results.pickle)"
        out = args.out or ROOT / "experiments" / "results" / "pr_saved_N1500.png"
    plot(PR, CF, out, title)


if __name__ == "__main__":
    main()
