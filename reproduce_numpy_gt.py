"""Old numpy run_trial, but topology built with graph_tool.random_graph
(matches the original notebook setup exactly).

Run with the system python3.12 since graph_tool is installed via apt:
    /usr/bin/python3.12 reproduce_numpy_gt.py
"""
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
           "MKL_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import itertools
import json
import time
from multiprocessing import Pool

import numpy as np
from scipy.stats import genpareto

from graph_tool.generation import random_graph
from graph_tool.spectral import adjacency

from utils import run_trial


GAMMA, A_PARAM = 2.4, 1.0
MEAN_K, BETA = 3.5, 3.5
N = 1500
T_MAX = 2000


def degree_rv(dist):
    if dist == "sf":
        return int(np.round(genpareto.rvs(c=1 / (GAMMA - 1),
                                          scale=A_PARAM / (GAMMA - 1),
                                          loc=A_PARAM)))
    if dist == "binom":
        return int(np.random.binomial(n=N, p=MEAN_K / N))
    if dist == "exp":
        return int(np.round(np.random.exponential(scale=BETA)))
    raise ValueError(dist)


def gt_topology(in_dist, out_dist):
    g = random_graph(
        N,
        deg_sampler=lambda: (degree_rv(in_dist), degree_rv(out_dist)),
        directed=True,
        verbose=False,
    )
    return np.array(adjacency(g).todense()).astype(np.uint8)


def one_trial(args):
    out_dist, in_dist, seed = args
    np.random.seed(seed)
    T = gt_topology(in_dist, out_dist)
    if T.sum() == 0:
        return out_dist, in_dist, 0, 0, 0
    max_in = int(T.sum(1).max())
    max_out = int(T.sum(0).max())
    is_s, _X, _idx = run_trial(T, N=N, t_max=T_MAX)
    return out_dist, in_dist, int(is_s), max_in, max_out


def main():
    n_trials = 10
    dists = ["sf", "binom", "exp"]
    combos = list(itertools.product(dists, dists))

    jobs = []
    seed = 1000
    for out_d, in_d in combos:
        for _ in range(n_trials):
            jobs.append((out_d, in_d, seed))
            seed += 1

    n_workers = 4
    print(f"[numpy + graph_tool]  N={N}, n_trials={n_trials}, t_max={T_MAX}, "
          f"{len(jobs)} trials, workers={n_workers}")
    t0 = time.time()
    results = {c: [] for c in combos}
    max_in_stats = {c: [] for c in combos}
    max_out_stats = {c: [] for c in combos}

    with Pool(n_workers) as pool:
        for i, (out_d, in_d, is_s, max_in, max_out) in enumerate(
            pool.imap_unordered(one_trial, jobs)
        ):
            results[(out_d, in_d)].append(is_s)
            max_in_stats[(out_d, in_d)].append(max_in)
            max_out_stats[(out_d, in_d)].append(max_out)
            if (i + 1) % 10 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (len(jobs) - (i + 1)) / rate
                print(f"  done {i + 1}/{len(jobs)}  elapsed={elapsed:.0f}s  ETA={eta:.0f}s")

    print(f"\nTotal: {time.time() - t0:.1f}s\n")
    CF = {c: float(np.mean(results[c])) for c in combos}
    print("CF table (rows=out_dist, cols=in_dist):")
    print("        " + " ".join(f"{d:>7}" for d in dists))
    for out_d in dists:
        row = f"{out_d:>6}: " + " ".join(
            f"{CF[(out_d, in_d)]:>7.2f}" for in_d in dists
        )
        print(row)

    print("\nside-by-side  numpy+gt / config-model+numpy / original:")
    config_model = {  # CF_repro.json from earlier run
        ("sf", "sf"): 0.50, ("sf", "binom"): 0.80, ("sf", "exp"): 0.80,
        ("binom", "sf"): 0.10, ("binom", "binom"): 0.0, ("binom", "exp"): 0.0,
        ("exp", "sf"): 0.30, ("exp", "binom"): 0.0, ("exp", "exp"): 0.0,
    }
    original = {
        ("sf", "sf"): 0.65, ("sf", "binom"): 0.64, ("sf", "exp"): 0.85,
        ("binom", "sf"): 0.02, ("binom", "binom"): 0.0, ("binom", "exp"): 0.0,
        ("exp", "sf"): 0.30, ("exp", "binom"): 0.01, ("exp", "exp"): 0.30,
    }
    for out_d in dists:
        for in_d in dists:
            print(
                f"  ({out_d:>5}, {in_d:>5})   "
                f"gt={CF[(out_d, in_d)]:.2f}  "
                f"cfg-model={config_model[(out_d, in_d)]:.2f}  "
                f"original={original[(out_d, in_d)]:.2f}"
            )

    print("\nmax in/out-degree per ensemble (mean over trials):")
    for c in combos:
        print(
            f"  {c}  max_in_avg={np.mean(max_in_stats[c]):.0f}  "
            f"max_out_avg={np.mean(max_out_stats[c]):.0f}"
        )

    with open("CF_repro_numpy_gt.json", "w") as f:
        json.dump({str(k): v for k, v in CF.items()}, f, indent=2)
    print("\nWrote CF_repro_numpy_gt.json")


if __name__ == "__main__":
    main()
