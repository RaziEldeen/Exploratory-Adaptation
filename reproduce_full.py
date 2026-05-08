"""Larger reproduction matching the original N=1500 setup.

Runs the full 3x3 (out_dist, in_dist) grid via numpy-only directed
configuration model. Parallelized across trials.
"""
import os
# Force single-threaded BLAS so the multiprocessing Pool doesn't
# oversubscribe 4 workers x N BLAS threads.
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
           "MKL_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import itertools
import json
import time
from multiprocessing import Pool

import numpy as np
from scipy.stats import genpareto

from utils import run_trial


GAMMA, A_PARAM = 2.4, 1.0
MEAN_K, BETA = 3.5, 3.5


def degree_rv(dist, N):
    if dist == "sf":
        return int(np.round(genpareto.rvs(c=1 / (GAMMA - 1), scale=A_PARAM / (GAMMA - 1), loc=A_PARAM)))
    if dist == "binom":
        return int(np.random.binomial(n=N, p=MEAN_K / N))
    if dist == "exp":
        return int(np.round(np.random.exponential(scale=BETA)))
    raise ValueError(dist)


def sample_degrees(dist, N):
    return np.array([max(0, degree_rv(dist, N)) for _ in range(N)], dtype=int)


def config_model_adj(in_deg, out_deg, N):
    while in_deg.sum() != out_deg.sum():
        if in_deg.sum() > out_deg.sum():
            i = np.random.randint(N)
            if in_deg[i] > 0:
                in_deg[i] -= 1
        else:
            i = np.random.randint(N)
            if out_deg[i] > 0:
                out_deg[i] -= 1
    out_stubs = np.repeat(np.arange(N), out_deg)
    in_stubs = np.repeat(np.arange(N), in_deg)
    np.random.shuffle(in_stubs)
    T = np.zeros((N, N), dtype=np.uint8)
    for u, v in zip(out_stubs, in_stubs):
        if u != v:
            T[v, u] = 1  # T[target, source]
    return T


def one_trial(args):
    out_dist, in_dist, N, t_max, seed = args
    np.random.seed(seed)
    in_deg = sample_degrees(in_dist, N)
    out_deg = sample_degrees(out_dist, N)
    T = config_model_adj(in_deg, out_deg, N)
    if T.sum() == 0:
        return out_dist, in_dist, 0, 0, 0
    max_in = int(T.sum(1).max())
    max_out = int(T.sum(0).max())
    is_s, _X, idx = run_trial(T, N=N, t_max=t_max)
    return out_dist, in_dist, int(is_s), max_in, max_out


def main():
    N = 1500
    n_trials = 10
    t_max = 2000

    dists = ["sf", "binom", "exp"]
    combos = list(itertools.product(dists, dists))

    jobs = []
    seed_counter = 0
    for out_d, in_d in combos:
        for _ in range(n_trials):
            jobs.append((out_d, in_d, N, t_max, seed_counter))
            seed_counter += 1

    n_workers = max(1, min(os.cpu_count() or 2, 4))
    print(
        f"N={N}, n_trials={n_trials}, t_max={t_max}, "
        f"{len(jobs)} total trials, workers={n_workers}"
    )
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

    elapsed = time.time() - t0
    print(f"\nTotal wall time: {elapsed:.1f}s\n")

    # CF table + max-degree summary
    CF = {c: float(np.mean(results[c])) for c in combos}
    print("CF table (rows=out_dist, cols=in_dist):")
    header = "        " + " ".join(f"{d:>7}" for d in dists)
    print(header)
    for out_d in dists:
        row = f"{out_d:>6}: " + " ".join(f"{CF[(out_d, in_d)]:>7.2f}" for in_d in dists)
        print(row)

    # Original published values for direct comparison
    original_cf = {
        ("sf", "sf"): 0.65, ("sf", "binom"): 0.64, ("sf", "exp"): 0.85,
        ("binom", "sf"): 0.02, ("binom", "binom"): 0.0, ("binom", "exp"): 0.0,
        ("exp", "sf"): 0.30, ("exp", "binom"): 0.01, ("exp", "exp"): 0.30,
    }
    print("\nside-by-side (out, in)  ours / original:")
    for out_d in dists:
        for in_d in dists:
            print(
                f"  ({out_d:>5}, {in_d:>5})   "
                f"{CF[(out_d, in_d)]:.2f}  vs  {original_cf[(out_d, in_d)]:.2f}"
            )

    print("\nmax in/out-degree per ensemble (mean over trials):")
    for c in combos:
        print(
            f"  {c}  max_in_avg={np.mean(max_in_stats[c]):.0f}  "
            f"max_out_avg={np.mean(max_out_stats[c]):.0f}"
        )

    # Save
    with open("CF_repro.json", "w") as f:
        json.dump({str(k): v for k, v in CF.items()}, f, indent=2)
    print("\nWrote CF_repro.json")


if __name__ == "__main__":
    main()
