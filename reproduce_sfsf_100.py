"""(sf, sf) only, 100 trials — direct apples-to-apples with the original
CF[(sf,sf)] = 0.65 in CF_results.txt.

Run with /usr/bin/python3.12 (graph_tool from apt).
"""
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
           "MKL_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import time
from multiprocessing import Pool

import numpy as np
from scipy.stats import genpareto
from graph_tool.generation import random_graph
from graph_tool.spectral import adjacency

from utils import run_trial


GAMMA, A_PARAM = 2.4, 1.0
N = 1500
T_MAX = 2000


def degree_rv():
    # SF only.
    return int(np.round(genpareto.rvs(c=1 / (GAMMA - 1),
                                      scale=A_PARAM / (GAMMA - 1),
                                      loc=A_PARAM)))


def gt_topology():
    g = random_graph(
        N,
        deg_sampler=lambda: (degree_rv(), degree_rv()),
        directed=True,
        verbose=False,
    )
    return np.array(adjacency(g).todense()).astype(np.uint8)


def one_trial(seed):
    np.random.seed(seed)
    T = gt_topology()
    is_s, _X, _idx = run_trial(T, N=N, t_max=T_MAX)
    max_in = int(T.sum(1).max())
    max_out = int(T.sum(0).max())
    return int(is_s), max_in, max_out


def main():
    n_trials = 100
    n_workers = 4
    print(f"[sf, sf]  N={N}, n_trials={n_trials}, t_max={T_MAX}, "
          f"workers={n_workers}")
    t0 = time.time()
    successes = []
    max_ins, max_outs = [], []
    seeds = list(range(2000, 2000 + n_trials))
    with Pool(n_workers) as pool:
        for i, (is_s, mi, mo) in enumerate(pool.imap_unordered(one_trial, seeds)):
            successes.append(is_s)
            max_ins.append(mi)
            max_outs.append(mo)
            if (i + 1) % 10 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (n_trials - (i + 1)) / rate
                print(f"  done {i + 1}/{n_trials}  CF_so_far="
                      f"{np.mean(successes):.3f}  elapsed={elapsed:.0f}s  "
                      f"ETA={eta:.0f}s", flush=True)

    cf = float(np.mean(successes))
    print(f"\nDone in {time.time() - t0:.1f}s")
    print(f"CF[(sf, sf)] = {cf:.3f}  (n={n_trials})")
    print(f"Original CF_results.txt[(sf, sf)] = 0.650")
    # 95% Wilson CI for proportions
    p = cf
    n = n_trials
    z = 1.96
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    print(f"95% CI for our estimate: [{centre - half:.3f}, {centre + half:.3f}]")
    print(f"max_in_avg={np.mean(max_ins):.0f}  max_out_avg={np.mean(max_outs):.0f}")


if __name__ == "__main__":
    main()
