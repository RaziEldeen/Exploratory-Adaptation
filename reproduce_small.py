"""Small reproduction of Convergence Fraction across (out_dist, in_dist) ensembles.

Builds adjacency via a configuration-model (numpy only, no graph_tool).
"""
import itertools
import numpy as np
from scipy.stats import genpareto

from utils import run_trial


def degree_rv(dist, gamma=2.4, a=1.0, beta=3.5, mean_k=3.5, N=300):
    if dist == "sf":
        return int(np.round(genpareto.rvs(c=1 / (gamma - 1), scale=a / (gamma - 1), loc=a)))
    if dist == "binom":
        return int(np.random.binomial(n=N, p=mean_k / N))
    if dist == "exp":
        return int(np.round(np.random.exponential(scale=beta)))
    raise ValueError(dist)


def sample_degrees(dist, N, **kw):
    return np.array([max(0, degree_rv(dist, N=N, **kw)) for _ in range(N)], dtype=int)


def config_model_adj(in_deg, out_deg, N, max_edge_cap=None):
    """Directed configuration model. Pair out-stubs with in-stubs uniformly at random.
    Drops self-loops and multi-edges (turns multi-edges into single edges)."""
    # Match totals by trimming the larger side at random.
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
            T[v, u] = 1  # row=target, col=source (matches W @ tanh(x))
    if max_edge_cap is not None and T.sum() > max_edge_cap:
        # cap to avoid huge matrices in case SF tail produces a giant hub
        idxs = np.argwhere(T == 1)
        keep = np.random.choice(len(idxs), max_edge_cap, replace=False)
        T2 = np.zeros_like(T)
        for k in keep:
            T2[idxs[k][0], idxs[k][1]] = 1
        T = T2
    return T


def main():
    # Smaller scale for fast smoke run
    N = 300
    n_trials = 8
    t_max = 1000

    rng_seed = 0
    np.random.seed(rng_seed)

    dists = ["sf", "binom", "exp"]
    combos = list(itertools.product(dists, dists))  # (out_dist, in_dist)

    print(f"N={N}, n_trials={n_trials}, t_max={t_max}")
    print(f"{'out':>6} {'in':>6}   CF")
    CF = {}
    for out_dist, in_dist in combos:
        successes = 0
        for _ in range(n_trials):
            in_deg = sample_degrees(in_dist, N)
            out_deg = sample_degrees(out_dist, N)
            T = config_model_adj(in_deg, out_deg, N)
            if T.sum() == 0:
                continue
            is_s, _X, _idx = run_trial(T, N=N, t_max=t_max)
            successes += int(is_s)
        cf = successes / n_trials
        CF[(out_dist, in_dist)] = cf
        print(f"{out_dist:>6} {in_dist:>6}   {cf:.2f}")

    # Pretty table
    print("\nCF table (rows=out_dist, cols=in_dist):")
    header = "        " + " ".join(f"{d:>7}" for d in dists)
    print(header)
    for out_d in dists:
        row = f"{out_d:>6}: " + " ".join(f"{CF[(out_d, in_d)]:>7.2f}" for in_d in dists)
        print(row)


if __name__ == "__main__":
    main()
