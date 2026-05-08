"""Focused reproduction: SF vs binom vs exp on the diagonal at larger N."""
import numpy as np
from scipy.stats import genpareto
from utils import run_trial


def degree_rv(dist, gamma=2.4, a=1.0, beta=3.5, mean_k=3.5, N=600):
    if dist == "sf":
        return int(np.round(genpareto.rvs(c=1 / (gamma - 1), scale=a / (gamma - 1), loc=a)))
    if dist == "binom":
        return int(np.random.binomial(n=N, p=mean_k / N))
    if dist == "exp":
        return int(np.round(np.random.exponential(scale=beta)))
    raise ValueError(dist)


def sample_degrees(dist, N):
    return np.array([max(0, degree_rv(dist, N=N)) for _ in range(N)], dtype=int)


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


def main():
    np.random.seed(42)
    N = 600
    n_trials = 6
    t_max = 1500

    print(f"N={N}, n_trials={n_trials}, t_max={t_max}")
    print(f"{'out=in':>10}   CF   max_out_deg(avg)   max_in_deg(avg)")

    for dist in ["sf", "binom", "exp"]:
        successes = 0
        max_outs, max_ins = [], []
        for _ in range(n_trials):
            in_deg = sample_degrees(dist, N)
            out_deg = sample_degrees(dist, N)
            T = config_model_adj(in_deg, out_deg, N)
            if T.sum() == 0:
                continue
            max_outs.append(int(T.sum(0).max()))
            max_ins.append(int(T.sum(1).max()))
            is_s, _X, _idx = run_trial(T, N=N, t_max=t_max)
            successes += int(is_s)
        cf = successes / n_trials
        print(
            f"{dist:>10}   {cf:.2f}   {np.mean(max_outs):>14.1f}   "
            f"{np.mean(max_ins):>13.1f}"
        )


if __name__ == "__main__":
    main()
