"""Fast JAX reproduction. The slow version sampled (N,N) noise per
step (2.25M values, mostly masked away). This version samples noise only
on active edges (~5k) using padded indices for static shape, so a single
jit compiles once and serves every topology.

Run with the system python3.12 (graph_tool comes from apt):
    /usr/bin/python3.12 reproduce_jax_gt.py
"""
import itertools
import json
import os
import time
from functools import partial

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
           "MKL_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
from scipy.stats import genpareto
import jax
import jax.numpy as jnp
from jax import jit, lax, random

from graph_tool.generation import random_graph
from graph_tool.spectral import adjacency


GAMMA, A_PARAM = 2.4, 1.0
MEAN_K, BETA = 3.5, 3.5
N = 1500
T_MAX = 2000
DT = 0.1
T_STOP = 100
TOL = 1e-2
EPS, MU, M0 = 3.0, 0.01, 4.0
D_NOISE = 1e-3
G0 = 10.0
B_ALPHA = 100.0
SPARSITY = 0.2
CHUNK = 200
N_CHUNKS = int(round(T_MAX / DT)) // CHUNK   # 100

MAX_EDGES = 30_000  # padding upper bound for active-edge index arrays


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
    return np.array(adjacency(g).todense()).astype(np.float32)


def pad_active_indices(T):
    """Return (rows, cols, valid_mask, avg_k) padded to MAX_EDGES."""
    rows, cols = np.where(T != 0)
    n = len(rows)
    if n > MAX_EDGES:
        raise RuntimeError(f"topology has {n} edges, exceeds MAX_EDGES={MAX_EDGES}")
    pad = MAX_EDGES - n
    rows_p = np.concatenate([rows, np.zeros(pad, dtype=rows.dtype)]).astype(np.int32)
    cols_p = np.concatenate([cols, np.zeros(pad, dtype=cols.dtype)]).astype(np.int32)
    valid = np.concatenate([np.ones(n), np.zeros(pad)]).astype(np.float32)
    avg_k = float(T.sum(1).mean())
    return rows_p, cols_p, valid, avg_k


@partial(jit, static_argnames=("chunk", "n_chunks", "max_edges"))
def trial_jax(key, T_mask, rows, cols, valid, avg_k,
              chunk=CHUNK, n_chunks=N_CHUNKS, max_edges=MAX_EDGES):
    """One EA trial, returns full Ms[t] sequence."""
    kb, kW, kx, kn = random.split(key, 4)

    cN = int(round(N * SPARSITY))
    idxs = random.permutation(kb, jnp.arange(N))[:cN]
    g_b = (1.0 / G0) * jnp.sqrt(B_ALPHA / cN)
    b = jnp.zeros((N,)).at[idxs].set(g_b * random.normal(kb, (cN,)))

    W0 = ((G0 / jnp.sqrt(avg_k)) * random.normal(kW, T_mask.shape)) * T_mask
    x0 = 10.0 * random.normal(kx, (N,))

    def body(carry, key_chunk):
        x, W = carry

        def inner(state, k_step):
            x, W = state
            s = jnp.abs(jnp.dot(b, x))
            Ms = (M0 / 2.0) * (1.0 + jnp.tanh((s - EPS) / MU))
            x_next = x + DT * (jnp.dot(W, jnp.tanh(x)) - x)
            # Sparse noise on active edges only (padded slots get masked off).
            noise = random.normal(k_step, (max_edges,)) * valid
            W_next = W.at[rows, cols].add(jnp.sqrt(Ms * DT * D_NOISE) * noise)
            return (x_next, W_next), Ms

        keys = random.split(key_chunk, chunk)
        (x_e, W_e), Ms_seq = lax.scan(inner, (x, W), keys)
        return (x_e, W_e), Ms_seq

    keys = random.split(kn, n_chunks)
    (_x, _W), Ms = lax.scan(body, (x0, W0), keys)
    return Ms.reshape(-1)


def converged(Ms):
    win = int(T_STOP / DT)
    below = np.asarray(Ms) <= TOL
    if len(below) < win:
        return False
    not_below = (~below).astype(int)
    csum = np.concatenate([[0], np.cumsum(not_below)])
    sliding_false = csum[win:] - csum[:-win]
    return bool(np.any(sliding_false == 0))


def main():
    np.random.seed(0)
    key = random.PRNGKey(0)
    n_trials = 10

    dists = ["sf", "binom", "exp"]
    combos = list(itertools.product(dists, dists))

    print(f"[JAX-sparse + graph_tool]  N={N}, n_trials={n_trials}, t_max={T_MAX}, "
          f"chunk={CHUNK}, n_chunks={N_CHUNKS}, max_edges={MAX_EDGES}")

    print("compiling JIT...", flush=True)
    t_compile = time.time()
    T_dummy = gt_topology("binom", "binom")
    rows_d, cols_d, valid_d, avg_k_d = pad_active_indices(T_dummy)
    _ = trial_jax(
        random.PRNGKey(99),
        jnp.asarray(T_dummy),
        jnp.asarray(rows_d), jnp.asarray(cols_d),
        jnp.asarray(valid_d), avg_k_d,
    ).block_until_ready()
    print(f"  compiled in {time.time() - t_compile:.1f}s", flush=True)

    t0 = time.time()
    CF = {}
    max_in_avg, max_out_avg = {}, {}
    for out_d, in_d in combos:
        succ = 0
        max_ins, max_outs = [], []
        cell_t0 = time.time()
        for _ in range(n_trials):
            T = gt_topology(in_d, out_d)
            max_ins.append(int(T.sum(1).max()))
            max_outs.append(int(T.sum(0).max()))
            rows, cols, valid, avg_k = pad_active_indices(T)
            key, sub = random.split(key)
            Ms = trial_jax(
                sub,
                jnp.asarray(T),
                jnp.asarray(rows), jnp.asarray(cols),
                jnp.asarray(valid), avg_k,
            )
            Ms.block_until_ready()
            if converged(Ms):
                succ += 1
        CF[(out_d, in_d)] = succ / n_trials
        max_in_avg[(out_d, in_d)] = float(np.mean(max_ins))
        max_out_avg[(out_d, in_d)] = float(np.mean(max_outs))
        elapsed = time.time() - t0
        cell_dt = time.time() - cell_t0
        print(
            f"  {out_d:>5} {in_d:>5}  CF={CF[(out_d, in_d)]:.2f}  "
            f"max_in={max_in_avg[(out_d, in_d)]:.0f}  "
            f"max_out={max_out_avg[(out_d, in_d)]:.0f}  "
            f"(cell {cell_dt:.0f}s, total {elapsed:.0f}s)",
            flush=True,
        )

    print(f"\nTotal trial time: {time.time() - t0:.1f}s\n")
    print("CF table (rows=out_dist, cols=in_dist):")
    print("        " + " ".join(f"{d:>7}" for d in dists))
    for out_d in dists:
        print(
            f"{out_d:>6}: "
            + " ".join(f"{CF[(out_d, in_d)]:>7.2f}" for in_d in dists)
        )

    original = {
        ("sf", "sf"): 0.65, ("sf", "binom"): 0.64, ("sf", "exp"): 0.85,
        ("binom", "sf"): 0.02, ("binom", "binom"): 0.0, ("binom", "exp"): 0.0,
        ("exp", "sf"): 0.30, ("exp", "binom"): 0.01, ("exp", "exp"): 0.30,
    }
    print("\nside-by-side  ours / original:")
    for out_d in dists:
        for in_d in dists:
            print(
                f"  ({out_d:>5}, {in_d:>5})   "
                f"{CF[(out_d, in_d)]:.2f}  vs  {original[(out_d, in_d)]:.2f}"
            )

    with open("CF_repro_jax_gt.json", "w") as f:
        json.dump({str(k): v for k, v in CF.items()}, f, indent=2)
    print("\nWrote CF_repro_jax_gt.json")


if __name__ == "__main__":
    main()
