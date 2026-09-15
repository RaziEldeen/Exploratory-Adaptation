"""Small-N experiments on *why* structure lets exploratory adaptation converge.

Convention everywhere: J[i, j] is the weight of edge j -> i, so
in-degree(i) = T[i, :].sum() (row sums) and out-degree(j) = T[:, j].sum().
Dynamics: x_dot = J tanh(x) - x.  EA: dW = sqrt(M(s) D dt) * eta on active edges.

Network variants
  er            plain Erdos-Renyi bulk, no hub
  hub           lumped hub as in simulate_lumped_hub.py (self-loop w.p. alpha,
                weight N(0, sigma^2); hub is dead unless w_hh > 1)
  hub_forced    hub with a forced self-loop of weight 3 (state ~ +-2.98)
  hub_clamped   hub state clamped to +1 forever: a pure bias node
  sf_in         scale-free in-degree (gamma=2.4), Poisson-like out-degree
  sf_out        scale-free out-degree, Poisson-like in-degree

Per-trial diagnostics
  conv, t_conv, hub_alive, frac_fp (fraction of steps with |x_dot| small),
  lyap_end (trailing-window Lyapunov exponent), n_frozen (|tanh x| > 0.99),
  hub_norm2 trajectory + prediction, |W - W0|, leading Jacobian eigenvalue at
  the end, robustness (does output stay in tolerance after a weight kick).
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from scipy.stats import genpareto

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"

DEFAULT_TRIAL = dict(b_alpha=100.0, c=0.2, g_0=10.0, m_b=0.0, D=1e-3, eps=3.0,
                     mu=0.01, M_0=4.0, target=0.0, dt=0.1, t_max=300.0,
                     T_stop=50.0, tol=1e-2)


# ---------------------------------------------------------------- networks --

def er_bulk(rng, n, mean_k):
    T = (rng.random((n, n)) < mean_k / n).astype(np.uint8)
    np.fill_diagonal(T, 0)
    return T


def sf_degrees(rng, n, gamma=2.4, a=1.0, cap=None):
    k = genpareto.rvs(c=1 / (gamma - 1), scale=a / (gamma - 1), loc=a,
                      size=n, random_state=rng)
    k = np.round(k).astype(int)
    if cap is not None:
        k = np.minimum(k, cap)
    return k


def sf_graph(rng, n, heavy="in", gamma=2.4):
    """Exact heavy-tailed degrees on one side, random partners on the other."""
    T = np.zeros((n, n), dtype=np.uint8)
    ks = sf_degrees(rng, n, gamma=gamma, cap=n - 1)
    for i in range(n):
        partners = rng.choice(n - 1, size=ks[i], replace=False)
        partners = partners + (partners >= i)  # skip self
        if heavy == "in":
            T[i, partners] = 1        # i receives from partners
        else:
            T[partners, i] = 1        # i sends to partners
    return T


def gaussian_weights(rng, T, g_0):
    active = np.where(T != 0)
    avg_k = T.sum(axis=1).mean() or 1.0
    J = np.zeros(T.shape)
    J[active] = (g_0 / np.sqrt(avg_k)) * rng.standard_normal(active[0].size)
    return J


def build(rng, kind, N, mean_k, sigma, alpha, g_0=10.0):
    """Return dict(T, J, hub, clamp) with hub = index of hub node or None."""
    if kind == "er":
        T = er_bulk(rng, N, mean_k)
        return dict(T=T, J=gaussian_weights(rng, T, g_0), hub=None, clamp=None)
    if kind in ("sf_in", "sf_out"):
        T = sf_graph(rng, N, heavy="in" if kind == "sf_in" else "out")
        return dict(T=T, J=gaussian_weights(rng, T, g_0), hub=None, clamp=None)

    nb = N - 1
    Tb = er_bulk(rng, nb, mean_k)
    Jb = gaussian_weights(rng, Tb, g_0)
    T = np.zeros((N, N), dtype=np.uint8)
    J = np.zeros((N, N))
    T[:nb, :nb], J[:nb, :nb] = Tb, Jb
    h = N - 1
    mask = rng.random(N) < alpha
    if kind == "hub":
        pass                                  # self-loop only w.p. alpha (original)
    else:
        mask[h] = kind == "hub_forced"        # forced self-loop / none if clamped
    T[:, h] = mask
    J[mask, h] = sigma * rng.standard_normal(mask.sum())
    clamp = None
    if kind == "hub_forced":
        J[h, h] = 3.0
    if kind == "hub_clamped":
        clamp = 1.0
    return dict(T=T, J=J, hub=h, clamp=clamp)


# ------------------------------------------------------------------- trial --

def init_b(rng, N, c, b_alpha, g_0, m_b):
    b = np.zeros(N)
    cN = int(round(N * c))
    idx = rng.permutation(N)[:cN]
    b[idx] = m_b + (1.0 / g_0) * np.sqrt(b_alpha / cN) * rng.standard_normal(cN)
    return b


def run_trial(rng, net, *, learn="all", hub_noise_gain=1.0, b_alpha, c, g_0, m_b,
              D, eps, mu, M_0, target, dt, t_max, T_stop, tol, kick=0.3):
    T, J, hub, clamp = net["T"], net["J"], net["hub"], net["clamp"]
    N = T.shape[0]
    Tlearn = T.copy()
    if learn == "hub_only" and hub is not None:
        Tlearn[:, :hub] = 0                   # only the hub's out-column moves
    elif learn == "bulk_only" and hub is not None:
        Tlearn[:, hub] = 0
    elif learn == "none":
        Tlearn[:] = 0
    active = np.where(Tlearn != 0)
    n_active = active[0].size
    gain = np.ones(n_active)
    if hub is not None and hub_noise_gain != 1.0:
        gain[active[1] == hub] = hub_noise_gain   # scale-matched hub exploration
    hub_col = np.where(T[:, hub] != 0)[0] if hub is not None else None

    b = init_b(rng, N, c, b_alpha, g_0, m_b)
    W, W0 = J.copy(), J.copy()
    x = 10.0 * rng.standard_normal(N)
    v = rng.standard_normal(N); v /= np.linalg.norm(v)   # tangent vector

    n_steps = int(round(t_max / dt))
    window = int(round(T_stop / dt))
    below, converged, t_conv = 0, 0, np.nan
    fp_steps = 0
    loggrowth = np.zeros(n_steps)
    intM = 0.0
    norm_t, norm_pred, times = [], [], []
    sqrt_dt_D = np.sqrt(dt * D)

    for i in range(n_steps):
        if clamp is not None:
            x[hub] = clamp
        th = np.tanh(x)
        s = abs(b @ x - target)
        M_s = (M_0 / 2.0) * (1.0 + np.tanh((s - eps) / mu))

        dx = W @ th - x
        if clamp is not None:
            dx[hub] = 0.0
        if np.sqrt(np.mean(dx ** 2)) < 1e-2:
            fp_steps += 1
        # tangent dynamics for Lyapunov exponent
        dv = W @ (v * (1 - th ** 2)) - v
        if clamp is not None:
            dv[hub] = 0.0
        v = v + dt * dv
        nv = np.linalg.norm(v)
        loggrowth[i] = np.log(nv) / dt
        v /= nv

        x = x + dt * dx
        if n_active and M_s > 0.0:
            W[active] += sqrt_dt_D * np.sqrt(M_s) * gain * rng.standard_normal(n_active)
        intM += M_s * dt
        if hub is not None and i % 50 == 0:
            times.append(i * dt)
            norm_t.append(float(np.sum(W[hub_col, hub] ** 2)))
            n_out_learn = int(Tlearn[:, hub].sum())
            norm_pred.append(float(np.sum(W0[hub_col, hub] ** 2) + n_out_learn * D * intM))

        if M_s <= tol:
            below += 1
            if below >= window and i * dt > T_stop:
                converged, t_conv = 1, i * dt
                break
        else:
            below = 0

    steps_done = i + 1
    lyap_end = float(np.mean(loggrowth[max(0, steps_done - window):steps_done]))
    th = np.tanh(x)
    n_frozen = int(np.sum(np.abs(th) > 0.99))
    hub_alive = float(abs(th[hub]) > 0.5) if hub is not None else np.nan

    # leading Jacobian eigenvalue at the final state
    Jac = W * (1 - th ** 2)[None, :] - np.eye(N)
    if clamp is not None:
        Jac[hub, :] = 0.0; Jac[hub, hub] = -1.0
    lead = float(np.max(np.linalg.eigvals(Jac).real))

    # robustness: kick the learned weights, relax, does the output stay in tol?
    robust = np.nan
    if converged:
        Wk = W.copy()
        Wk[active] += kick * np.sqrt(D) * rng.standard_normal(n_active) * np.sqrt(T_stop)
        xk = x.copy()
        ok = 1
        for _ in range(window):
            if clamp is not None:
                xk[hub] = clamp
            xk = xk + dt * (Wk @ np.tanh(xk) - xk)
            sk = abs(b @ xk - target)
            if (M_0 / 2.0) * (1.0 + np.tanh((sk - eps) / mu)) > tol:
                ok = 0
                break
        robust = float(ok)

    return dict(conv=converged, t_conv=t_conv, frac_fp=fp_steps / steps_done,
                lyap_end=lyap_end, n_frozen=n_frozen, hub_alive=hub_alive,
                lead_eig=lead, dW=float(np.linalg.norm(W - W0)), robust=robust,
                max_in=int(T.sum(1).max()), max_out=int(T.sum(0).max()),
                norm_t=norm_t, norm_pred=norm_pred, norm_times=times)


def worker(seed, kind, N, mean_k, sigma, alpha, learn, trial_kwargs, hub_noise_gain=1.0):
    rng = np.random.default_rng(seed)
    net = build(rng, kind, N, mean_k, sigma, alpha)
    r = run_trial(rng, net, learn=learn, hub_noise_gain=hub_noise_gain, **trial_kwargs)
    r.update(seed=seed, kind=kind, N=N, sigma=sigma, alpha=alpha, learn=learn)
    return r


def condition(rng, n_trials, n_jobs, trial_kwargs, **cfg):
    seeds = rng.integers(0, 2 ** 31 - 1, size=n_trials).tolist()
    return Parallel(n_jobs=n_jobs)(delayed(worker)(
        s, cfg["kind"], cfg["N"], cfg["mean_k"], cfg["sigma"], cfg["alpha"],
        cfg.get("learn", "all"), trial_kwargs, cfg.get("hub_noise_gain", 1.0)) for s in seeds)


def summarize(rs):
    a = lambda k: np.array([r[k] for r in rs], dtype=float)
    conv = a("conv")
    out = dict(n=len(rs), CF=conv.mean(),
               frac_fp=a("frac_fp").mean(), lyap_end=np.nanmean(a("lyap_end")),
               n_frozen=a("n_frozen").mean(), hub_alive=np.nanmean(a("hub_alive")),
               P_fp_any=float(np.mean(a("frac_fp") > 0.05)),
               lead_eig_conv=float(np.nanmean(np.where(conv == 1, a("lead_eig"), np.nan))),
               lead_eig_nonconv=float(np.nanmean(np.where(conv == 0, a("lead_eig"), np.nan))),
               dW_conv=float(np.nanmean(np.where(conv == 1, a("dW"), np.nan))),
               robust=float(np.nanmean(a("robust"))),
               t_conv_median=float(np.nanmedian(a("t_conv"))))
    fp = a("frac_fp") > 0.05
    out["CF_given_fp"] = float(conv[fp].mean()) if fp.any() else np.nan
    out["CF_given_nofp"] = float(conv[~fp].mean()) if (~fp).any() else np.nan
    ordered = a("lyap_end") < 0
    out["P_ordered"] = float(ordered.mean())
    out["CF_given_ordered"] = float(conv[ordered].mean()) if ordered.any() else np.nan
    out["CF_given_chaotic"] = float(conv[~ordered].mean()) if (~ordered).any() else np.nan
    return out


# --------------------------------------------------------------- experiments --

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=120)
    ap.add_argument("--mean-k", type=float, default=7.5)
    ap.add_argument("--n-trials", type=int, default=32)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--t-max", type=float, default=300.0)
    ap.add_argument("--only", nargs="*", default=None,
                    help="subset of experiments: E1 E3 E4 E5 E6")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    tk = dict(DEFAULT_TRIAL, t_max=args.t_max)
    rng = np.random.default_rng(args.seed)
    base = dict(N=args.N, mean_k=args.mean_k, alpha=0.5)
    want = lambda e: args.only is None or e in args.only
    results = {}
    raw = {}
    t0 = time.perf_counter()

    def run(name, **cfg):
        cfg = dict(base, **cfg)
        rs = condition(rng, args.n_trials, args.n_jobs, tk, **cfg)
        s = summarize(rs)
        results[name] = dict(cfg=cfg, **s)
        raw[name] = [{k: v for k, v in r.items() if not k.startswith("norm")} for r in rs]
        print(f"[{time.perf_counter() - t0:6.1f}s] {name:34s} CF={s['CF']:.2f} "
              f"fp={s['frac_fp']:.2f} P(fp)={s['P_fp_any']:.2f} "
              f"lyap={s['lyap_end']:+.2f} frozen={s['n_frozen']:5.1f} "
              f"alive={s['hub_alive']:.2f} CF|fp={s['CF_given_fp']:.2f} "
              f"CF|nofp={s['CF_given_nofp']:.2f} P(ord)={s['P_ordered']:.2f} "
              f"CF|ord={s['CF_given_ordered']:.2f} CF|chaos={s['CF_given_chaotic']:.2f}", flush=True)
        return rs

    # E1: hub variants x sigma (decomposition E2 comes for free from diagnostics)
    if want("E1"):
        print("== E1: hub variants vs sigma (alpha=0.5) ==")
        run("E1 er", kind="er", sigma=0.0)
        for kind in ("hub", "hub_forced", "hub_clamped"):
            for sigma in (2.0, 5.0, 10.0, 20.0, 40.0):
                run(f"E1 {kind} s={sigma:g}", kind=kind, sigma=sigma)
        for alpha in (0.25, 1.0):
            run(f"E1 hub_clamped s=20 a={alpha}", kind="hub_clamped", sigma=20.0, alpha=alpha)

    # E3: which weights need to move? (intrinsic dimension)
    if want("E3"):
        print("== E3: learning restricted to subspaces (hub_clamped, sigma=20) ==")
        for learn in ("all", "hub_only", "bulk_only", "none"):
            run(f"E3 {learn}", kind="hub_clamped", sigma=20.0, learn=learn)
        run("E3 er none", kind="er", sigma=0.0, learn="none")

    # E7: scale-matched exploration of the hub column (fair intrinsic-dimension test)
    if want("E7"):
        print("== E7: hub-only learning with noise scaled to the hub weight scale ==")
        g_eff = 10.0 / np.sqrt(args.mean_k)
        for sigma in (5.0, 20.0):
            gain = sigma / g_eff
            run(f"E7 hub_only scaled s={sigma:g}", kind="hub_clamped", sigma=sigma,
                learn="hub_only", hub_noise_gain=gain)
            run(f"E7 all scaled-hub s={sigma:g}", kind="hub_clamped", sigma=sigma,
                learn="all", hub_noise_gain=gain)
            run(f"E7 hub_only unscaled s={sigma:g}", kind="hub_clamped", sigma=sigma,
                learn="hub_only")

    # E4: scale-free in vs out degree, my own orientation convention
    if want("E4"):
        print("== E4: scale-free in- vs out-degree ==")
        for kind in ("sf_in", "sf_out", "er"):
            run(f"E4 {kind}", kind=kind, sigma=0.0)

    # E5: norm growth prediction and N scaling of convergence time
    if want("E5"):
        print("== E5: hub column norm growth + N scaling ==")
        rs = run("E5 hub_forced s=5 growth", kind="hub_forced", sigma=5.0)
        meas = [r["norm_t"][-1] - r["norm_t"][0] for r in rs]
        pred = [r["norm_pred"][-1] - r["norm_pred"][0] for r in rs]
        results["E5 growth"] = dict(
            n0=float(np.mean([r["norm_t"][0] for r in rs])),
            growth_measured=float(np.mean(meas)), growth_predicted=float(np.mean(pred)),
            corr=float(np.corrcoef(meas, pred)[0, 1]))
        print("   norm prediction rel. error:", results["E5 growth"])
        for N in (60, 120, 240):
            run(f"E5 hub_clamped s=20 N={N}", kind="hub_clamped", sigma=20.0, N=N)
            run(f"E5 er N={N}", kind="er", sigma=0.0, N=N)

    # E8: N scaling of every ensemble at equal mean degree
    if want("E8"):
        print(f"== E8: N scaling at mean_k={args.mean_k} ==")
        for N in (60, 120, 240, 480):
            for kind in ("er", "sf_in", "sf_out", "hub_clamped"):
                run(f"E8 {kind} N={N}", kind=kind, sigma=20.0, N=N)

    # E6: implicit bias / robustness of converged solutions (uses E1/E4 raw data)
    if want("E6"):
        print("== E6: stability of converged solutions (fresh runs) ==")
        run("E6 hub_clamped s=20", kind="hub_clamped", sigma=20.0)
        run("E6 sf_in", kind="sf_in", sigma=0.0)
        for name in ("E6 hub_clamped s=20", "E6 sf_in"):
            s = results[name]
            print(f"   {name}: lead_eig conv={s['lead_eig_conv']:+.3f} "
                  f"nonconv={s['lead_eig_nonconv']:+.3f} robust={s['robust']:.2f} "
                  f"|dW| conv={s['dW_conv']:.2f} t_conv med={s['t_conv_median']:.0f}")

    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, float) and np.isnan(o):
            return None
        if isinstance(o, np.generic):
            return clean(o.item())
        return o

    tag = f"N{args.N}_t{int(args.t_max)}_n{args.n_trials}"
    (OUT / f"summary_{tag}.json").write_text(json.dumps(clean(results), indent=1))
    (OUT / f"raw_{tag}.json").write_text(json.dumps(clean(raw)))
    print(f"wrote {OUT}/summary_{tag}.json  ({time.perf_counter() - t0:.0f}s)")


if __name__ == "__main__":
    main()
