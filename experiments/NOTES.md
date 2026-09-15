# Why does structure let exploratory adaptation converge? Small-N findings

Script: `experiments/hub_mechanism.py`. Logs and JSON in `experiments/results/`.
Convention: `J[i, j]` is edge j -> i, in-degree = row sums. Gain g_0 = 10,
D = 1e-3, eps = 3, t_max = 300, T_stop = 50, N = 120 unless stated.
40 trials per condition unless stated (standard error about 0.07 on CF).

## 1. The lumped hub as written is mostly dead

`hub` (self-loop with prob. alpha, weight N(0, sigma^2)) is alive in only
10-45 % of trials and never reaches CF above 0.23. Forcing the self-loop or
clamping the hub state to a constant gives CF 0.30-0.35 at sigma 5-20.
The alpha dependence in the original sweep mostly measures self-loop presence.

## 2. Convergence is chaos suppression, and it is non-monotonic in field strength

P(trailing Lyapunov exponent < 0) rises monotonically with sigma
(forced hub: 0.20 -> 0.45 -> 0.45 -> 0.75 -> 0.90 for sigma 2, 5, 10, 20, 40).
CF does not: forced hub CF is 0.05, 0.33, 0.33, 0.35, 0.10. At sigma 40 the
network is ordered 90 % of the time but CF given ordered is 0.08: the field
pins the fixed point so hard that the readout cannot be moved into the
window. Same with coverage: alpha 1.0 gives CF 0.15 vs 0.28-0.30 for 0.25-0.5.
So the useful structure is a field strong enough to quench chaos but weak
enough that the bulk keeps free directions for the walk to exploit.

## 3. Learning barely moves anything on this horizon

Over t_max = 300 the total weight displacement of converged trials is about
10 % of the weight norm, and the hub-column norm grows by 3 %
(measured 42, predicted from N_out * D * integral(M) = 43). The
"deterministic annealing of the field strength" idea is correct in
principle but irrelevant at these D and t_max. Without any learning at all
CF is 0.10 (clamped hub) vs 0.03 (Erdos-Renyi): part of the effect is that a
good fixed point already exists at initialization; learning roughly triples it.

## 4. Intrinsic dimension: the hub column alone suffices, if the noise is scale-matched

Learning restricted to the 60 hub-out weights with the default D gives CF
0.15-0.25 (the walk is too small relative to sigma = 20). Scaling the hub
noise by sigma / g_eff gives CF 0.38, equal to learning all 900 weights
(0.38). So one bias-like column of weights is a sufficient search space,
and exploratory adaptation is effectively a random walk on a bias vector.
Bulk-only learning (0.47) is at least as good as all (0.33), so the hub's
role is as a static field, not as a learnable parameter.

## 5. Scale-free in vs out: both help at N = 120, but sparsity is a confound

160 trials: sf_in 0.32, sf_out 0.34, Erdos-Renyi(k = 7.5) 0.06,
Erdos-Renyi(k = 3.5, the mean degree of the scale-free ensembles) 0.26.
At N = 120 most of the scale-free gain over the dense baseline is sparsity.
The heavy tail only shows up in the N scaling below.

## 6. N scaling separates the ensembles (mean degree 3.5, 60 trials, sigma = 20)

| N   | Erdos-Renyi | sf_in | sf_out | clamped hub |
|-----|-------------|-------|--------|-------------|
| 60  | 0.28        | 0.20  | 0.37   | 0.35        |
| 120 | 0.28        | 0.27  | 0.40   | 0.40        |
| 240 | 0.12        | 0.32  | 0.30   | 0.33        |
| 480 | 0.00        | 0.13  | 0.33   | 0.42        |

P(ordered) for the clamped hub is 0.42-0.52 at every N, as a mean-field
static-input picture predicts. Erdos-Renyi CF vanishes with N. Out-degree
hubs (sf_out) and the single clamped hub stay flat. In-degree heavy tails
(sf_in) drop at N = 480 with P(ordered) falling 0.68 -> 0.35, so at
least in this convention broadcast structure is what survives large N.
This is the small-N analogue of the (sf, binom) vs (binom, sf) asymmetry in
`CF_results.txt`; which index that table calls "in" still needs checking
against the graph-tool adjacency orientation.

## 7. Converged solutions are not marginal

Leading Jacobian eigenvalue at converged fixed points is -0.70 (both clamped
hub and sf_in), more negative than at non-converged endpoints (-0.38, -0.62),
and 70 % of converged solutions survive a weight kick. The first-passage
"edge of stability" hypothesis is not supported. Around 20-40 % of
converged clamped-hub trials have a positive trailing Lyapunov exponent:
the readout can sit inside the tolerance window while the network is still
weakly chaotic in directions the readout does not see.

## What to do next

- Repeat section 6 at N = 1000-1500 with the JAX code to see whether sf_in
  really decays and whether sf_out stays flat.
- Vary sigma * sqrt(alpha) jointly to map the CF ridge from section 2 and
  compare it with the static-input chaos boundary (Rajan, Abbott, Sompolinsky 2010).
- Run longer horizons (t_max 3000+) with larger D to test whether the
  norm-growth annealing mechanism takes over.
