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

## 8. Dimensionality predicts convergence per trial (E9, N = 240, k = 3.5, 80 trials)

Participation ratio (PR) measured at t = 20-120 orders the ensembles as in the
earlier analysis: Erdos-Renyi 6.0, sf_in 4.3, sf_out 3.1, clamped hub 3.0.
Pooled across ensembles, low PR predicts convergence with AUC 0.76
(Lyapunov exponent: 0.81), and it keeps predicting inside the ordered
subset (0.70) and the chaotic subset (0.85). The overlap of the readout with
the activity covariance is weakly predictive early (0.61); by the end,
converged trials have readout variance about 0.6 of the isotropic
expectation vs about 1.0 for non-converged ones.

## 9. But CF does not collapse onto PR (E10, N = 240, 60 trials, PR from a no-learning probe)

| condition        | probe PR | CF   |
|------------------|----------|------|
| Erdos-Renyi      | 5.95     | 0.10 |
| 1 clamped hub    | 1.60     | 0.32 |
| 2 clamped hubs   | 1.13     | 0.28 |
| 4 clamped hubs   | 1.08     | 0.08 |
| 8 clamped hubs   | 1.05     | 0.18 |
| sf_out gamma 2.2 | 2.49     | 0.38 |
| sf_out gamma 2.4 | 2.29     | 0.32 |
| sf_out gamma 3.0 | 2.83     | 0.27 |
| sf_out gamma 4.0 | 2.65     | 0.42 |
| sf_in gamma 2.2  | 3.61     | 0.15 |
| sf_in gamma 2.4  | 3.79     | 0.22 |

Pooled CF by PR sextile: 0.20, 0.28, 0.45, 0.24, 0.18, 0.13 for PR from 1.0
up to 12. The relation is non-monotonic: convergence peaks at PR about
1.2-2 and falls on both sides. Adding more clamped hubs drives PR to 1 (the
network is pinned to a fixed point) and convergence collapses, because the
readout can no longer be moved. Within an ensemble lower PR still helps
(sf_out converged trials have PR 1.9 vs 2.9). Caveat: the gamma sweep also
changes mean degree (6, 3.5, 2, 1.5), so it is not a clean tail-only sweep.

Reading: low dimensionality is necessary but not sufficient. The learner
needs an attractor whose readout is quiet in time yet still sensitive to
the weights the walk is moving. The natural next quantity is the pair
(temporal variance of y, sensitivity of the fixed-point readout to weight
perturbations), measured in the same probe.

## 10. What is special about out-hubs: leverage (E11, N = 240, k = 3.5, 60 trials)

Learning restricted to subsets of weights. "hub in-edges" = incoming edges of
the three highest out-degree nodes (sf_out, er) or highest in-degree nodes (sf_in).

| ensemble | all weights | hub in-edges only | all except hub in-edges | hub out-edges only | none |
|----------|-------------|-------------------|-------------------------|--------------------|------|
| sf_out   | 0.37 (766)  | 0.20 (9)          | 0.28 (756)              | 0.18 (190)         | 0.08 |
| sf_in    | 0.25 (748)  | 0.13 (153)        | 0.27 (598)              | 0.05 (8)           | 0.07 |
| er       | 0.10 (829)  | 0.03 (10)         | 0.15 (835)              | 0.03 (27)          | 0.05 |

(number of learnable weights in parentheses)

In sf_out, nine weights, the in-edges of the three biggest broadcasters,
recover almost half of the full learning gain. The same nine-ish weights in
Erdos-Renyi, whose top out-degree is about 10 rather than 50+, do nothing.
In sf_in the in-hubs' own out-edges do nothing, and their 153 in-edges do
little: a high in-degree node is a saturated sink whose state barely responds
to any single weight.

Reading: an out-hub is a dynamic latent variable with ordinary in-degree
(about 3.5 inputs) and very high out-degree. Each of its few in-weights
changes its state by O(1), and its state is broadcast to a large fraction
of the network. Leverage per weight scales like k_out / sqrt(k_in). Heavy-
tailed out-degree with independent in-degree maximizes it; heavy-tailed
in-degree minimizes it; clamped hubs have none (no in-edges). Low PR is the
observability half (the readout is quiet); leverage is the controllability
half (few weights move it). sf_out is the structure that has both.

A single dynamic hub (3 in-edges, 120 out-edges, sigma 20) is too strong a
common drive at this sigma: CF 0.13 for all weights, but the same 0.13 with
only its 3 in-edges learnable, again showing that those weights are where
the leverage is.
