"""
simulation.py
--------------
Reproduces the spirit of Section 6.4 of Khashanah et al. (2025): simulate a
self-exciting (Hawkes) jump-clustering process superposed on Gaussian white
noise, then compare Med9 vs the ABD test's ability to recover the *injected*
jump times as the clustering intensity increases. This is a controlled,
ground-truth-exact experiment (unlike the noisy 'Dummy Bivariate' column) and
is what lets us make a clean causal claim: "Med9 is more robust than ABD when
jumps cluster."
"""
import numpy as np


def simulate_hawkes_jump_times(n_days, M, lam0, gamma, beta, rng, dt=1.0):
    """Fast discrete-time recursive approximation of a 1-D self-exciting
    (Hawkes) jump process on a unit grid of n_days*M intervals. At each step
    the excitation state decays by exp(-beta*dt) and jumps by `gamma` every
    time a jump occurs in the previous interval:
        A[i] = exp(-beta*dt) * (A[i-1] + 1{jump at i-1})
        lambda[i] = lam0 + gamma * A[i]
        jump[i] ~ Bernoulli(min(lambda[i]*dt, 1))
    This has the same qualitative self-exciting-clustering behaviour as a
    true Hawkes process (jumps beget more jumps in the near future) and is
    what drives the contagion effect we want to test detectors against.
    Returns a boolean array of shape (n_days, M)."""
    total = n_days * M
    decay = np.exp(-beta * dt)
    A = 0.0
    jumps = np.zeros(total, dtype=bool)
    u = rng.uniform(size=total)
    for i in range(total):
        lam_i = lam0 + gamma * A
        p = min(lam_i * dt, 1.0)
        j = u[i] < p
        jumps[i] = j
        A = decay * (A + (1.0 if j else 0.0))
    return jumps.reshape(n_days, M)


def build_clustered_dataset(n_days, M, lam0, gamma, beta, base_sigma, jump_sigma_mult, seed):
    """Returns (returns[n_days,M], true_jump_grid[n_days,M])."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, base_sigma, size=(n_days, M))
    jump_grid = simulate_hawkes_jump_times(n_days, M, lam0, gamma, beta, rng)
    jump_sizes = rng.normal(0, base_sigma * jump_sigma_mult, size=(n_days, M))
    returns = noise + np.where(jump_grid, jump_sizes, 0.0)
    return returns, jump_grid
