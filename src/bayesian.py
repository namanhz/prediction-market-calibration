"""NumPyro hierarchical Bayesian models (M0, M1, M2) and MCMC infrastructure."""
from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

# Hardware tuning (before importing JAX)
_n_threads = os.cpu_count() or 8
NUM_CHAINS = min(8, _n_threads)
os.environ.setdefault("XLA_FLAGS",
    f"--xla_force_host_platform_device_count={NUM_CHAINS} "
    "--xla_cpu_multi_thread_eigen=true"
)

import arviz as az
import jax
import jax.numpy as jnp
import jax.random as random
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, log_likelihood

numpyro.set_host_device_count(NUM_CHAINS)


# ═══════════════════════════════════════════════════════════════════
# Data preparation
# ═══════════════════════════════════════════════════════════════════

def prepare_arrays(df, platform_filter=None):
    """Convert DataFrame to JAX arrays for model fitting.

    Args:
        df: calibration data DataFrame with columns: category, logit_p, outcome, platform_id
        platform_filter: None for all data, 'kalshi' or 'polymarket' for subset

    Returns:
        dict with logit_p, y, cat_idx, platform, J, N, cat_names, df
    """
    if platform_filter:
        df = df[df["platform"] == platform_filter].copy()

    categories = sorted(df["category"].unique())
    cat_remap = {name: i for i, name in enumerate(categories)}
    cat_idx = df["category"].map(cat_remap).values.astype(np.int32)
    J = len(categories)

    logit_p = jnp.array(df["logit_p"].values.astype(np.float64))
    y = jnp.array(df["outcome"].values.astype(np.int32))
    platform = jnp.array(df["platform_id"].values.astype(np.int32))

    return {
        "logit_p": logit_p,
        "y": y,
        "cat_idx": jnp.array(cat_idx),
        "platform": platform,
        "J": J,
        "N": len(df),
        "cat_names": categories,
        "df": df,
    }


# ═══════════════════════════════════════════════════════════════════
# Model definitions
# ═══════════════════════════════════════════════════════════════════

def model_m0(logit_p, y=None):
    """M0: Pooled logistic regression, no hierarchy."""
    alpha = numpyro.sample("alpha", dist.Normal(0, 2))
    beta = numpyro.sample("beta", dist.Normal(1, 2))
    eta = alpha + beta * logit_p
    numpyro.sample("y_obs", dist.Bernoulli(logits=eta), obs=y)


def model_m1(logit_p, cat_idx, J, y=None):
    """M1: Category-level MVN random effects with LKJ correlation (Kalshi only)."""
    mu_alpha = numpyro.sample("mu_alpha", dist.Normal(0, 2))
    mu_beta = numpyro.sample("mu_beta", dist.Normal(1, 2))

    sigma_alpha = numpyro.sample("sigma_alpha", dist.Exponential(1))
    sigma_beta = numpyro.sample("sigma_beta", dist.Exponential(1))

    L_Omega = numpyro.sample("L_Omega", dist.LKJCholesky(2, concentration=2.0))
    sigma_vec = jnp.stack([sigma_alpha, sigma_beta])
    L_Sigma = jnp.diag(sigma_vec) @ L_Omega

    z = numpyro.sample("z", dist.Normal(0, 1).expand([J, 2]))
    mu_vec = jnp.stack([mu_alpha, mu_beta])
    effects = mu_vec + z @ L_Sigma.T

    alpha = numpyro.deterministic("alpha_j", effects[:, 0])
    beta = numpyro.deterministic("beta_j", effects[:, 1])

    eta = alpha[cat_idx] + beta[cat_idx] * logit_p
    numpyro.sample("y_obs", dist.Bernoulli(logits=eta), obs=y)


def model_m2(logit_p, cat_idx, platform, J, y=None):
    """M2: Category RE + platform fixed effect (full data)."""
    mu_alpha = numpyro.sample("mu_alpha", dist.Normal(0, 2))
    mu_beta = numpyro.sample("mu_beta", dist.Normal(1, 2))
    gamma = numpyro.sample("gamma", dist.Normal(0, 2))

    sigma_alpha = numpyro.sample("sigma_alpha", dist.Exponential(1))
    sigma_beta = numpyro.sample("sigma_beta", dist.Exponential(1))

    L_Omega = numpyro.sample("L_Omega", dist.LKJCholesky(2, concentration=2.0))
    sigma_vec = jnp.stack([sigma_alpha, sigma_beta])
    L_Sigma = jnp.diag(sigma_vec) @ L_Omega

    z = numpyro.sample("z", dist.Normal(0, 1).expand([J, 2]))
    mu_vec = jnp.stack([mu_alpha, mu_beta])
    effects = mu_vec + z @ L_Sigma.T

    alpha = numpyro.deterministic("alpha_j", effects[:, 0])
    beta = numpyro.deterministic("beta_j", effects[:, 1])

    eta = alpha[cat_idx] + gamma * platform + beta[cat_idx] * logit_p
    numpyro.sample("y_obs", dist.Bernoulli(logits=eta), obs=y)


# ═══════════════════════════════════════════════════════════════════
# MCMC fitting
# ═══════════════════════════════════════════════════════════════════

def fit_mcmc(model_fn, model_args, model_kwargs, n_obs=0, chains=4, draws=2000,
             warmup=1000, seed=42, label="model"):
    """Run NUTS MCMC and return (mcmc, trace)."""
    print(f"\n{'=' * 60}")
    print(f"Fitting {label}")
    print(f"{'=' * 60}")
    print(f"  N={n_obs}")
    print(f"  {chains} chains x {draws} draws + {warmup} warmup")
    print(f"  JAX devices: {jax.device_count()}, backend: {jax.default_backend()}")

    kernel = NUTS(model_fn, target_accept_prob=0.95)
    mcmc = MCMC(
        kernel,
        num_warmup=warmup,
        num_samples=draws,
        num_chains=chains,
        chain_method="parallel",
        progress_bar=True,
    )

    rng_key = random.PRNGKey(seed)
    mcmc.run(rng_key, *model_args, **model_kwargs)
    mcmc.print_summary(exclude_deterministic=False)

    trace = az.from_numpyro(mcmc)
    return mcmc, trace


def compute_log_lik(model_fn, mcmc, model_args, model_kwargs, thin=4):
    """Compute pointwise log-likelihood from posterior samples (thinned).

    Returns array of shape (chains, thinned_draws_per_chain, N).
    """
    chain_samples = mcmc.get_samples(group_by_chain=True)
    n_chains = list(chain_samples.values())[0].shape[0]

    chain_lls = []
    for c in range(n_chains):
        single = {k: v[c, ::thin, ...] for k, v in chain_samples.items()}
        ll = log_likelihood(model_fn, single, *model_args, **model_kwargs)
        chain_lls.append(np.asarray(ll["y_obs"]))

    return np.stack(chain_lls, axis=0)


def run_loo_comparison(results, label_a, label_b):
    """Compare two models via LOO-CV using ArviZ.

    Args:
        results: dict of {label: az.InferenceData with log_likelihood}
        label_a, label_b: names of models to compare

    Returns comparison DataFrame or None.
    """
    compare_dict = {label_a: results[label_a], label_b: results[label_b]}
    try:
        comparison = az.compare(compare_dict, ic="loo")
        print(f"\nLOO-CV Comparison: {label_a} vs {label_b}")
        print(comparison.to_string())
        return comparison
    except Exception as e:
        print(f"  LOO comparison failed: {e}")
        return None


def print_diagnostics(trace, var_names, label=""):
    """Print convergence diagnostics for specified variables."""
    print(f"\n--- Diagnostics: {label} ---")
    summary = az.summary(trace, var_names=var_names, hdi_prob=0.90)
    print(summary.to_string())

    sampled_vars = [v for v in var_names if v not in ("alpha_j", "beta_j")]
    if sampled_vars:
        s = az.summary(trace, var_names=sampled_vars, hdi_prob=0.90)
        rhat_ok = (s["r_hat"] < 1.01).all()
        ess_ok = (s["ess_bulk"] > 400).all()
        print(f"  R-hat < 1.01: {'PASS' if rhat_ok else 'FAIL'}")
        print(f"  ESS_bulk > 400: {'PASS' if ess_ok else 'FAIL'}")

    return summary
