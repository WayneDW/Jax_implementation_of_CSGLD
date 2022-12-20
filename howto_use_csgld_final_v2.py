#!/usr/bin/env python3
import jax
import jax.numpy as jnp

import blackjax
import blackjax.sgmcmc.gradients as gradients

from typing import Callable, NamedTuple
from blackjax.types import Array, PRNGKey, PyTree

import matplotlib.pyplot as plt

### set seeds
mySeed = 888
rng_key = jax.random.PRNGKey(mySeed)
rng_key, data_key0, data_key1, data_key2 = jax.random.split(rng_key, 4)

# 0. A baby data setup, but quite hard to simulate.

### specify data and batch size 
data_size = 1000
batch_size = 100


### create a data that follows a Gaussian mixture distribution
sigma, mu, gamma = 5., -5., 20.
prob_mixture = jax.random.bernoulli(data_key0, p=0.5, shape=(data_size, 1))

mixture_1 = jax.random.normal(data_key1, shape=(data_size, 1)) * sigma + mu
mixture_2 = jax.random.normal(data_key2, shape=(data_size, 1)) * sigma + gamma - mu

X_data = prob_mixture * mixture_1 + (1 - prob_mixture) * mixture_2

### set the initial position
init_position = 10.

# 1. Build energy function
### set the data likelihood
def logprior_fn(position):
    return 0

def loglikelihood_fn(position, x):
    mixture_1 = jax.scipy.stats.norm.pdf(x, loc=position, scale=sigma)
    mixture_2 = jax.scipy.stats.norm.pdf(x, loc=-position + gamma, scale=sigma)
    return jnp.log(0.5 * mixture_1 + 0.5 * mixture_2).sum()

### compute the energy function
def energy_fn(position, minibatch):
    logprior = logprior_fn(position)
    batch_loglikelihood = jax.vmap(loglikelihood_fn, in_axes=(None, 0))
    return -logprior - data_size * jnp.mean(
        batch_loglikelihood(position, minibatch), axis=0
    )


### Build the log-probability and gradient functions
logprob_fn, grad_fn = gradients.logprob_and_grad_estimator(
    logprior_fn, loglikelihood_fn, data_size
)


# 2. SGLD baseline
### specify hyperparameters for SGLD
total_iter = 40_000


temperature = 50
lr = 1e-3
thinning_factor = 100


'''
""" SGLD module (CSGLD with zeta=0 is equivalent to SGLD, if you have a better way to use SGLD, feel free to use that one) """
sgld = blackjax.csgld(
    logprob_fn,
    grad_fn,
    zeta=0,  # can be specified at each step in lower-level interface
    temperature=temperature,
)

### Initialize and take one step using the vanilla SGLD algorithm
state = sgld.init(init_position)
sgld_sample_list = jnp.array([])
for iter_ in range(total_iter):
    rng_key, subkey = jax.random.split(rng_key)
    data_batch = jax.random.shuffle(rng_key, X_data)[: batch_size, :]
    state = jax.jit(sgld.step)(subkey, state, data_batch, lr, 0)
    if iter_ % thinning_factor == 0:
        energy_value = energy_fn(state.position, data_batch)
        sgld_sample_list = jnp.append(sgld_sample_list, state.position)
        print(f'iter {iter_/1000:.0f}k/{total_iter/1000:.0f}k position {state.position: .2f}')


### Make plots for SGLD trajectory
plt.plot(sgld_sample_list, label='SGLD')
plt.xlabel(f'Iterations (x{thinning_factor})')
plt.ylabel('X')
plt.legend()
plt.title('SGLD in sample trajectory')
plt.savefig(f'./howto_use_csgld_SGLD_trajectory_T{temperature}_iter{total_iter}_seed{mySeed}_v2.pdf')
plt.close()

### Make plots for SGLD sample histogram
plt.hist(sgld_sample_list, 100)
plt.legend()
plt.xlabel(f'X')
plt.ylabel('Frequency')
plt.xlim(left=-15, right=35)
plt.title('SGLD distribution')
plt.savefig(f'./howto_use_csgld_SGLD_distributions_T{temperature}_iter{total_iter}_seed{mySeed}_v2.pdf')
plt.close()
'''


# 3. CSGLD baseline
class CSGLDState(NamedTuple):
    position: PyTree
    energy_pdf: Array
    energy_idx: int


### specify hyperparameters (zeta and sz are the only two hyperparameters to tune)
zeta = 2 
sz = 10

### The following parameters partition the energy space and no tuning is needed. 
num_partitions = 50000
energy_gap = 0.25
domain_radius = 50 # restart sampling when the particle explores too deep over the tails and leads to nan.
min_energy = 3000 # an estimate of the minimum energy, should be strictly lower than the exact one.

csgld = blackjax.csgld(
    logprob_fn,
    grad_fn,
    zeta=zeta,  # can be specified at each step in lower-level interface
    temperature=temperature,  # can be specified at each step
    num_partitions=num_partitions,  # cannot be specified at each step
    energy_gap=energy_gap,  # cannot be specified at each step
    min_energy=min_energy, 
)

## 3.1 Simulate via the CSGLD algorithm
state = csgld.init(init_position)

re_start_counter = 0 # in case the particle goes beyond the domain
csgld_sample_list, csgld_energy_idx_list = jnp.array([]), jnp.array([])
for iter_ in range(total_iter):
    rng_key, subkey = jax.random.split(rng_key)
    stepsize_SA = min(1e-2, (iter_+100)**(-0.8)) * sz

    data_batch = jax.random.shuffle(rng_key, X_data)[:batch_size, :]
    state = jax.jit(csgld.step)(subkey, state, data_batch, lr, stepsize_SA)
    if jnp.abs(state.position) > domain_radius:
        re_start_counter += 1
        state = CSGLDState(jax.random.uniform(rng_key, minval=-domain_radius, maxval=domain_radius), \
                           state.energy_pdf, \
                           state.energy_idx)

    if iter_ % thinning_factor == 0:
        energy_value = energy_fn(state.position, data_batch)
        csgld_sample_list = jnp.append(csgld_sample_list, state.position)
        ### For re-sampling only.
        idx = jax.lax.min(jax.lax.max(jax.lax.floor((energy_value - min_energy) / energy_gap + 1).astype("int32"), 1,), num_partitions - 1,)
        csgld_energy_idx_list = jnp.append(csgld_energy_idx_list, idx)
        print(f'iter {iter_/1000:.0f}k/{total_iter/1000:.0f}k position {state.position: .2f} energy {energy_value: .2f} re-restart counter {re_start_counter}')

### Make plots for CSGLD trajectory
plt.plot(csgld_sample_list, label='CSGLD')
#plt.plot(sgld_sample_list, label='SGLD')
plt.xlabel(f'Iterations (x{thinning_factor})')
plt.ylabel('X')
plt.legend()
plt.title('CSGLD v.s. SGLD in sample trajectory')
plt.savefig(f'./howto_use_csgld_CSGLD_trajectory_T{temperature}_zeta{zeta}_iter{total_iter}_sz{sz}_seed{mySeed}_v2.pdf')
plt.close()

### Make plots for CSGLD sample histogram before re-sampling
plt.hist(csgld_sample_list, 200)
plt.xlabel(f'X')
plt.ylabel('Frequency')
plt.legend()
plt.xlim(left=-15, right=35)
plt.title('CSGLD distribution before re-sampling')
plt.savefig(f'./howto_use_csgld_CSGLD_distributions_before_T{temperature}_zeta{zeta}_iter{total_iter}_sz{sz}_seed{mySeed}_v2.pdf')
plt.close()


# 3.2 Re-sampling via importance sampling (state.energy_pdf ** zeta)
# ==============================================================================================================
scaled_energy_pdf = (state.energy_pdf / state.energy_pdf.max())  # state.energy_pdf ** zeta is not stable
normalized_energy_pdf = (scaled_energy_pdf**zeta) / (scaled_energy_pdf**zeta).sum()

# pick important partitions and ignore the rest
non_trivial_idx = jnp.where(normalized_energy_pdf > jnp.quantile(normalized_energy_pdf, 0.95))[0]
scaled_density = normalized_energy_pdf / normalized_energy_pdf[non_trivial_idx].max()

csgld_re_sample_list = jnp.array([])
for my_idx in non_trivial_idx:
    if jax.random.bernoulli(rng_key, p=scaled_density[my_idx], shape=None) == 1:
        samples_in_my_idx = csgld_sample_list[csgld_energy_idx_list == my_idx]
        csgld_re_sample_list = jnp.concatenate(
            (csgld_re_sample_list, samples_in_my_idx)
        )

### Make plots for CSGLD sample histogram after re-sampling
plt.hist(csgld_re_sample_list, 200)
plt.xlabel(f'X')
plt.ylabel('Frequency')
plt.legend()
plt.xlim(left=-15, right=35)
plt.title('CSGLD distribution after re-sampling')
plt.savefig(f'./howto_use_csgld_CSGLD_distributions_after_T{temperature}_zeta{zeta}_iter{total_iter}_sz{sz}_seed{mySeed}_v2.pdf')
plt.close()


# 3.3 Analyze why CSGLD works

plt.plot(normalized_energy_pdf)
plt.xlabel(f'Partition index')
plt.ylabel('Density')
plt.legend()
plt.xlim(left=-15, right=35)
plt.title('Normalized energy PDF')
plt.savefig(f'./howto_use_csgld_CSGLD_energy_pdf_T{temperature}_zeta{zeta}_iter{total_iter}_sz{sz}_seed{mySeed}_v2.pdf')
plt.close()
