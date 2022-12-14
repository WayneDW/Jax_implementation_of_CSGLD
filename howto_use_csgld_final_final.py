#!/usr/bin/env python3
import jax
import jax.scipy as jsp
import jax.numpy as jnp

import blackjax
import blackjax.sgmcmc.gradients as gradients

from typing import Callable, NamedTuple
from blackjax.types import Array, PRNGKey, PyTree

import matplotlib.pyplot as plt

'''
Sampling in big data problems is fundamentally limited by the multi-modality of the target distributions, with extremely high energy barriers. Multi-modality is often empirically solved via cyclical learning rates or different initializations (parallel chains).

Contour SgLD takes a different approach altogether: the algorithms learns the energy landscape with sampling, and uses this approximation to effectively integrate the diffusion on a flat landscape, before using the importance weight to reweigh the obtained samples.

In this notebook we will compare the performance of SGLD and Contour SGLD on a simple bimodal gaussian target. This example looks fundamental, but is rather challenging to solve.
'''


'''
## The model

Let us first generate data points that follow a gaussian mixture distributions, a relatively simple example yet hard enought for most algorithms to fail to recover the two modes.
'''
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
    loglikelihood_fn_1 = jax.scipy.stats.multivariate_normal.logpdf(x, position, sigma**2)
    loglikelihood_fn_2 = jax.scipy.stats.multivariate_normal.logpdf(x, -position + gamma, sigma**2)
    return jsp.special.logsumexp(jnp.array([loglikelihood_fn_1, loglikelihood_fn_2])) + jnp.log(0.5)

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
total_iter = 50_000


temperature = 50
lr = 1e-3
thinning_factor = 100



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
        
        print(f'iter {iter_/1000:.0f}k/{total_iter/1000:.0f}k position {state.position: .2f} energy {energy_value: .2f}')


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



# 3. CSGLD baseline
class CSGLDState(NamedTuple):
    position: PyTree
    energy_pdf: Array
    energy_idx: int


### specify hyperparameters (zeta and sz are the only two hyperparameters to tune)
zeta = 2
sz = 10

### The following parameters partition the energy space and no tuning is needed. 
num_partitions = 100000
energy_gap = 0.25
min_energy = 0 #81 # an estimate of the minimum energy, should be strictly lower than the exact one. !!!!!!!! more comment needed

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

csgld_sample_list, csgld_energy_idx_list = jnp.array([]), jnp.array([])

for iter_ in range(total_iter):
    rng_key, subkey = jax.random.split(rng_key)
    stepsize_SA = min(1e-2, (iter_+100)**(-0.8)) * sz
    data_batch = jax.random.shuffle(rng_key, X_data)[:batch_size, :]
    state = jax.jit(csgld.step)(subkey, state, data_batch, lr, stepsize_SA)
    if iter_ % thinning_factor == 0:
        energy_value = energy_fn(state.position, data_batch)
        csgld_sample_list = jnp.append(csgld_sample_list, state.position)
        ### For re-sampling only.
        idx = jax.lax.min(jax.lax.max(jax.lax.floor((energy_value - min_energy) / energy_gap + 1).astype("int32"), 1,), num_partitions - 1, )
        csgld_energy_idx_list = jnp.append(csgld_energy_idx_list, idx)
        print(f'iter {iter_/1000:.0f}k/{total_iter/1000:.0f}k position {state.position: .2f} energy {energy_value: .2f} idx {idx}')

### Make plots for CSGLD trajectory
plt.plot(csgld_sample_list, label='CSGLD')
plt.plot(sgld_sample_list, label='SGLD')
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
important_idx = jnp.where(state.energy_pdf > jnp.quantile(state.energy_pdf, 0.95))[0]
scaled_energy_pdf = state.energy_pdf[important_idx]**zeta / (state.energy_pdf[important_idx]**zeta).max()

csgld_re_sample_list = jnp.array([])
for _ in range(5):
    rng_key, subkey = jax.random.split(rng_key)
    for my_idx in important_idx:
        if jax.random.bernoulli(rng_key, p=scaled_energy_pdf[my_idx], shape=None) == 1:
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

'''
Since the energy density is crucial for us to build a flat density, let's take a look at state.energy_pdf. For illustration purposes, we smooth out fluctations and focus on the energy range from 3700 to 100000, which covers the major part of sample space.
'''

smooth_energy_pdf = jnp.convolve(state.energy_pdf, jsp.stats.norm.pdf(jnp.arange(-100, 101), scale=10), mode='same')
interested_idx = jax.lax.floor((jnp.arange(3700, 10000)) / energy_gap).astype('int32') # min 3681
plt.plot(jnp.arange(num_partitions)[interested_idx]*energy_gap, smooth_energy_pdf[interested_idx])
plt.xlabel(f'Energy')
plt.ylabel('Energy Density')
plt.savefig(f'./howto_use_csgld_CSGLD_energy_pdf_T{temperature}_zeta{zeta}_iter{total_iter}_sz{sz}_seed{mySeed}_v2.pdf')
plt.close()

'''
From the figure above, we see that low-energy regions usually lead to much higher probability mass. Moreover, the slope is negative with a higher scale in low energy regions. In view of Eq.(8) in the paper: https://proceedings.neurips.cc/paper/2020/file/b5b8c484824d8a06f4f3d570bc420313-Paper.pdf, we can expect a **negative learning rate** to help the particle escape the local trap. Eventually, a particle is able to bounce out of the deep local traps freely instead of being absorbed into it.
'''

# 3.4 Conclusions and limitations
'''
Admittedly, this algorithm is a little sophisticated due to the need to partition the energy space; Learning energy pdf also makes this algorithm delicate and leads to a large variance. However, considering the great promise to escape deep local traps in a principled sampling framework without using any tricks (cyclical learning rates or different initializations), these efforts are all well-deserved. The variance-reduced version is studied in this work https://arxiv.org/pdf/2202.09867.pdf. For versions with better stability, feel free to contact us if you have any suggestions.
'''



