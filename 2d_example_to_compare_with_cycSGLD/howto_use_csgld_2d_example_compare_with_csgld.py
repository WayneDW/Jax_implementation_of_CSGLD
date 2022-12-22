#!/usr/bin/env python3
import sys
import itertools

import jax
import jax.scipy as jsp
import jax.numpy as jnp

from typing import Callable, NamedTuple
from blackjax.types import Array, PRNGKey, PyTree

import matplotlib.pyplot as plt
import seaborn as sns


''' starting module for hyperparameter tuning '''

import argparse

parser = argparse.ArgumentParser(description='Grid search')
parser.add_argument('-lr',  default=0.01, type=float)
parser.add_argument('-zeta',  default=0.06, type=float)
parser.add_argument('-sz',  default=10, type=float)
parser.add_argument('-temperature',  default=1, type=float)
parser.add_argument('-num_partitions',  default=10000, type=int)
parser.add_argument('-energy_gap',  default=0.001, type=float)
parser.add_argument('-seed',  default=1, type=int)

pars = parser.parse_args()



lr = pars.lr
zeta = pars.zeta
sz = pars.sz
temperature = pars.temperature

cur_seed = pars.seed
### The following parameters partition the energy space and no tuning is needed. 
num_partitions = pars.num_partitions
energy_gap = pars.energy_gap
''' ending module for hyperparameter tuning '''



lmbda = 1/25
positions = [-4, -2, 0, 2, 4]
mu = jnp.array([list(prod) for prod in itertools.product(positions, positions)])
sigma = 0.03 * jnp.eye(2)

def logprob_fn(x, *_):
    return lmbda * jsp.special.logsumexp(
        jax.scipy.stats.multivariate_normal.logpdf(x, mu, sigma)
    )

def sample_fn(rng_key):
    choose_key, sample_key = jax.random.split(rng_key)
    samples = jax.random.multivariate_normal(sample_key, mu, sigma)
    return jax.random.choice(choose_key, samples)



## SGLD baseline 

import blackjax
from fastprogress import progress_bar

# 250k iterations
num_training_steps = 2500000
schedule_fn = lambda k: 0.05 * k ** (-0.55)
# TODO: There is no need to pre-compute the schedule
schedule = [schedule_fn(i) for i in range(1, num_training_steps+1)]

grad_fn = lambda x, _: jax.grad(logprob_fn)(x)
sgld = blackjax.sgld(grad_fn)

rng_key = jax.random.PRNGKey(cur_seed)
init_position = -10 + 20 * jax.random.uniform(rng_key, shape=(2,))

position = init_position

'''
sgld_samples = []
for i in progress_bar(range(num_training_steps)):
    _, rng_key = jax.random.split(rng_key)
    position = jax.jit(sgld)(rng_key, position, 0, schedule[i])
    sgld_samples.append(position)

sgld_samples = jnp.array(sgld_samples)

fig, scatter = plt.subplots(figsize = (20, 20), dpi = 100)
kde = sns.kdeplot(x=sgld_samples[:, 0], y=sgld_samples[:, 1],  cmap="Blues", fill=True, thresh=0.05, bw_method=0.15)
kde.set_xlim(left=-5, right=5)
kde.set_ylim(bottom=-5, top=5)
plt.savefig("./2d_figures/2d_sgld.pdf")
plt.close()
'''

### CSGLD part


class CSGLDState(NamedTuple):
    position: PyTree
    energy_pdf: Array
    energy_idx: int

'''
### specify hyperparameters (zeta and sz are the only two hyperparameters to tune)
lr = 1e-3
zeta = 2
sz = 10
temperature = 1

### The following parameters partition the energy space and no tuning is needed. 
num_partitions = 10000
energy_gap = 0.001
'''
min_energy = -2 # min energy -1.2 max energy around 0.067 ?


thinning_factor = 10

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

csgld_samples = [] 

print('csgld sampling\n')
for iter_ in progress_bar(range(num_training_steps)):
    rng_key, subkey = jax.random.split(rng_key)
    stepsize_SA = min(1e-2, (iter_+100)**(-0.8)) * sz
    state = jax.jit(csgld.step)(subkey, state, 0, lr, stepsize_SA)
    
    csgld_samples.append(state.position)

csgld_samples = jnp.array(csgld_samples)[1000: ][:: thinning_factor] # change to jnp format and remove warm-up samples + thinning

fig, scatter = plt.subplots(figsize = (20, 20), dpi = 100)
kde = sns.kdeplot(x=csgld_samples[:, 0], y=csgld_samples[:, 1],  cmap="Blues", fill=True, thresh=0.05, bw_method=0.15)
kde.set_xlim(left=-5, right=5)
kde.set_ylim(bottom=-5, top=5)
plt.savefig(f"./2d_figures/2d_csgld_before_resampling_lr_{lr}_zeta_{zeta}_sz_{sz}_T_{temperature}_Egap_{energy_gap}_num_part_{num_partitions}_seed_{cur_seed}.pdf")
plt.close()


if jnp.isnan(csgld_samples).sum() > 0:
    sys.exit('nan value exists')

print('csgld get index\n')
csgld_energy_idx_list = []
# get index of each particle for importance sampling
for position in progress_bar(csgld_samples):
    energy_value = logprob_fn(position)
    idx = jax.lax.min(jax.lax.max(jax.lax.floor((energy_value - min_energy) / energy_gap + 1).astype("int32"), 1,), num_partitions - 1, )
    csgld_energy_idx_list.append(idx)

csgld_energy_idx_list = jnp.array(csgld_energy_idx_list)


# pick important partitions for re-sampling
important_idx = jnp.where(state.energy_pdf > jnp.quantile(state.energy_pdf, 0.95))[0]
scaled_energy_pdf = state.energy_pdf[important_idx]**zeta / (state.energy_pdf[important_idx]**zeta).max()

print('csgld re-sampling\n')
# importance sampling via re-sampling
csgld_re_sample_list = jnp.empty([0, 2])
for _ in range(5):
    rng_key, subkey = jax.random.split(rng_key)
    for my_idx in important_idx:
        if jax.random.bernoulli(rng_key, p=scaled_energy_pdf[my_idx], shape=None) == 1:
            samples_in_my_idx = csgld_samples[csgld_energy_idx_list == my_idx]
            csgld_re_sample_list = jnp.concatenate(
                (csgld_re_sample_list, samples_in_my_idx), axis=0
            )

fig, scatter = plt.subplots(figsize = (20, 20), dpi = 100)
kde = sns.kdeplot(x=csgld_re_sample_list[:, 0], y=csgld_re_sample_list[:, 1],  cmap="Blues", fill=True, thresh=0.05, bw_method=0.15)
kde.set_xlim(left=-5, right=5)
kde.set_ylim(bottom=-5, top=5)
plt.savefig(f"./2d_figures/2d_csgld_after_resampling_lr_{lr}_zeta_{zeta}_sz_{sz}_T_{temperature}_Egap_{energy_gap}_num_part_{num_partitions}_seed_{cur_seed}.pdf")
plt.close()
