import itertools

import jax
import jax.scipy as jsp
import jax.numpy as jnp


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

import jax
from jax import numpy as jnp

import blackjax
from fastprogress import progress_bar

# 50k iterations
num_training_steps = 10000
schedule_fn = lambda k: 0.05 * k ** (-0.55)
# TODO: There is no need to pre-compute the schedule
schedule = [schedule_fn(i) for i in range(1, num_training_steps+1)]

grad_fn = lambda x, _: jax.grad(logprob_fn)(x)
sgld = blackjax.sgld(grad_fn)

rng_key = jax.random.PRNGKey(3)
init_position = -10 + 20 * jax.random.uniform(rng_key, shape=(2,))

position = init_position
sgld_samples = []
for i in progress_bar(range(num_training_steps)):
    _, rng_key = jax.random.split(rng_key)
    position = jax.jit(sgld)(rng_key, position, 0, schedule[i])
    sgld_samples.append(position)


import matplotlib.pyplot as plt
import seaborn as sns

sgld_samples = jnp.array(sgld_samples)

sns.kdeplot(x=sgld_samples[:, 0], y=sgld_samples[:, 1],  cmap="Blues", fill=True, thresh=0.05, bw_method=0.15)

plt.savefig("./2d_sgld.pdf")
plt.close()
