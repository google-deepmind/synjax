# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utils for perturbation based sampling."""
# pylint: disable=g-multiple-import
# pylint: disable=g-importing-member
# pylint: disable=g-long-lambda

from functools import partial
from typing import Union, Callable, Literal
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PyTree
from synjax._src.typing import Key, Shape, typed
from synjax._src.utils.special import tadd, tsub, tscale_inexact_arrays


@typed
def sample_sum_of_gamma(key: Key, shape: Shape, k: Union[int, jax.Array],
                        s: int = 10) -> Float[Array, "..."]:
  """Decomposition of Gumbel distribution into k sub-parts from Niepert et al.

  Adding k of Sum-of-Gamma independent samples produces 1 Gumbel sample.

  Args:
    key: PRNGKey for sampling.
    shape: Shape of the returned sample.
    k: Number of parts that when combined should be the same as Gumbel sample.
    s: Approximation parameter for Sum-of-Gamma perturbation.
       Higher value implies better approximation for the cost of using
       more memory and computation.
  Returns:
    A sample from Sum-of-Gamma distribution.
  References:
    Niepert et al, 2021: https://arxiv.org/pdf/2106.01798.pdf#page=6
  """
  k = jnp.asarray(k)
  k = jnp.expand_dims(k, axis=range(-len(shape)+k.ndim, 0))
  alpha = 1./k[..., None]
  beta = k[..., None]/jnp.arange(1, s+1, dtype=jnp.float32)
  samples = jax.random.gamma(key, alpha, shape + (s,)) * beta
  samples = (jnp.sum(samples, axis=-1) - jnp.log(s)) / k
  return samples


@typed
def build_noise_fn(
    noise: Union[Literal["Sum-of-Gamma", "Gumbel", "None"],
                 Callable[..., Array]],
    parts: int, sum_of_gamma_s: int) -> Callable[[Key, Shape], PyTree]:
  """Builds function that for a given key and shape produces noise sample."""
  if noise == "Sum-of-Gamma":
    return partial(sample_sum_of_gamma, k=parts, s=sum_of_gamma_s)
  elif noise == "Gumbel":
    return jax.random.gumbel
  elif noise == "None":
    return lambda key, shape: jnp.zeros(shape)
  elif isinstance(noise, Callable):
    return noise
  else:
    raise NotImplementedError


@typed
def noise_for_pytree(key: Key, noise_fn, tree: PyTree) -> PyTree:
  """Splits the key for each leaf and samples noise with noise_fn."""
  leaves, tree_def = jax.tree_util.tree_flatten(tree)
  keys = jax.random.split(key, len(leaves))
  zero = jnp.reshape(0*key, (-1,))[0]
  new_leaves = [
      noise_fn(akey, leaf.shape) if eqx.is_inexact_array(leaf)
      else jnp.full(leaf.shape, zero, dtype=leaf.dtype)
      for akey, leaf in zip(keys, leaves)]
  return jax.tree_util.tree_unflatten(tree_def, new_leaves)


@typed
def implicit_mle(*, noise_fn, argmax_fn: Callable[[PyTree], PyTree],
                 internal_learning_rate: Float[Array, ""],
                 temperature: Float[Array, ""]
                 ) -> Callable[[Key, PyTree], PyTree]:
  """Implicit Maximum-Likelihood Estimation from Niepert et al.

  Implicit MLE allows for flow of gradient trough discrete structure sampled by
  perturbation. Show in Algorithm 1 in Niepert et al (2021).

  Args:
    noise_fn: Noise function that accepts key and shape parameters.
    argmax_fn: Argmax function.
    internal_learning_rate: Internal learning rate used for Implicit-MLE update.
                            Niepert et al. refer to it as lambda parameter.
    temperature: Temperature for the added noise.
  Returns:
    New sampling function that accepts a key and parameters and
    returns differentiable discrete sample of the same shape as parameters.
  References:
    Niepert et al, 2021: https://arxiv.org/pdf/2106.01798.pdf#page=4
  """
  @jax.custom_vjp
  def sampling(key: Key, theta: PyTree) -> PyTree:
    return argmax_fn(tadd(theta, noise_for_pytree(key, noise_fn, theta)))
  def sampling_forward(key: Key, theta: PyTree) -> PyTree:
    eta = tscale_inexact_arrays(temperature,
                                noise_for_pytree(key, noise_fn, theta))
    z_hat = argmax_fn(tadd(theta, eta))
    return z_hat, (eta, theta, z_hat)
  def sampling_backward(residuals, g) -> PyTree:
    eta, theta, z_hat = residuals
    theta_hat = tsub(theta, tscale_inexact_arrays(internal_learning_rate, g))
    return None, tsub(z_hat, argmax_fn(tadd(theta_hat, eta)))
  sampling.defvjp(sampling_forward, sampling_backward)
  return sampling
