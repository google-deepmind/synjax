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

"""Tests for HMM."""
from absl.testing import absltest
import distrax
import jax
import jax.numpy as jnp
from synjax._src import distribution_test
from synjax._src import hmm
from tensorflow_probability.substrates import jax as tfp


def chain_is_connected(samples) -> bool:
  scores = jax.lax.associative_scan(jnp.matmul, samples, axis=-3)
  scores = scores[..., -1, :, :].sum((-1, -2))
  return jnp.all(scores == 1)


def tfp_marginals(
    init_logits: jax.Array, transition_logits: jax.Array,
    emission_dist: ..., observations: jax.Array) -> jax.Array:
  tfd = tfp.distributions
  initial_distribution = tfd.Categorical(logits=init_logits)
  transition_distribution = tfd.Categorical(logits=transition_logits)
  observation_distribution = tfd.Categorical(logits=emission_dist)
  model = tfd.HiddenMarkovModel(
      initial_distribution=initial_distribution,
      transition_distribution=transition_distribution,
      observation_distribution=observation_distribution,
      num_steps=observations.shape[-1])
  return jnp.exp(model.posterior_marginals(observations).logits)


def distrax_marginals(
    init_logits: jax.Array, transition_logits: jax.Array,
    emission_dist: ..., observations: jax.Array) -> jax.Array:
  observation_distribution = distrax.Categorical(logits=emission_dist)
  dhmm = distrax.HMM(
      trans_dist=distrax.Categorical(logits=transition_logits),
      init_dist=distrax.Categorical(logits=init_logits),
      obs_dist=observation_distribution)
  return dhmm.forward_backward(observations)[2]


class HMMTest(distribution_test.DistributionTest):

  def create_random_batched_dists(self, key):
    b, n, t = 3, 6, 4
    keys = jax.random.split(key, 5)
    dists = []
    kwargs = dict(
        init_logits=jax.random.uniform(keys[0], (b, t)),
        transition_logits=jax.random.uniform(keys[1], (b, t, t)))
    for is_categorical in [True, False]:
      kwargs = dict(
          init_logits=jax.random.uniform(keys[0], (b, t)),
          transition_logits=jax.random.uniform(keys[1], (b, t, t)))
      if is_categorical:
        v = 100
        kwargs["emission_dist"] = jax.random.uniform(keys[2], (b, t, v))
        kwargs["observations"] = jax.random.randint(keys[3], (b, n), 0, v)
      else:
        d = 100
        kwargs["emission_dist"] = distrax.MultivariateNormalDiag(
            jax.random.uniform(keys[2], (b, t, d)),
            jax.random.uniform(keys[3], (b, t, d), maxval=10),
        )
        kwargs["observations"] = jax.random.uniform(keys[4], (b, n, d))
      dists.append(hmm.HMM(**kwargs))
    return dists

  def create_symmetric_batched_dists(self):
    b, n, t, v = 3, 6, 4, 100
    kwargs = dict(
        init_logits=jnp.zeros((b, t)),
        transition_logits=jnp.zeros((b, t, t)),
        emission_dist=jnp.zeros((b, t, v)),
        observations=jax.random.randint(jax.random.PRNGKey(0), (b, n), 0, v),
    )
    return [hmm.HMM(**kwargs)]

  def create_invalid_shape_distribution(self):
    b, n, t, v = 3, 6, 4, 100
    kwargs = dict(
        init_logits=jnp.zeros((b, t)),
        transition_logits=jnp.zeros((b, t, t)),
        emission_dist=jnp.zeros((b, t+1, v)),
        observations=jax.random.randint(jax.random.PRNGKey(0), (b, n), 0, v),
    )
    return hmm.HMM(**kwargs)

  def analytic_log_count(self, dist) -> jax.Array:
    t = dist.log_potentials.shape[-1]
    return dist.lengths * jnp.log(t)

  def assert_is_symmetric(self, dist, marginals) -> bool:
    self.assert_allclose(marginals[..., 1:-1, :, :], marginals[..., 2:, :, :])

  def assert_batch_of_valid_samples(self, dist, samples):
    self.assertTrue(chain_is_connected(samples),
                    "The chain needs to be connected")

  def assert_valid_marginals(self, dist, marginals):
    self.assert_allclose(marginals.sum((-1, -2)), 1)

  def test_agrees_with_tfp_and_distrax(self):
    n, t, v = 6, 4, 100
    keys = jax.random.split(jax.random.PRNGKey(0), 5)
    kwargs = dict(
        init_logits=jax.random.uniform(keys[0], (t,)),
        transition_logits=jax.random.uniform(keys[1], (t, t)),
        emission_dist=jax.random.uniform(keys[2], (t, v)),
        observations=jax.random.randint(keys[3], (n,), 0, v))
    synjax_hmm_marginals = hmm.HMM(**kwargs).marginals().sum(-2)
    self.assert_allclose(synjax_hmm_marginals, distrax_marginals(**kwargs))
    self.assert_allclose(synjax_hmm_marginals, tfp_marginals(**kwargs))


if __name__ == "__main__":
  absltest.main()
