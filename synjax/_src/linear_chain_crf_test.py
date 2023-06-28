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

"""Tests for linear_chain_crf."""
from absl.testing import absltest
import jax
import jax.numpy as jnp
from synjax._src import distribution_test
from synjax._src import linear_chain_crf


def chain_is_connected(samples) -> bool:
  scores = jax.lax.associative_scan(jnp.matmul, samples, axis=-3)
  scores = scores[..., -1, :, :].sum((-1, -2))
  return jnp.all(scores == 1)


class LinearChainTest(distribution_test.DistributionTest):

  def create_random_batched_dists(self, key):
    b, n, t = 3, 6, 4
    log_potentials = jnp.log(jax.random.uniform(key, (b, n, t, t)))
    return [linear_chain_crf.LinearChainCRF(log_potentials)]

  def create_symmetric_batched_dists(self):
    b, n, t = 3, 6, 4
    log_potentials = jnp.zeros((b, n, t, t))
    return [linear_chain_crf.LinearChainCRF(log_potentials)]

  def create_invalid_shape_distribution(self):
    b, n, t = 3, 6, 4
    log_potentials = jnp.zeros((b, n, t, t-1))
    return linear_chain_crf.LinearChainCRF(log_potentials)

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

  def test_algorithm_sequential_same_as_parallel(self):
    dist = self.create_random_batched_dists(jax.random.PRNGKey(0))[0]
    m1 = dist.marginals(forward_algorithm="sequential")
    m2 = dist.marginals(forward_algorithm="parallel")
    self.assert_allclose(m1, m2)


if __name__ == "__main__":
  absltest.main()
