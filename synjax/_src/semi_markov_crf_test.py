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

"""Tests for semi_markov_crf."""
from absl.testing import absltest
import jax
import jax.numpy as jnp
from synjax._src import distribution_test
from synjax._src import linear_chain_crf
from synjax._src import semi_markov_crf


def chain_is_connected(samples) -> bool:
  scores = jax.lax.associative_scan(jnp.matmul, samples, axis=-3)
  scores = scores[..., -1, :, :].sum((-1, -2))
  return jnp.all(scores == 1)


class SemiMarkovCRFTest(distribution_test.DistributionTest):

  def create_random_batched_dists(self, key):
    b, n, m, t = 3, 6, 3, 4
    log_potentials = jnp.log(jax.random.uniform(key, (b, n, m, t, t)))
    return [semi_markov_crf.SemiMarkovCRF(log_potentials)]

  def create_symmetric_batched_dists(self):
    b, n, m, t = 3, 6, 3, 4
    log_potentials = jnp.zeros((b, n, m, t, t))
    return [semi_markov_crf.SemiMarkovCRF(log_potentials)]

  def create_invalid_shape_distribution(self):
    b, n, m, t = 3, 6, 3, 4
    log_potentials = jnp.zeros((b, n, m, t+1, t))
    return semi_markov_crf.SemiMarkovCRF(log_potentials)

  def assert_is_symmetric(self, dist, marginals) -> bool:
    # There is no simple symmetric constraint to test against.
    pass

  def assert_batch_of_valid_samples(self, dist, samples):
    labels = semi_markov_crf.SemiMarkovCRF.convert_sample_to_element_labels(
        samples)
    n = labels.shape[-2]
    self.assert_allclose(jnp.cumsum(labels.sum(-1), -1), jnp.arange(1, n+1))

  def assert_valid_marginals(self, dist, marginals):
    active_edges_weight = marginals.sum((-1, -2, -3))
    self.assert_allclose(active_edges_weight[..., -1], 1)

  def test_semi_markov_simple_agrees_with_linear_chain_crf(self):
    b, n, t = 3, 6, 4
    key = jax.random.PRNGKey(0)
    log_potentials = jnp.log(jax.random.uniform(key, (b, n, t, t)))
    dist1 = linear_chain_crf.LinearChainCRF(log_potentials)
    dist2 = semi_markov_crf.SemiMarkovCRF(
        jnp.concatenate([log_potentials[..., None, :, :],
                         jnp.full((b, n, 3, t, t), -1e5)], axis=-3))
    m1 = dist1.marginals()
    m2 = dist2.marginals().max(-3)
    self.assert_allclose(m1, m2)


if __name__ == "__main__":
  absltest.main()
