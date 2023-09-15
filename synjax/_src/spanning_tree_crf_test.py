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

"""Tests for spanning_tree_crf.

Here we test only undirected cases since directed cases are tested in the
spanning_tree_non_projective_crf_test and spanning_tree_projective_crf_test.
"""
# pylint: disable=g-importing-member

from absl.testing import absltest

import jax
import jax.numpy as jnp

from synjax._src import distribution_test
from synjax._src import spanning_tree_crf
from synjax._src.deptree_algorithms.deptree_padding import undirected_tree_mask


SpanningTreeCRF = spanning_tree_crf.SpanningTreeCRF


class SpanningTreeCrfTest(distribution_test.DistributionTest):

  def _create_dist(self, f):
    b, n = 2, 6
    return [spanning_tree_crf.SpanningTreeCRF(
        log_potentials=f((b, n, n)), lengths=jnp.array([n-1, n-2]),
        directed=False, single_root_edge=True, projective=projective)
            for projective in [True, False]]

  def create_random_batched_dists(self, key: jax.Array):
    return self._create_dist(lambda shape: jax.random.uniform(key, shape))

  def create_symmetric_batched_dists(self):
    return self._create_dist(jnp.zeros)

  def create_invalid_shape_distribution(self):
    return spanning_tree_crf.SpanningTreeCRF(
        log_potentials=jnp.zeros((2, 6, 6-1)), directed=False,
        single_root_edge=True, projective=True)

  def test_log_count(self):
    # Skips testing for log-count since there is no simple unified formula for
    # all supported sub-types of distributions.
    pass

  def assert_is_symmetric(self, dist, marginals) -> bool:
    del dist
    self.assert_allclose(marginals, jnp.swapaxes(marginals, -1, -2))

  def assert_batch_of_valid_samples(self, dist, samples):
    _, n = dist.event_shape
    l = dist.lengths[..., None, None]
    mask = (jnp.arange(n) < l) & (jnp.arange(n)[:, None] < l)
    self.assert_allclose(jnp.where(mask, 0, samples), 0)
    if not dist.directed:
      self.assert_is_symmetric(dist, samples)
    if dist.single_root_edge:
      self.assert_allclose(jnp.sum(samples[..., 0, :], -1), 1)

  def assert_valid_marginals(self, dist, marginals):
    if not dist.directed:
      self.assert_is_symmetric(dist, marginals)
    n = marginals.shape[-1]
    self.assert_allclose(marginals * ~undirected_tree_mask(n, dist.lengths), 0)
    row = jnp.arange(n) < dist.lengths[..., None, None]
    col = jnp.arange(n)[:, None] < dist.lengths[..., None, None]
    self.assert_allclose(jnp.where(row, 0, marginals), 0)
    self.assert_allclose(jnp.where(col, 0, marginals), 0)

  def test_top_k(self):
    for dist in self.create_random_batched_dists(jax.random.PRNGKey(0)):
      self.check_top_k_single_dist(dist, check_prefix_condition=dist.projective)

  def test_sample_without_replacement(self):
    args = [jax.random.PRNGKey(0), 3]
    for dist in self.create_random_batched_dists(jax.random.PRNGKey(0)):
      if dist.projective:
        self.assertRaises(NotImplementedError,
                          dist.sample_without_replacement, *args)
      else:
        dist.sample_without_replacement(*args)  # Should not crash.


if __name__ == "__main__":
  absltest.main()
