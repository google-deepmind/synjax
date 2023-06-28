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

"""Tests for spanning_tree_projective_crf."""

from absl.testing import absltest

import jax
import jax.numpy as jnp
import numpy as np

from synjax._src import constants
from synjax._src import distribution_test
from synjax._src import spanning_tree_projective_crf
from synjax._src.deptree_algorithms import deptree_utils
from synjax._src.utils import special


class SpanningTreeProjectiveTest(distribution_test.DistributionTest):

  def create_random_batched_dists(self, key: jax.random.KeyArray):
    b = 3
    n_words = 5
    log_potentials = jax.random.normal(key, (b, n_words+1, n_words+1))
    dists = [spanning_tree_projective_crf.SpanningTreeProjectiveCRF(
        log_potentials=log_potentials, lengths=None, single_root=single_root)
             for single_root in [True, False]]
    return dists

  def create_invalid_shape_distribution(self):
    return spanning_tree_projective_crf.SpanningTreeProjectiveCRF(
        log_potentials=jnp.zeros((3, 6, 5)), lengths=None, single_root=True)

  def test_Eisner_and_Kuhlmann_argmax_agree(self):
    for dist in self.create_random_batched_dists(jax.random.PRNGKey(0)):
      kuhlmann_argmax, kuhlmann_max = dist.argmax_and_max(algorithm="Kuhlmann")
      eisner_argmax, eisner_max = dist.argmax_and_max(algorithm="Eisner")
      self.assert_allclose(kuhlmann_argmax, eisner_argmax)
      self.assert_allclose(kuhlmann_max, eisner_max)

  def create_symmetric_batched_dists(self):
    b = 3
    n_words = 5
    log_potentials = jnp.zeros((b, n_words+1, n_words+1))
    dists = [spanning_tree_projective_crf.SpanningTreeProjectiveCRF(
        log_potentials=log_potentials, lengths=None, single_root=single_root)
             for single_root in [True, False]]
    return dists

  def analytic_log_count(self, dist) -> jax.Array:
    """Computes the log of the number of the projective trees in the support.

    The number of projective trees in multi-root case is computed using
    Theorem 2 from Yuret (1998, page 29).
    https://arxiv.org/pdf/cmp-lg/9805009.pdf
    For single-root custom adaptation of the multi-root case and will be
    explained in the technical report.

    Args:
      dist: Projective trees distribution object.
    Returns:
      The log of the number of the projective trees.
    """
    def multi_root_projective_log_count(n_words):
      return special.log_comb(3*n_words, n_words) - jnp.log(2*n_words+1)
    if dist.single_root:
      max_n = dist.log_potentials.shape[-1]-1
      first_term = multi_root_projective_log_count(jnp.arange(max_n))
      second_term = multi_root_projective_log_count(
          dist.lengths[..., None]-jnp.arange(max_n)-2)
      mask = jnp.arange(max_n) < dist.lengths[..., None]-1
      to_sum = jnp.where(mask, first_term + second_term, -constants.INF)
      return jax.scipy.special.logsumexp(to_sum, axis=-1)
    else:
      return multi_root_projective_log_count(dist.lengths-1)

  def assert_is_symmetric(self, dist, marginals) -> bool:
    sub_matrix = marginals[..., 1:, 1:]
    self.assert_allclose(sub_matrix, jnp.rot90(sub_matrix, 2, (-1, -2)))
    root_marginals = marginals[..., 0, 1:]
    self.assert_allclose(root_marginals, root_marginals[..., ::-1])

  def assert_batch_of_valid_samples(self, dist, samples):
    trees = np.asarray(jnp.argmax(samples, -2).reshape(-1, samples.shape[-1]))
    for tree in trees:
      self.assertTrue(deptree_utils.is_projective_tree(tree))
    n_words = trees.shape[-1]-1
    self.assert_allclose(jnp.diagonal(samples, axis1=-2, axis2=-1), 0)
    self.assert_allclose(samples[..., 0], 0)
    if dist.single_root:
      self.assert_allclose(jnp.count_nonzero(trees[..., 1:], axis=-1),
                           n_words - 1)

  def assert_valid_marginals(self, dist, marginals):
    self.assert_allclose(
        jnp.sum(jnp.sum(marginals, -2)[..., 1:], -1),
        marginals.shape[-1]-1)
    self.assert_allclose(jnp.diagonal(marginals, axis1=-2, axis2=-1), 0)
    self.assert_allclose(marginals[..., 0], 0)
    if dist.single_root:
      self.assert_allclose(jnp.sum(marginals[:, 0, 1:], axis=-1), 1)


if __name__ == "__main__":
  absltest.main()
