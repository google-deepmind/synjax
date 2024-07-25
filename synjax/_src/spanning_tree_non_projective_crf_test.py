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

"""Tests for spanning_tree_non_projective_crf."""

# pylint: disable=g-importing-member

from absl.testing import absltest
from absl.testing import parameterized

import jax
import jax.numpy as jnp
import numpy as np

from synjax._src import distribution_test
from synjax._src import spanning_tree_non_projective_crf
from synjax._src.deptree_algorithms import deptree_non_proj_argmax
from synjax._src.deptree_algorithms import deptree_non_proj_wilson_sampling
from synjax._src.deptree_algorithms.deptree_padding import directed_tree_mask

SpanningTreeNonProjectiveCRF = (
    spanning_tree_non_projective_crf.SpanningTreeNonProjectiveCRF)


class SpanningTreeNonProjectiveCRFTest(
    distribution_test.DistributionTest):

  def create_random_batched_dists(self, key: jax.Array):
    b = 3
    n = 6
    log_potentials = jax.random.normal(key, (b, n, n))
    dists = []
    for single_root_edge in [True, False]:
      dists.append(SpanningTreeNonProjectiveCRF(
          log_potentials=log_potentials, lengths=jnp.array([n-1, n-2, n]),
          single_root_edge=single_root_edge))
    return dists

  def create_symmetric_batched_dists(self):
    b = 3
    n_words = 5
    log_potentials = jnp.zeros((b, n_words+1, n_words+1))
    dists = []
    for single_root_edge in [True, False]:
      dists.append(SpanningTreeNonProjectiveCRF(
          log_potentials=log_potentials, lengths=None,
          single_root_edge=single_root_edge))
    return dists

  def analytic_log_count(self, dist) -> jax.Array:
    """Computes the log of the number of the spanning trees in the support.

    Computes the log of the number of spanning trees in the
    support of the distribution by using Cayle's formula with the
    modification for single-root trees from
    Stanojević (2022) https://aclanthology.org/2021.emnlp-main.823.pdf

    Args:
      dist: Non-Projective distribution object.
    Returns:
      The log of the number of the spanning trees.
    """
    if dist.single_root_edge:
      return (dist.lengths-2)*jnp.log(dist.lengths-1)
    else:
      return (dist.lengths-2)*jnp.log(dist.lengths)

  def assert_is_symmetric(self, dist, marginals) -> bool:
    sub_matrix = marginals[..., 1:, 1:]
    self.assert_allclose(sub_matrix, jnp.swapaxes(sub_matrix, -1, -2))
    self.assert_allclose(marginals[..., 0, 1:-1], marginals[..., 0, 2:])

  def test_sampling_can_be_jitted(self):
    key = jax.random.PRNGKey(0)
    for algorithm in ["wilson", "colbourn"]:
      f = jax.jit(lambda x: x.sample(key, algorithm=algorithm))  # pylint: disable=cell-var-from-loop
      for dist in self.create_random_batched_dists(jax.random.PRNGKey(0)):
        jax.block_until_ready(f(dist))
        jax.block_until_ready(f(dist))

  def assert_batch_of_valid_samples(self, dist, samples):
    self.assert_zeros_and_ones(samples)
    for tree in jnp.argmax(samples, -2).reshape(-1, samples.shape[-1]):
      self.assertTrue(deptree_non_proj_argmax.is_tree(np.asarray(tree)))
    self.assert_allclose(jnp.diagonal(samples, axis1=-2, axis2=-1), 0)
    self.assert_allclose(samples[..., 0], 0)
    if dist.single_root_edge:
      self.assert_allclose(samples[..., 0, :].sum(-1), 1)

  def assert_valid_marginals(self, dist, marginals):
    self.assert_allclose(
        jnp.sum(jnp.sum(marginals, -2)[..., 1:], -1),
        dist.lengths-1)
    n = marginals.shape[-1]
    self.assert_allclose(marginals * ~directed_tree_mask(n, dist.lengths), 0)
    self.assert_allclose(jnp.diagonal(marginals, axis1=-2, axis2=-1), 0)
    self.assert_allclose(marginals[..., 0], 0)
    if dist.single_root_edge:
      self.assert_allclose(jnp.sum(marginals[..., 0, 1:], axis=-1), 1)

  def test_top_k(self):
    """This method overrides the test of the superclass.

    Currently Non-Projective trees top_k implementation supports only
    approximate decoding so if we use the superclass implementation it would
    fail on subtests 'Top k-1 is a prefix of top k' and
    'Two exact algorithms (top_k and sort) give the same result'.
    The rest of this test is the same.
    """
    for dist in self.create_random_batched_dists(jax.random.PRNGKey(0)):
      self.check_top_k_single_dist(dist, check_prefix_condition=False)

  def test_sample_without_replacement(self):
    k = 4
    for dist in self.create_random_batched_dists(jax.random.PRNGKey(0)):
      k_samples, k_logprobs_direct, _ = dist.sample_without_replacement(
          jax.random.PRNGKey(0), k=k)
      k_logprobs = dist.log_prob(k_samples)
      self.assert_allclose(k_logprobs, k_logprobs_direct)
      self.assert_all(k_logprobs <= 0)
      self.assert_all(k_logprobs > -1e5)

      # All trees are valid.
      for i in range(k):
        self.assert_valid_marginals(dist, k_samples[i])
        self.assert_batch_of_valid_samples(dist, k_samples[i])

      # All structs are different from each other.
      def single_instance_has_duplicates(instance_k_samples):
        k_flattened = instance_k_samples.reshape(
            instance_k_samples.shape[0], -1)
        comparison_matrix = jnp.all(k_flattened[:, None] == k_flattened, -1)
        comparison_matrix = jnp.where(jnp.eye(comparison_matrix.shape[-1]),
                                      False, comparison_matrix)
        return jnp.any(comparison_matrix)
      self.assert_all(
          ~jax.vmap(single_instance_has_duplicates, in_axes=1)(k_samples),
          msg="All top-k structures should be unique.")

  def test_wilson_and_colbourn_do_not_crash(self):
    for dist in self.create_random_batched_dists(jax.random.PRNGKey(0)):
      with self.subTest("testing Wilson"):
        sample = dist.sample(jax.random.PRNGKey(0), algorithm="wilson")
        self.assert_valid_marginals(dist, sample)
      with self.subTest("testing Colbourn"):
        sample = dist.sample(jax.random.PRNGKey(0), algorithm="colbourn")
        self.assert_valid_marginals(dist, sample)

  @parameterized.parameters([dict(single_root_edge=True),
                             dict(single_root_edge=False)])
  def test_marginals_with_given_laplacian_invt(self, single_root_edge: bool):
    n = 5
    potentials = jax.random.uniform(jax.random.PRNGKey(0), (n, n))
    potentials = potentials.at[:, 0].set(0)   # Nothing enters root node.
    potentials = potentials * (1-jnp.eye(n))  # No self-loops.
    log_potentials = jnp.log(potentials)

    laplacian = spanning_tree_non_projective_crf._construct_laplacian_hat(
        log_potentials, single_root_edge=single_root_edge)
    laplacian_invt = jnp.linalg.inv(laplacian).T
    marginals_a = (
        spanning_tree_non_projective_crf._marginals_with_given_laplacian_invt(
            log_potentials, laplacian_invt, single_root_edge=single_root_edge))
    marginals_b = jnp.asarray(
        deptree_non_proj_wilson_sampling._marginals_with_given_laplacian_invt(
            np.asarray(log_potentials), np.asarray(laplacian_invt),
            single_root_edge=single_root_edge))
    marginals_c = SpanningTreeNonProjectiveCRF(
        log_potentials=log_potentials,
        single_root_edge=single_root_edge).marginals()
    # pylint: disable=g-long-lambda
    marginals_d = jax.grad(            # This should in principle be the same
        lambda x: jnp.linalg.slogdet(  # as marginals_a but without API fluff.
            spanning_tree_non_projective_crf._construct_laplacian_hat(
                x, single_root_edge=single_root_edge))[1])(log_potentials)
    self.assert_allclose(marginals_a, marginals_b)
    self.assert_allclose(marginals_a, marginals_c)
    self.assert_allclose(marginals_a, marginals_d)


if __name__ == "__main__":
  absltest.main()
