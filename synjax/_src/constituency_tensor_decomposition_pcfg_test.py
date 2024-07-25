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

"""Tests for constituency_tensor_decomposition_pcfg."""

from absl.testing import absltest

import jax
import jax.numpy as jnp

from synjax._src import constituency_tensor_decomposition_pcfg as td
from synjax._src import distribution_test
from synjax._src.utils import special


def is_symmetric(x, axis1, axis2):
  x1 = jnp.rot90(x, axes=(axis1, axis2))
  x2 = jnp.swapaxes(x1, axis1, axis2)
  return jnp.allclose(x2, x1)


class TensorDecompositionPCFGTest(distribution_test.DistributionTest):

  def test_argmax(self):
    for dist in self.create_random_batched_dists(jax.random.PRNGKey(0)):
      self.assertRaises(NotImplementedError, dist.argmax)

  def test_sampling(self):
    # Sampling is not supported.
    for dist in self.create_random_batched_dists(jax.random.PRNGKey(0)):
      self.assertRaises(NotImplementedError, dist.sample, jax.random.PRNGKey(0))

  def test_top_k(self):
    # top_k is not supported.
    for dist in self.create_random_batched_dists(jax.random.PRNGKey(0)):
      self.assertRaises(NotImplementedError, dist.top_k, 2)

  def analytic_log_count(self, dist) -> jax.Array:
    log_rank_combs = (dist.lengths-1)*jnp.log(dist.size_rank)
    log_nt_combs = (dist.lengths-1) * jnp.log(
        dist.size_nonterminals)
    log_pt_combs = dist.lengths * jnp.log(dist.size_preterminals)
    return (special.log_catalan(dist.lengths-1) + log_rank_combs
            + log_nt_combs + log_pt_combs)

  def create_random_batched_dists(self, key: jax.Array):
    f = jax.random.uniform
    keys = jax.random.split(key, 6)
    b, n, voc, nt, pt, r = 2, 6, 10, 2, 3, 2
    log_potentials = dict(
        root=f(keys[0], (b, nt)),
        nt_to_rank=f(keys[1], (b, nt, r)),
        rank_to_left_nt=f(keys[2], (b, r, nt+pt)),
        rank_to_right_nt=f(keys[3], (b, r, nt+pt)),
        emission=f(keys[4], (b, pt, voc)),
    )
    word_ids = jax.random.randint(keys[5], (b, n), 0, voc)
    return [td.TensorDecompositionPCFG(**log_potentials, word_ids=word_ids)]

  def create_symmetric_batched_dists(self):
    f = jnp.zeros
    b, n, voc, nt, pt, r = 2, 6, 10, 2, 3, 2
    log_potentials = dict(
        root=f((b, nt)),
        nt_to_rank=f((b, nt, r)),
        rank_to_left_nt=f((b, r, nt+pt)),
        rank_to_right_nt=f((b, r, nt+pt)),
        emission=f((b, pt, voc)),
    )
    word_ids = jax.random.randint(jax.random.PRNGKey(0), (b, n), 0, voc)
    return [td.TensorDecompositionPCFG(**log_potentials, word_ids=word_ids)]

  def assert_is_symmetric(self, dist, marginals) -> bool:
    del dist
    chart_marginals, preterm_marginals = marginals
    self.assertTrue(is_symmetric(chart_marginals, 1, 2))
    self.assert_allclose(preterm_marginals, preterm_marginals[..., ::-1, :])

  def assert_batch_of_valid_samples(self, dist, samples):
    chart, preterms = samples
    self.assert_allclose(chart.sum((-1, -2, -3)), dist.lengths-1)
    self.assert_allclose(preterms.sum((-1, -2)), dist.lengths)

  def assert_valid_marginals(self, dist, marginals):
    chart_marginals, preterminal_marginals = marginals
    span_marginals = chart_marginals.sum(-1)

    lengths = jnp.broadcast_to(dist.lengths, span_marginals.shape[:-2])
    root_probs = jnp.take_along_axis(span_marginals[..., 0, :],
                                     lengths[..., None]-1, axis=-1)
    self.assert_allclose(root_probs, 1)

    self.assert_zeros_and_ones(preterminal_marginals.sum(-1))
    self.assert_allclose(preterminal_marginals.sum((-1, -2)), dist.lengths)

  def test_argmax_can_be_jitted(self):
    pass

  def test_sampling_can_be_jitted(self):
    pass

  def test_differentiable_sample(self):
    super().test_differentiable_sample(methods=("Perturb-and-Marginals",))

  def test_mbr(self):
    for dist in self.create_random_batched_dists(jax.random.PRNGKey(0)):
      assert dist.batch_shape
      best = dist.mbr(marginalize_labels=False)
      self.assert_zeros_and_ones(best)
      self.assert_batch_of_valid_samples(dist, best)
      self.assert_valid_marginals(dist, best)

  def test_entropy_cross_entropy(self):
    for dist in self.create_random_batched_dists(jax.random.PRNGKey(0)):
      assert dist.batch_shape
      self.assert_allclose(dist.entropy(), dist.cross_entropy(dist))
      self.assert_allclose(dist.kl_divergence(dist), 0)


if __name__ == "__main__":
  absltest.main()
