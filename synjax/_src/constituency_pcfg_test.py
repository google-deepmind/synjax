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

"""Tests for PCFG."""

from absl.testing import absltest

import jax
import jax.numpy as jnp

from synjax._src import constituency_pcfg
from synjax._src import distribution_test
from synjax._src.utils import special


PCFG = constituency_pcfg.PCFG


def is_symmetric(x, axis1, axis2):
  x1 = jnp.rot90(x, axes=(axis1, axis2))
  x2 = jnp.swapaxes(x1, axis1, axis2)
  return jnp.allclose(x2, x1)


class PcfgTest(distribution_test.DistributionTest):

  def _create_dist(self, f):
    b, n, nt, pt, voc = 2, 4, 2, 3, 40
    log_potentials = dict(
        root=f((b, nt)),
        rule=f((b, nt, nt+pt, nt+pt)),
        emission=f((b, pt, voc))
    )
    word_ids = jnp.tile(jnp.arange(n), (b, 1))
    return [PCFG(**log_potentials, word_ids=word_ids)]

  def create_random_batched_dists(self, key: jax.random.KeyArray):
    return self._create_dist(
        lambda shape: jnp.log(jax.random.uniform(key, shape)))

  def create_invalid_shape_distribution(self):
    b, n, nt, pt, voc = 2, 4, 2, 3, 40
    f = jnp.zeros
    return PCFG(root=f((b, nt)),
                rule=f((b, nt, nt+pt, nt)),
                emission=f((b, pt, voc)),
                word_ids=jnp.tile(jnp.arange(n), (b, 1)))

  def create_symmetric_batched_dists(self):
    return self._create_dist(jnp.zeros)

  def analytic_log_count(self, dist) -> jax.Array:
    log_tree_count = special.log_catalan(dist.lengths-1)
    log_nt_labeling_count = (dist.lengths-1) * jnp.log(dist.size_nonterminals)
    log_t_labeling_count = dist.lengths * jnp.log(dist.size_preterminals)
    return log_tree_count + log_nt_labeling_count + log_t_labeling_count

  def assert_is_symmetric(self, dist, marginals) -> bool:
    chart_marginals, preterm_marginals = marginals
    self.assertTrue(is_symmetric(chart_marginals, 1, 2))
    self.assert_allclose(preterm_marginals, preterm_marginals[..., ::-1, :])

  def assert_batch_of_valid_samples(self, dist, samples):
    chart_marginals, preter_marginals = samples
    self.assert_allclose(jnp.sum(chart_marginals, axis=(-1, -2, -3)),
                         dist.lengths-1)
    self.assert_allclose(preter_marginals.sum((-1, -2)), dist.lengths)

  def assert_valid_marginals(self, dist, marginals):
    chart_marginals, preterm_marginals = marginals
    for i in range(dist.batch_shape[0]):
      n = dist.lengths[i]
      root_prob = chart_marginals[i, ..., 0, n-1, :].sum(-1)
      self.assert_allclose(root_prob, 1)
    self.assert_zeros_and_ones(preterm_marginals.sum(-1))
    self.assert_allclose(preterm_marginals.sum((-1, -2)), dist.lengths)


if __name__ == "__main__":
  absltest.main()
