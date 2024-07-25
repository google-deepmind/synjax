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

"""Tests for constituency_tree_crf."""
# pylint: disable=g-importing-member

from absl.testing import absltest
import jax
import jax.numpy as jnp

from synjax._src import distribution_test
from synjax._src.constituency_tree_crf import TreeCRF
from synjax._src.utils import special


def is_symmetric(x, axis1, axis2):
  x1 = jnp.rot90(x, axes=(axis1, axis2))
  x2 = jnp.swapaxes(x1, axis1, axis2)
  return jnp.allclose(x2, x1)


class TreeCrfTest(distribution_test.DistributionTest):

  def create_random_batched_dists(self, key: jax.Array):
    b, n, t = 3, 5, 2
    log_potentials = jnp.log(jax.random.uniform(key, (b, n, n, t)))
    lengths = jnp.array(list(range(n-b+1, n+1)))
    return [TreeCRF(log_potentials, lengths=lengths)]

  def create_symmetric_batched_dists(self):
    b, n, t = 1, 5, 4
    log_potentials = jnp.zeros((b, n, n, t))
    return [TreeCRF(log_potentials, lengths=None)]

  def analytic_log_count(self, dist) -> jax.Array:
    # Note: terminal labels are included as part of the combinatorial structure.
    log_tree_count = special.log_catalan(dist.lengths-1)
    t = dist.log_potentials.shape[-1]
    log_nt_labeling_count = (2*dist.lengths-1) * jnp.log(t)
    return log_tree_count + log_nt_labeling_count

  def assert_is_symmetric(self, dist, marginals) -> bool:
    self.assertTrue(is_symmetric(marginals, 1, 2))

  def assert_batch_of_valid_samples(self, dist, samples):
    self.assert_allclose(jnp.sum(samples, axis=(-1, -2, -3)), dist.lengths*2-1)

  def assert_valid_marginals(self, dist, marginals):
    lengths = jnp.broadcast_to(dist.lengths, marginals.shape[:-3])
    root_probs = jnp.take_along_axis(marginals[..., 0, :, :].sum(-1),
                                     lengths[..., None]-1, axis=-1)
    self.assert_allclose(root_probs, 1)
    terminals = jnp.diagonal(marginals.sum(-1), axis1=-1, axis2=-2)  # b, n
    self.assert_zeros_and_ones(terminals)
    self.assert_allclose(terminals.sum(-1), dist.lengths)


if __name__ == "__main__":
  absltest.main()
