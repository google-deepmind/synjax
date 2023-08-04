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

"""Tests for alignment_monotone_general."""

# pylint: disable=g-importing-member
from absl.testing import absltest
import jax
import jax.numpy as jnp
from synjax._src import distribution_test
from synjax._src.alignment_monotone_general import GeneralMonotoneAlignmentCRF
from synjax._src.utils import special
import typeguard


class GeneralMonotoneAlignmentCrfTest(distribution_test.DistributionTest):

  def analytic_log_count(self, dist: distribution_test.Distribution
                         ) -> jax.Array:
    if dist.log_potentials_vertical is not None and (
        len(dist.log_potentials_horizontal) == 2):
      return special.log_delannoy(
          dist.lengths_rows-1, dist.lengths_cols-1,
          max_input_value=min(*dist.event_shape))
    else:
      raise NotImplementedError

  def create_random_batched_dists(self, key: jax.random.KeyArray):
    b, m, n = 3, 5, 6
    step_0 = step_1 = jax.random.normal(key, (b, m, n))
    dists = [GeneralMonotoneAlignmentCRF((step_0, step_1), step_0),
             GeneralMonotoneAlignmentCRF((step_0, step_1), None)]
    return dists

  def create_symmetric_batched_dists(self):
    b, m = 3, 5
    step_0 = step_1 = jnp.zeros((b, m, m))
    dists = [GeneralMonotoneAlignmentCRF((step_0, step_1), step_0)]
    return dists

  def test_crash_on_invalid_shapes(self):
    b = 3
    m = 5
    step_0 = jnp.zeros((b, m, m))
    step_1 = jnp.zeros((b, m, m-1))

    e = typeguard.TypeCheckError
    self.assertRaises(
        e, lambda: GeneralMonotoneAlignmentCRF((step_0, step_1), step_0))
    self.assertRaises(
        e, lambda: GeneralMonotoneAlignmentCRF((step_0, step_1), None))
    self.assertRaises(
        e, lambda: GeneralMonotoneAlignmentCRF((step_0,), step_1))

  def assert_is_symmetric(self, dist, marginals) -> bool:
    self.assert_allclose(marginals, jnp.swapaxes(marginals, -1, -2))
    self.assert_allclose(marginals,
                         jnp.rot90(jnp.swapaxes(marginals, -1, -2),
                                   k=2, axes=(-1, -2)))
    if dist.log_potentials_vertical is not None:
      self.assert_all(marginals > 0)

  def assert_batch_of_valid_samples(self, dist, samples):
    transitions_count = jnp.sum(samples, (-1, -2))
    self.assert_all(transitions_count >= max(*dist.event_shape))
    self.assert_all(transitions_count <= sum(dist.event_shape)-1)

  def assert_valid_marginals(self, dist, marginals):
    self.assert_allclose(marginals[..., -1, -1], 1)
    self.assert_allclose(marginals[..., 0, 0], 1)
    self.assert_all(marginals.sum(-2) >= 0.98)


if __name__ == "__main__":
  absltest.main()
