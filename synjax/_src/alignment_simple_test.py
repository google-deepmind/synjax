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

"""Tests for alignment_simple."""

from absl.testing import absltest
import jax
import jax.numpy as jnp
from synjax._src import alignment_simple
from synjax._src import distribution_test
from synjax._src.utils import special

AlignmentCRF = alignment_simple.AlignmentCRF


class AlignmentCrfTest(distribution_test.DistributionTest):

  def analytic_log_count(self, dist: distribution_test.Distribution
                         ) -> jax.Array:
    if dist.alignment_type == "non_monotone_one_to_one":
      return jax.scipy.special.gammaln(dist._lengths)
    elif dist.alignment_type == "monotone_many_to_many":
      return special.log_delannoy(
          dist._dist.lengths_rows-1, dist._dist.lengths_cols-1,
          max_input_value=min(*dist.event_shape))
    else:
      raise NotImplementedError

  def test_argmax(self):
    key = jax.random.PRNGKey(0)
    b, n = 3, 5
    dists = self.create_random_batched_dists(key)
    dists.append(AlignmentCRF(jax.random.normal(key, (b, n, n)),
                              alignment_type="non_monotone_one_to_one"))
    for dist in dists:
      assert dist.batch_shape
      best = dist.argmax()
      self.assert_zeros_and_ones(best)
      self.assert_batch_of_valid_samples(dist, best)
      self.assert_valid_marginals(dist, best)

      struct_potential = jnp.exp(dist.unnormalized_log_prob(best))
      self.assertEqual(struct_potential.shape, dist.batch_shape)
      self.assert_all(struct_potential > 0)

  def create_random_batched_dists(self, key: jax.Array):
    b, n, m = 3, 5, 6
    log_potentials = jax.random.normal(key, (b, n, m))
    return [AlignmentCRF(log_potentials, alignment_type=ttype)
            for ttype in ("monotone_one_to_many", "monotone_many_to_many")]

  def create_symmetric_batched_dists(self):
    b, n = 3, 5
    return [AlignmentCRF(jnp.zeros((b, n, n)),
                         alignment_type="monotone_many_to_many")]

  def test_crash_on_invalid_shapes(self):
    b = 3
    m = 5

    # pylint: disable=g-long-lambda
    self.assertRaises(
        ValueError, lambda: AlignmentCRF(
            log_potentials=jnp.zeros((b, m, m-1)),
            alignment_type="non_monotone_one_to_one"))

  def assert_is_symmetric(self, dist, marginals) -> bool:
    self.assert_allclose(marginals, jnp.swapaxes(marginals, -1, -2))
    self.assert_allclose(marginals,
                         jnp.rot90(jnp.swapaxes(marginals, -1, -2),
                                   k=2, axes=(-1, -2)))
    if dist.alignment_type == "monotone_many_to_many":
      self.assert_all(marginals > 0)

  def assert_batch_of_valid_samples(self, dist, samples):
    transitions_count = jnp.sum(samples, (-1, -2))
    self.assert_all(transitions_count >= max(*dist.event_shape))
    self.assert_all(transitions_count <= sum(dist.event_shape)-1)

  def assert_valid_marginals(self, dist, marginals):
    if dist.alignment_type != "non_monotone_one_to_one":
      self.assert_allclose(marginals[..., -1, -1], 1)
      self.assert_allclose(marginals[..., 0, 0], 1)
    self.assert_all(marginals.sum(-2) >= 0.98)


if __name__ == "__main__":
  absltest.main()
