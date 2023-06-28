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

"""Tests for synjax._src.utils.semirings_dot_general."""

from absl.testing import absltest
from absl.testing import parameterized

import jax
import jax.numpy as jnp
import numpy as np

from synjax._src import constants
from synjax._src.utils import semirings_dot_general

# pylint: disable=g-complex-comprehension


class SemiringsTest(parameterized.TestCase):

  def assert_allclose(self, x, y):
    np.testing.assert_allclose(x, y, rtol=constants.TESTING_RELATIVE_TOLERANCE,
                               atol=constants.TESTING_ABSOLUTE_TOLERANCE)

  def test_real_dot_general(self):
    lhs = lhs = jax.random.uniform(jax.random.PRNGKey(0), (4, 3))
    rhs = jax.random.uniform(jax.random.PRNGKey(2), (3, 5))
    dimension_numbers = (([1], [0]), ([], []))
    x = jax.lax.dot_general(lhs, rhs, dimension_numbers)
    real_dot_general = semirings_dot_general.build_dot_general(jnp.sum,
                                                               jnp.multiply)
    y = real_dot_general(lhs, rhs, dimension_numbers)
    self.assert_allclose(x, y)

  def test_log_dot_general(self):
    lhs = lhs = jax.random.uniform(jax.random.PRNGKey(0), (4, 3))
    rhs = jax.random.uniform(jax.random.PRNGKey(2), (3, 5))
    dimension_numbers = (([1], [0]), ([], []))
    dot_general_log = semirings_dot_general.build_dot_general(
        jax.nn.logsumexp, jnp.add)
    x = dot_general_log(lhs, rhs, dimension_numbers)
    y = jnp.log(jax.lax.dot_general(jnp.exp(lhs), jnp.exp(rhs),
                                    dimension_numbers))
    self.assert_allclose(x, y)


if __name__ == "__main__":
  absltest.main()
