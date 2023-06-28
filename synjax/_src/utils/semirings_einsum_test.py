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

"""Tests for semirings_einsum."""

from absl.testing import absltest
from absl.testing import parameterized

import jax
import jax.numpy as jnp
import numpy as np

from synjax._src import constants
from synjax._src.utils import semirings_dot_general
from synjax._src.utils import semirings_einsum


def einsum_log(*operands, **kwargs):
  sum_fn = jax.nn.logsumexp
  mul_op = jnp.add
  dot_general = semirings_dot_general.build_dot_general(sum_fn, mul_op)
  return semirings_einsum.einsum_generalized(
      *operands, **kwargs,
      sum_fn=sum_fn, mul_op=mul_op, dot_general=dot_general)


def einsum_real(*operands, **kwargs):
  sum_fn = jnp.sum
  mul_op = jnp.multiply
  dot_general = semirings_dot_general.build_dot_general(sum_fn, mul_op)
  return semirings_einsum.einsum_generalized(
      *operands, **kwargs,
      sum_fn=sum_fn, mul_op=mul_op, dot_general=dot_general)


def einsum_tropical(*operands, **kwargs):
  sum_fn = jnp.max
  mul_op = jnp.add
  dot_general = semirings_dot_general.build_dot_general(sum_fn, mul_op)
  return semirings_einsum.einsum_generalized(
      *operands, **kwargs,
      sum_fn=sum_fn, mul_op=mul_op, dot_general=dot_general)


class SemiringsEinsumTest(parameterized.TestCase):

  def assert_allclose(self, x, y):
    np.testing.assert_allclose(x, y, rtol=constants.TESTING_RELATIVE_TOLERANCE,
                               atol=constants.TESTING_ABSOLUTE_TOLERANCE)

  def test_einsum_generalized(self):
    bs = (11, 3)
    x = jax.random.uniform(jax.random.PRNGKey(0), (2, *bs, 5, 4))
    y = jax.random.uniform(jax.random.PRNGKey(0), (2, 4, 7, *bs))
    expression = "s...ab,sbc...->s...ac"

    self.assert_allclose(
        einsum_real(expression, x, y),
        jnp.einsum(expression, x, y))

  def test_einsum_tropical_semiring(self):
    x = jax.random.uniform(jax.random.PRNGKey(0), (2, 4, 3, 5, 4))
    self.assert_allclose(
        jnp.max(x, (0, -2, -1)),
        einsum_tropical("a...bc->...", x))

  def test_einsum_log_semiring(self):
    bs = (11, 3)
    x = jax.random.uniform(jax.random.PRNGKey(0), (2, *bs, 5, 4))
    y = jax.random.uniform(jax.random.PRNGKey(0), (2, 4, 7, *bs))
    expression = "s...ab,sbc...->s...ac"

    self.assert_allclose(
        jnp.log(jnp.einsum(expression, jnp.exp(x), jnp.exp(y))),
        einsum_log(expression, x, y))


if __name__ == "__main__":
  absltest.main()
