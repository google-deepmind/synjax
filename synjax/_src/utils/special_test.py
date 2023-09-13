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

"""Tests for synjax._src.utils.general."""

# pylint: disable=g-complex-comprehension
# pylint: disable=g-importing-member

from functools import partial
import math

from absl.testing import absltest
from absl.testing import parameterized

import jax
import jax.numpy as jnp
import jaxopt
import numpy as np

from synjax._src import constants
from synjax._src.utils import special


class GeneralTest(parameterized.TestCase):

  def assert_allclose(self, x, y, rtol=constants.TESTING_RELATIVE_TOLERANCE,
                      atol=constants.TESTING_ABSOLUTE_TOLERANCE):
    np.testing.assert_allclose(x, y, rtol=rtol, atol=atol)

  def assert_all(self, x, *, msg=""):
    self.assertTrue(all(map(jnp.all, jax.tree_util.tree_flatten(x)[0])),
                    msg=msg)

  def test_log_comb(self):
    self.assert_allclose(jnp.log(math.comb(32, 10)), special.log_comb(32, 10))

  def test_log_catalan(self):
    self.assert_allclose(special.log_catalan(jnp.arange(8)),
                         jnp.log(jnp.array([1, 1, 2, 5, 14, 42, 132, 429])))

  def test_log_delannoy(self):
    self.assert_allclose(special.log_delannoy(jnp.arange(7), jnp.arange(7),
                                              max_input_value=10),
                         jnp.log(jnp.array([1, 3, 13, 63, 321, 1683, 8989])))
    self.assert_allclose(special.log_delannoy(jnp.arange(7), jnp.arange(1, 8),
                                              max_input_value=10),
                         jnp.log(jnp.array([1, 5, 25, 129, 681, 3653, 19825])))

  def test_tpu_roll(self):
    x = jnp.arange(7*7*7).reshape(7, 7, 7)
    axis = -2
    shift = 2
    self.assert_allclose(special._tpu_roll(x, shift=shift, axis=axis),
                         jnp.roll(x, shift=shift, axis=axis))

  def test_tpu_take(self):
    n = 4
    x = jnp.arange(n*n*n).reshape(n, n, n)
    axis = -2
    indices = jnp.arange(n-1, 1, -1)
    self.assert_allclose(special._tpu_take(x, indices, axis=axis),
                         jnp.take(x, indices, axis=axis))

  def test_inv(self):
    def summarize_fn(x):
      # This is a meaningless asymmetric function that produces a scalar.
      return jnp.sum(x*jax.random.normal(jax.random.PRNGKey(0), x.shape))
    grad = lambda ff, **kw: jax.grad(lambda x: summarize_fn(ff(x, **kw)))
    n = 20
    for method in ["solve", "qr"]:
      if method == "solve":
        assert_allclose = self.assert_allclose
      else:
        # Allow lower precision closeness because of no gold standard.
        assert_allclose = partial(self.assert_allclose, rtol=1e-3, atol=1e-3)

      assert_not_finite = lambda x: self.assertTrue(jnp.all(~jnp.isfinite(x)))

      with self.subTest(method):
        matrix = jax.random.uniform(jax.random.PRNGKey(0), (n, n))
        assert_allclose(special.inv(matrix, inv_method=method),
                        jnp.linalg.inv(matrix))
        assert_allclose(grad(special.inv, inv_method=method)(matrix),
                        grad(jnp.linalg.inv)(matrix))

        matrix = jnp.zeros((n, n))  # Non-invertable matrix.

        assert_not_finite(special.inv(matrix, inv_method=method,
                                      test_invertability=False))
        assert_allclose(special.inv(matrix, inv_method=method,
                                    test_invertability=True), 0.)

        assert_not_finite(grad(special.inv, inv_method=method,
                               test_invertability=False)(matrix))
        assert_allclose(grad(special.inv, inv_method=method,
                             test_invertability=True)(matrix), 0.)

        self.assert_all(~jnp.isfinite(special.inv(matrix, inv_method=method,
                                                  test_invertability=False)))
        self.assert_all(jnp.isnan(grad(special.inv, inv_method=method,
                                       test_invertability=False)(matrix)))

  def test_safe_slogdet(self):
    grad = lambda ff: jax.grad(lambda x: ff(x)[1])
    n = 20
    matrix = jax.random.uniform(jax.random.PRNGKey(0), (n, n))
    self.assert_allclose(special.safe_slogdet(matrix),
                         jnp.linalg.slogdet(matrix))
    self.assert_allclose(grad(special.safe_slogdet)(matrix),
                         grad(jnp.linalg.slogdet)(matrix))
    matrix = jnp.zeros((n, n))
    self.assert_all(jnp.isfinite(special.safe_slogdet(matrix)[1]))
    self.assert_allclose(grad(special.safe_slogdet)(matrix), 0)
    matrix = jnp.ones((n, n))
    self.assert_all(jnp.isfinite(special.safe_slogdet(matrix)[1]))
    self.assert_allclose(grad(special.safe_slogdet)(matrix), 0)

  def test_safe_log(self):
    self.assert_all(jnp.exp(special.safe_log(0)) == 0)
    self.assert_all(special.safe_log(-0.001) < -1e4)
    self.assert_all(special.safe_log(0) < -1e4)
    self.assert_all(special.safe_log(1) == 0)
    self.assert_all(jax.grad(special.safe_log)(1.) == 1)
    self.assert_all(jax.grad(special.safe_log)(0.) > 1e4)

  def test_sparsemax(self):
    x = jax.random.normal(jax.random.PRNGKey(0), (10, 5))
    jaxopt_sparsemax = jax.vmap(jaxopt.projection.projection_simplex)
    self.assert_allclose(special.sparsemax(x), jaxopt_sparsemax(x))
    g1 = jax.grad(lambda y: jnp.sum(special.sparsemax(y)[:, 0]))(x)
    g2 = jax.grad(lambda y: jnp.sum(jaxopt_sparsemax(y)[:, 0]))(x)
    self.assert_allclose(g1, g2)
    self.assertGreater(jnp.count_nonzero(g1), 2)


if __name__ == "__main__":
  absltest.main()
