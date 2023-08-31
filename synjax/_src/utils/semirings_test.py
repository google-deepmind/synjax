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

"""Tests for semirings."""

from absl.testing import absltest
from absl.testing import parameterized

import jax
import jax.numpy as jnp
import numpy as np

from synjax._src import constants
from synjax._src.utils import semirings

# pylint: disable=g-complex-comprehension


class SemiringsTest(parameterized.TestCase):

  def assert_allclose(self, x, y):
    np.testing.assert_allclose(x, y, rtol=constants.TESTING_RELATIVE_TOLERANCE,
                               atol=constants.TESTING_ABSOLUTE_TOLERANCE)

  def test_sampling_semiring(self):
    sr = semirings.SamplingSemiring()
    a = jax.random.uniform(jax.random.PRNGKey(0), (34, 12, 32))
    b = jax.random.uniform(jax.random.PRNGKey(1), (34, 12, 32))

    # Test wrap/unwrap.
    a_wrapped = sr.wrap(a)
    self.assertEqual(a_wrapped.shape, (1,) + a.shape)
    a_unwrapped = sr.unwrap(a_wrapped)
    self.assertEqual(a_unwrapped.shape, a.shape)

    # Test mul.
    self.assert_allclose(sr.mul(a, b), a + b)

    # Test sum.
    logsumexp = jax.scipy.special.logsumexp
    key = jax.random.PRNGKey(0)
    self.assert_allclose(sr.sum(a, -1, key=key), logsumexp(a, -1))
    self.assert_allclose(sr.sum(a, -2, key=key), logsumexp(a, -2))
    self.assert_allclose(sr.sum(a, (-1, -2), key=key), logsumexp(a, (-1, -2)))

    # Test backprop.
    def f(a, key):
      a = sr.wrap(a)
      keys = jax.random.split(key, a.ndim-1)
      for akey in keys:
        a = sr.sum(a, -1, key=akey)
      return sr.unwrap(a)
    sample1 = jax.grad(f)(a, jax.random.PRNGKey(2))
    sample2 = jax.grad(f)(a, jax.random.PRNGKey(3))
    self.assertEqual(sample1.shape, a.shape)
    self.assertEqual(sample2.shape, a.shape)
    self.assertEqual(jnp.sum(sample1), 1)
    self.assertEqual(jnp.sum(sample2), 1)
    self.assertFalse(jnp.all(sample1 == sample2))

  @parameterized.parameters(
      [dict(approximate=approximate, k=k)
       for approximate in [True, False]
       for k in [1, 5]])
  def test_kbest_semiring(self, approximate, k):
    sr = semirings.KBestSemiring(k=k, approximate=approximate)
    a = jax.random.uniform(jax.random.PRNGKey(0), (34, 12, 32))

    def is_ordered(x):
      return x.shape[0] == 1 or jnp.all(x[:-1] >= x[1:])

    # Test wrap/unwrap.
    a_wrapped = sr.wrap(a)
    self.assertEqual(a_wrapped.shape, (k,) + a.shape)
    a_unwrapped = sr.unwrap(a_wrapped)
    self.assertEqual(a_unwrapped.shape, (k,) + a.shape)
    self.assertTrue(is_ordered(a_unwrapped))

    # Test mul.
    x = jnp.log(jnp.arange(k, 0, -1, dtype=jnp.float32))
    kmax = sr.mul(x, x)
    if k >= 3:
      self.assert_allclose(kmax[:3], jnp.array([2*x[0], x[0]+x[1], x[0]+x[1]]))

    # Test sum.
    for axis in [-1, -2]:
      kbest = sr.sum(a_wrapped, axis)
      target_shape = list(a_wrapped.shape)
      target_shape.pop(axis)
      self.assertEqual(kbest.shape, tuple(target_shape))
      self.assertTrue(approximate or is_ordered(a_unwrapped))

    # Test backprop.
    def f(a):
      a = sr.wrap(a)
      while a.ndim > 1:
        a = sr.sum(a, -1)
      return sr.unwrap(a)
    samples = jax.jacrev(f)(a)
    self.assertEqual(samples.shape, (k,)+a.shape)
    def flatten(x):
      return x.reshape(x.shape[0], -1)
    self.assertTrue(jnp.all(jnp.sum(flatten(samples), -1) == 1))
    self.assert_allclose(jnp.sum(flatten(samples * a[None]), -1),
                         jax.lax.top_k(a.reshape(-1), k)[0])

  def test_log_semiring(self):
    sr = semirings.LogSemiring()
    a = jax.random.uniform(jax.random.PRNGKey(0), (34, 12, 32))
    b = jax.random.uniform(jax.random.PRNGKey(1), (34, 12, 32))

    # Test wrap/unwrap.
    a_wrapped = sr.wrap(a)
    self.assertEqual(a_wrapped.shape, (1,) + a.shape)
    a_unwrapped = sr.unwrap(a_wrapped)
    self.assertEqual(a_unwrapped.shape, a.shape)

    # Test mul.
    self.assert_allclose(sr.mul(a, b), a + b)

    # Test sum.
    logsumexp = jax.nn.logsumexp
    self.assert_allclose(sr.sum(a, -1), logsumexp(a, -1))
    self.assert_allclose(sr.sum(a, -2), logsumexp(a, -2))

    # Test grad.
    g = jax.grad(lambda x: sr.sum(x, axis=(-1, -2, -3)))(a)
    self.assertGreater(jnp.count_nonzero(g), 1)

  def test_max_semiring(self):
    sr = semirings.MaxSemiring()
    a = jax.random.uniform(jax.random.PRNGKey(0), (34, 12, 32))
    b = jax.random.uniform(jax.random.PRNGKey(1), (34, 12, 32))

    # Test wrap/unwrap.
    a_wrapped = sr.wrap(a)
    self.assertEqual(a_wrapped.shape, (1,) + a.shape)
    a_unwrapped = sr.unwrap(a_wrapped)
    self.assertEqual(a_unwrapped.shape, a.shape)

    # Test mul.
    self.assert_allclose(sr.mul(a, b), a + b)

    # Test add.
    self.assert_allclose(sr.add(a, b), jnp.maximum(a, b))

    # Test sum.
    self.assert_allclose(sr.sum(a, -1), jnp.max(a, -1))
    self.assert_allclose(sr.sum(a, -2), jnp.max(a, -2))

    # Test grad.
    for smoothing in [None, "softmax", "st-softmax", "sparsemax"]:
      # pylint: disable=cell-var-from-loop
      sr = semirings.MaxSemiring(smoothing=smoothing)
      g = jax.grad(lambda x: sr.sum(x, axis=(-1, -2, -3)))(a)
      if smoothing:
        self.assertGreater(jnp.count_nonzero(g), 1)
      else:
        self.assertEqual(jnp.count_nonzero(g), 1)


if __name__ == "__main__":
  absltest.main()
