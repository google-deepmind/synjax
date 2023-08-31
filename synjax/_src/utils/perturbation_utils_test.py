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

"""Tests for perturbing SynJax distributions."""
# pylint: disable=g-importing-member
# pylint: disable=g-multiple-import
# pylint: disable=g-complex-comprehension
# pylint: disable=g-long-lambda

from absl.testing import absltest
from absl.testing import parameterized

import jax
import jax.numpy as jnp
import numpy as np
import scipy

from synjax._src.utils.perturbation_utils import implicit_mle, sample_sum_of_gamma


def _chisquare_against_gumbel(
    samples, bins=10, start=-3, end=8):
  samples = np.asarray(samples)
  observed_count, bin_edges = np.histogram(samples, bins=bins-2,
                                           range=(start, end))
  left = (samples < bin_edges[0]).sum()
  right = (samples > bin_edges[-1]).sum()
  observed_count = np.concatenate([left[None], observed_count, right[None]])

  cumulative = np.exp(-np.exp(-bin_edges))
  left = cumulative[0]
  right = 1-cumulative[-1]
  expected_count = np.concatenate(
      [left[None], cumulative[1:] - cumulative[:-1], right[None]])
  return scipy.stats.chisquare(observed_count,
                               expected_count*observed_count.sum())


class GeneralTest(parameterized.TestCase):

  def test_sample_sum_of_gamma(self):
    k = 100
    n = 10_000
    key = jax.random.PRNGKey(0)
    samples_sog_s10 = sample_sum_of_gamma(key, shape=(k, n), k=k, s=10).sum(0)
    samples_sog_s50 = sample_sum_of_gamma(key, shape=(k, n), k=k, s=50).sum(0)
    samples_gumbel = jax.random.gumbel(key, shape=(n,))
    samples_gumbel_k_mean = jax.random.gumbel(key, shape=(k, n)).sum(0)
    samples_gumbel_k_sum = jax.random.gumbel(key, shape=(k, n)).sum(0)
    self.assertLess(_chisquare_against_gumbel(samples_sog_s10).statistic, 1000)
    self.assertLess(_chisquare_against_gumbel(samples_sog_s50).statistic, 1000)
    self.assertLess(_chisquare_against_gumbel(samples_gumbel).statistic, 1000)
    self.assertGreater(
        _chisquare_against_gumbel(samples_gumbel_k_mean).statistic, 10_000)
    self.assertGreater(
        _chisquare_against_gumbel(samples_gumbel_k_sum).statistic, 10_000)

  def test_implicit_mle(self):
    n, k = 10, 5
    x = jax.random.uniform(jax.random.PRNGKey(0), (n,))
    top_fn = lambda y: jax.nn.one_hot(jax.lax.top_k(y, k)[1], y.shape[-1]
                                      ).sum(0)
    def loss(x, key):
      sampling_fn = implicit_mle(
          noise_fn=jax.random.gumbel, argmax_fn=top_fn,
          internal_learning_rate=jnp.float32(1.), temperature=jnp.float32(10.))
      sample = sampling_fn(key, x)
      return jnp.sum(sample * jnp.arange(x.shape[-1]))
    for step in range(1000):
      g = jax.grad(loss)(x, jax.random.PRNGKey(step))
      x -= 0.1 * g
    self.assertTrue(jnp.all(top_fn(x)[:k]))


if __name__ == "__main__":
  absltest.main()
