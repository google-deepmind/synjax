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

"""Tests for synjax._src.utils.chart_struct."""

from absl.testing import absltest
from absl.testing import parameterized

import jax.numpy as jnp

from synjax._src.utils import chart_struct
from synjax._src.utils import semirings

# pylint: disable=g-complex-comprehension


def create_numbered_chart(n):
  # Creates a chart with 4 axes (semiring, n, n, 2) where last axis
  # contains a pair signifying the span that that node covers.
  chart_table = jnp.zeros((1, n, n, 2))
  chart_table = chart_table.at[0, :, :, 0].add(jnp.arange(n).reshape(n, 1))
  chart_table = chart_table.at[0, :, :, 1].add(jnp.arange(n).reshape(1, n))
  return chart_struct.from_cky_table(chart_table)


class ChartStructTest(parameterized.TestCase):

  def assert_all(self, x, *, msg=""):
    self.assertTrue(jnp.all(x), msg=msg)

  @parameterized.parameters([
      dict(d=d, n=n)
      for n in [5] for d in range(1, n+1)
  ])
  def test_get_entries(self, d, n):
    chart = create_numbered_chart(n)
    diagonal = chart.get_entries(d)
    self.assertEqual(diagonal.shape, (1, n, 2))
    self.assert_all(diagonal[0, :d, 0] == jnp.arange(d))
    spans_count = n-d+1
    valid_starts = diagonal[0, :spans_count, 0]
    valid_ends = diagonal[0, :spans_count, 1]
    invalid_starts = diagonal[0, spans_count:, 0]
    invalid_ends = diagonal[0, spans_count:, 1]
    self.assertTrue(jnp.all(valid_ends-valid_starts+1 == d))
    self.assertTrue(jnp.all(invalid_ends-invalid_starts < 0))

  @parameterized.parameters([
      dict(d=d, n=n)
      for n in [5] for d in range(1, n+1)
  ])
  def test_mask(self, d, n):
    chart = create_numbered_chart(n)
    sr = semirings.LogSemiring()
    # Testing mask without excluding word nodes.
    mask = chart.mask(d, sr, exclude_word_nodes=False) > sr.zero()
    split_points = jnp.sum(mask[0, :, :, 0], -1)
    self.assertTrue(jnp.all(split_points[:-d+1] == d-1))
    self.assertTrue(jnp.all(split_points[-d+1:] == 0))
    spans_count = jnp.sum(mask[0, :, :, 0], -2)
    self.assertTrue(jnp.all(spans_count[:d-1] == n-d+1))
    self.assertTrue(jnp.all(spans_count[d-1:] == 0))
    # Testing mask with excluding word nodes.
    mask = chart.mask(d, sr, exclude_word_nodes=True) > sr.zero()
    split_points = jnp.sum(mask[0, :, :, 0], -1)
    if d >= 4:
      self.assert_all(split_points[:-d+1] == d-3)
    else:
      self.assert_all(split_points[:-d+1] == 0)
    self.assert_all(split_points[-d+1:] == 0)
    spans_count = jnp.sum(mask[0, :, :, 0], -2)
    if d >= 4:
      self.assert_all(spans_count[1:d-2] == n-d+1)
    else:
      self.assert_all(spans_count[:-d+1] == 0)
    self.assert_all(spans_count[d-1:] == 0)

    # Testing that left and masked right match.
    left_cut = chart.left()[0, :-d+1, :d-1]
    sr = semirings.LogSemiring()
    right_cut = chart.right(d, sr)[0, :-d+1, :d-1]
    # End of left matched beginning of right for all nodes.
    self.assert_all(left_cut[:, :, 1]+1 == right_cut[:, :, 0])
    # Difference between end of right and beginning of left is parent span size.
    self.assertTrue(jnp.all(right_cut[:, :, 1] - left_cut[:, :, 0] + 1 == d))


if __name__ == "__main__":
  absltest.main()
