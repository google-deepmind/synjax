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

"""Tests for deptree_utils."""

from absl.testing import absltest
from absl.testing import parameterized

import jax.numpy as jnp
import numpy as np

from synjax._src import constants
from synjax._src.deptree_algorithms import deptree_utils


class DepTreeUtilsTest(parameterized.TestCase):

  def assert_allclose(self, x, y):
    np.testing.assert_allclose(x, y, rtol=constants.TESTING_RELATIVE_TOLERANCE,
                               atol=constants.TESTING_ABSOLUTE_TOLERANCE)

  def test_mask_for_padding(self):
    mask = deptree_utils._mask_for_padding(max_nodes=6, lengths=jnp.array([4]))
    self.assertTrue(jnp.allclose(
        mask.astype(jnp.int32),
        jnp.array(
            [[[0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 1, 1],
              [0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0]]])))

  def test_mask_for_potentials(self):
    mask = deptree_utils._mask_for_potentials(max_nodes=6,
                                              lengths=jnp.array([4]))
    self.assertTrue(jnp.allclose(
        mask.astype(jnp.int32),
        jnp.array(
            [[[0, 1, 1, 1, 0, 0],
              [0, 0, 1, 1, 0, 0],
              [0, 1, 0, 1, 0, 0],
              [0, 1, 1, 0, 0, 0],
              [0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0]]])))

  def test_pad_log_potentials(self):
    log_potentials = jnp.full((6, 6), 2)
    mask = deptree_utils.pad_log_potentials(log_potentials, jnp.array([4]))
    ninf = -constants.INF
    self.assertTrue(jnp.allclose(
        mask.astype(jnp.int32),
        jnp.array(
            [[[ninf, 2, 2, 2, ninf, ninf],
              [ninf, ninf, 2, 2, ninf, ninf],
              [ninf, 2, ninf, 2, ninf, ninf],
              [ninf, 2, 2, ninf, 0, 0],
              [ninf, ninf, ninf, ninf, ninf, ninf],
              [ninf, ninf, ninf, ninf, ninf, ninf]]])))


if __name__ == "__main__":
  absltest.main()
