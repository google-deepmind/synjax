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

"""Utils for efficient chart manipulation."""
from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from synjax._src import constants
from synjax._src.utils import semirings
from synjax._src.utils import special


Array = jax.Array
ArrayLike = jax.typing.ArrayLike
Semiring = semirings.Semiring
roll = special.roll


def from_cky_table(cky_table):
  """Creates a chart from a table of shape (s, n, n, ...).

  Args:
    cky_table: Entries with shape (s, n, n, ...) where first axis is
               dedicated for semiring's usage, and second and third index
               a constituent that spans from (i, j) inclusive on both sides.
  Returns:
    Initialized Chart instance.
  """
  n = cky_table.shape[1]

  f = lambda c, i: roll(c, -i, 1)
  table_left_child = jax.vmap(f, in_axes=(1, 0), out_axes=1
                              )(cky_table, jnp.arange(n))

  f = lambda c, i: roll(c, n-i-1, 1)
  cky_table_transposed = jnp.swapaxes(cky_table, 2, 1)
  table_right_child = jax.vmap(f, in_axes=(1, 0), out_axes=1
                               )(cky_table_transposed, jnp.arange(n))

  return Chart(table_left_child, table_right_child)


class Chart(eqx.Module):
  """Vectorized chart methods described by Rush (2020).

  References:
    Rush, 2020 - Section 6b: https://arxiv.org/pdf/2002.00876.pdf#page=5
  """

  _table_left_child: Array   # Corresponds to Cr in Rush (2020).
  _table_right_child: Array  # Corresponds to Cl in Rush (2020).

  def __init__(self, table_left_child, table_right_child):
    self._table_left_child = table_left_child
    self._table_right_child = table_right_child

  def left(self) -> Array:
    return self._table_left_child

  def right_unmasked(self, d):
    a = roll(self._table_right_child, -d+1, axis=1)
    b = roll(a, d-1, axis=2)
    return b

  def get_entries(self, d):
    return self._table_left_child[:, :, d-1]

  def set_entries(self, d, entries):
    new_table_left_child = self._table_left_child.at[:, :, d-1].set(entries)
    new_table_right_child = self._table_right_child.at[:, :, -d].set(
        roll(entries, d-1, axis=1))
    return Chart(new_table_left_child, new_table_right_child)

  def __repr__(self):
    s = f"Chart[{self._table_left_child.shape}](\n"
    s += f"  Cr:\n{self._table_left_child}\n"
    s += f"  Cl:\n{self._table_right_child}\n"
    s += ")"
    return s

  def left_non_empty(self) -> Array:
    return roll(self.left(), -1, axis=2)

  def right(self, d: ArrayLike, sr: Semiring,
            exclude_word_nodes: bool = False) -> Array:
    return sr.mul(self.mask(d, sr, exclude_word_nodes), self.right_unmasked(d))

  def right_non_empty(self, d: ArrayLike, sr: Semiring) -> Array:
    return sr.mul(self.mask(d, sr, exclude_word_nodes=False),
                  self.right_unmasked_non_empty(d))

  def right_unmasked_non_empty(self, d: ArrayLike) -> Array:
    return roll(self.right_unmasked(d), 1, axis=2)

  def mask(self, d: ArrayLike, sr: Semiring, exclude_word_nodes: bool
           ) -> Array:
    n = self._table_left_child.shape[1]
    vertical = jnp.arange(n) < n-d+1
    if exclude_word_nodes:
      horizontal = (jnp.arange(n) < d-2).at[0].set(False)
    else:
      horizontal = jnp.arange(n) < d-1
    mask = vertical[:, None] & horizontal
    mask = sr.wrap(jnp.where(mask, 0., -constants.INF))
    mask = jnp.expand_dims(mask, range(3, self._table_left_child.ndim))
    return mask

  def pick_length(self, length: Array) -> Array:
    return self._table_left_child[:, 0, length-1]
