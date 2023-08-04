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

# pylint: disable=g-multiple-import
# pylint: disable=g-importing-member

from typing import Union

import equinox as eqx
import jax
import jax.numpy as jnp

from jaxtyping import Int, Float, Array
from synjax._src import constants
from synjax._src.typing import typed
from synjax._src.utils import semirings
from synjax._src.utils import special


Semiring = semirings.Semiring
roll = special.roll
SpanSize = Union[int, Int[Array, ""]]


@typed
def from_cky_table(cky_table: Float[Array, "s n n ..."]) -> "Chart":
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


@typed
class Chart(eqx.Module):
  """Vectorized chart methods described by Rush (2020).

  References:
    Rush, 2020 - Section 6b: https://arxiv.org/pdf/2002.00876.pdf#page=5
  """

  _table_left_child: Float[Array, "s n n ..."]   # Cr in Rush (2020).
  _table_right_child: Float[Array, "s n n ..."]  # Cl in Rush (2020).

  @typed
  def __init__(self,
               table_left_child: Float[Array, "s n n ..."],
               table_right_child: Float[Array, "s n n ..."]):
    self._table_left_child = table_left_child
    self._table_right_child = table_right_child

  @typed
  def left(self) -> Float[Array, "s n n ..."]:
    return self._table_left_child

  @typed
  def right_unmasked(self, d: SpanSize) -> Float[Array, "s n n ..."]:
    a = roll(self._table_right_child, -d+1, axis=1)
    b = roll(a, d-1, axis=2)
    return b

  @typed
  def get_entries(self, d: SpanSize) -> Float[Array, "s n ..."]:
    return self._table_left_child[:, :, d-1]

  @typed
  def set_entries(self, d: SpanSize, entries) -> "Chart":
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

  @typed
  def left_non_empty(self) -> Float[Array, "s n n ..."]:
    return roll(self.left(), -1, axis=2)

  @typed
  def right(self, d: SpanSize, sr: Semiring,
            exclude_word_nodes: bool = False) -> Float[Array, "s n n ..."]:
    return sr.mul(self.mask(d, sr, exclude_word_nodes), self.right_unmasked(d))

  @typed
  def right_non_empty(self, d: SpanSize, sr: Semiring
                      ) -> Float[Array, "s n n ..."]:
    return sr.mul(self.mask(d, sr, exclude_word_nodes=False),
                  self.right_unmasked_non_empty(d))

  @typed
  def right_unmasked_non_empty(self, d: SpanSize) -> Float[Array, "s n n ..."]:
    return roll(self.right_unmasked(d), 1, axis=2)

  @typed
  def mask(self, d: SpanSize, sr: Semiring, exclude_word_nodes: bool
           ) -> Float[Array, "s n n ..."]:
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

  @typed
  def pick_length(self, length: Int[Array, ""]) -> Float[Array, "s ..."]:
    return self._table_left_child[:, 0, length-1]
