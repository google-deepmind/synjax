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

"""General utils for processing dependency trees."""
# pylint: disable=g-long-lambda
import jax
import jax.numpy as jnp

from synjax._src import constants

Array = jax.Array
INF = constants.INF


def pad_log_potentials(log_potentials: Array, length: Array) -> Array:
  """Pads adjecancy matrix of log_potentials so that it has same log_partition.

  Args:
    log_potentials: Log-potentials of shape (..., n, n) for n nodes.
    length: Number of nodes (including ROOT) of each element in a batch.
  Returns:
    Padded log_potentials so that log-partition function value is preserved.
  """
  max_nodes = log_potentials.shape[-1]
  padding_mask = _mask_for_padding(max_nodes, length)
  potentials_mask = _mask_for_potentials(max_nodes, length)
  # Set padded elems to 0.
  log_potentials = jnp.where(padding_mask, 0, log_potentials)
  # Ignore everything else except padding and selected potentials.
  log_potentials = jnp.where(potentials_mask|padding_mask, log_potentials, -INF)
  return log_potentials


def _mask_for_padding(max_nodes: int, lengths: Array) -> Array:
  horizontal = jnp.arange(max_nodes) >= lengths[..., None]
  vertical = jnp.arange(max_nodes) == lengths[..., None]-1
  return vertical[..., None] & horizontal[..., None, :]


def _mask_for_potentials(max_nodes: int, lengths: Array) -> Array:
  horizontal = jnp.arange(max_nodes) < lengths[..., None]
  vertical = jnp.arange(max_nodes) < lengths[..., None]
  matrix = vertical[..., None] & horizontal[..., None, :]
  return matrix.at[..., 0].set(False) & ~jnp.eye(max_nodes, dtype=bool)
