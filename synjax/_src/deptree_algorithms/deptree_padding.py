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


def pad_directed_log_potentials(log_potentials: Array, length: Array) -> Array:
  """Pads adjacency matrix of log_potentials so that it has same log_partition.

  Sets padded arcs to have weight 0, valid arcs to have their original weight
  and invalid arcs to have weight -INF.

  Args:
    log_potentials: Log-potentials of shape (..., n, n) for n nodes.
    length: Number of nodes (including ROOT) of each element in a batch.
  Returns:
    Padded log_potentials so that log-partition function value is preserved.
  """
  max_nodes = log_potentials.shape[-1]
  padding_mask = _mask_for_padding(max_nodes, length)
  potentials_mask = directed_tree_mask(max_nodes, length)
  return jnp.where(potentials_mask, log_potentials,
                   jnp.where(padding_mask, 0, -INF))


def _mask_for_padding(max_nodes: int, lengths: Array) -> Array:
  horizontal = jnp.arange(max_nodes) >= lengths[..., None]
  vertical = jnp.arange(max_nodes) == lengths[..., None]-1
  return vertical[..., None] & horizontal[..., None, :]


def directed_tree_mask(max_nodes: int, lengths: Array) -> Array:
  return undirected_tree_mask(max_nodes, lengths).at[..., 0].set(False)


def undirected_tree_mask(max_nodes: int, lengths: Array) -> Array:
  horizontal = jnp.arange(max_nodes) < lengths[..., None]
  vertical = jnp.arange(max_nodes) < lengths[..., None]
  matrix = vertical[..., None] & horizontal[..., None, :]
  return matrix & ~jnp.eye(max_nodes, dtype=bool)  # Delete diagonal.
