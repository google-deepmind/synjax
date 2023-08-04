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
import numba
import numpy as np

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


# pylint: disable=g-explicit-length-test
@numba.njit
def is_tree(proposal: np.ndarray) -> bool:
  """Checks if proposal forms a valid spanning tree.

  Linear time algorithm from Stanojević and Cohen (2021).

  References:
    Stanojević and Cohen, 2021 - Figure 9: https://aclanthology.org/2021.emnlp-main.823.pdf#page=16

  Args:
    proposal: Numpy array in which element at position i specifies arc
              proposal[i] -> i.
  Returns:
    Boolean for the condition of tree connectedness.
  """  # pylint: disable=line-too-long
  n = proposal.shape[0]
  children = [[1 for _ in range(0)] for _ in range(n)]
  for i in range(1, n):
    children[proposal[i]].append(i)
  is_visited = np.zeros(n, dtype=np.int64)
  stack = [0]
  while len(stack) != 0:
    i = stack.pop()
    is_visited[i] = True
    stack.extend(children[i])
  return is_visited.all()


@numba.njit
def is_projective_tree(proposal):
  """Checks if proposal forms a valid projective spanning tree.

  Linear time algorithm from Stanojević and Cohen (2021).

  References:
    Stanojević and Cohen, 2021 - Figure 10: https://aclanthology.org/2021.emnlp-main.823.pdf#page=17

  Args:
    proposal: Numpy array in which element at position i specifies arc
              proposal[i] -> i.
  Returns:
    Boolean for the condition of projectivity.
  """  # pylint: disable=line-too-long
  n = proposal.shape[0]
  deps_count = np.zeros(n, dtype=np.int64)
  for i in range(1, n):
    deps_count[proposal[i]] += 1
  stack = [0]
  for i in range(1, n):
    stack.append(i)
    while len(stack) > 1:
      right = stack.pop()
      left = stack.pop()
      if proposal[left] == right:
        # Exists left arc.
        stack.append(right)
        deps_count[right] -= 1
      elif proposal[right] == left and deps_count[right] == 0:
        # Exists right arc.
        stack.append(left)
        deps_count[left] -= 1
      else:
        # No attachments possible.
        # Restore stack and move to next word.
        stack.append(left)
        stack.append(right)
        break
  return stack == [0]
