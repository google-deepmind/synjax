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

"""Implements Maximum Spanning Tree algorithm for directed graphs.

Based on Reweighting+Tarjan algorithm from
Stanojević and Cohen (2021): https://aclanthology.org/2021.emnlp-main.823.pdf
"""
from __future__ import annotations

from typing import Optional, Any

import numba
import numpy as np


NPArray = Any


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


@numba.njit
def _reweighting(log_potentials: NPArray) -> NPArray:
  weights_no_inf = np.where(np.isinf(log_potentials), np.nan, log_potentials)
  log_potentials = log_potentials.copy()
  n = log_potentials.shape[0]-1
  correction = n*(np.nanmax(weights_no_inf)-np.nanmin(weights_no_inf))+1
  log_potentials[0] -= correction
  log_potentials[0, 0] = -np.inf
  return log_potentials


@numba.njit
def _nanargmax(x: numba.float64[:]):
  max_i = 0
  max_val = -np.inf
  for i in range(len(x)):
    if not np.isnan(x[i]) and x[i] >= max_val:
      max_val = x[i]
      max_i = i
  return max_i


@numba.experimental.jitclass([
    ("_target", numba.int64[:]),
    ("_entering_log_potentials", numba.float64[:]),
    ])
class _EdgePriorityQueue:
  """This is a lossy priority queue used for an efficient MST implementation.

     See appendix A in (Stanojević and Cohen, 2021) for more details.
  """

  def __init__(self, node_id: int, edge_weights: np.ndarray):
    self._target = np.full(edge_weights.shape, node_id)
    self._entering_log_potentials = edge_weights
    self._entering_log_potentials[node_id] = np.nan

  def len(self):
    # Counts anything that is not nan.
    return np.count_nonzero(~np.isnan(self._entering_log_potentials))

  def extract_max(self):
    i: int = _nanargmax(self._entering_log_potentials)
    w = self._entering_log_potentials[i]
    self._entering_log_potentials[i] = np.nan
    return i, self._target[i], w

  def meld_inplace(self, other: _EdgePriorityQueue) -> None:
    # pylint: disable=protected-access
    to_replace = (
        self._entering_log_potentials < other._entering_log_potentials)
    self._target[to_replace] = other._target[to_replace]
    self._entering_log_potentials[to_replace] = (
        other._entering_log_potentials[to_replace])
    self._entering_log_potentials[np.isnan(other._entering_log_potentials)
                                 ] = np.nan

  def add_const(self, const: float):
    self._entering_log_potentials[~np.isinf(self._entering_log_potentials)
                                 ] += const


@numba.njit
def _tarjan(log_potentials: np.ndarray) -> np.ndarray:
  """Computes unconstrained Tarjan's (1977) algorithm."""
  null_edge = (-1, -1, -np.inf)
  log_potentials = log_potentials.copy()  # Just in case.
  log_potentials[:, 0] = -np.inf
  n = log_potentials.shape[0]
  max_vertices = n*2-1
  vertices_in = [null_edge for _ in range(max_vertices)]
  vertices_prev = np.zeros(max_vertices, dtype=np.int64)-1
  vertices_children = [[1 for _ in range(0)] for _ in range(max_vertices)]
  vertices_queues = (
      [_EdgePriorityQueue(dep, log_potentials[:, dep]) for dep in range(n)] +
      [None for _ in range(max_vertices-n)])
  vertices_parent = np.arange(max_vertices)
  vertices_highway = np.arange(max_vertices)
  next_free = n

  ######### Compression phase ########
  a = n-1
  while vertices_queues[a].len() != 0:
    u, v, w = vertices_queues[a].extract_max()
    b = vertices_highway[u]  # find
    assert a != b, "there should be no self-loop in this implementation"
    vertices_in[a] = (u, v, w)
    vertices_prev[a] = b
    if vertices_in[u] == null_edge:
      # path extended
      a = b
    else:
      # new cycle formed, collapse
      c = next_free
      next_free += 1

      i = a
      while True:
        i = vertices_highway[i]  # find
        vertices_children[c].append(i)
        i = vertices_prev[i]
        if vertices_highway[i] == a:  # find
          break

      for i in vertices_children[c]:
        vertices_parent[i] = c
        # union by collapsing
        vertices_highway[vertices_highway == vertices_highway[i]] = c
        vertices_queues[i].add_const(-vertices_in[i][2])
        if vertices_queues[c] is None:
          vertices_queues[c] = vertices_queues[i]
        else:
          vertices_queues[c].meld_inplace(vertices_queues[i])
      a = c

  ######### Expansion phase ########
  # Next line is just supervertices = [] but is written as a weird comprehension
  # so that Numba infers the correct type List[int].
  supervertices = [1 for _ in range(0)]
  _dismantle(0, vertices_parent, vertices_children, supervertices)
  # pylint: disable=g-explicit-length-test
  while len(supervertices) > 0:
    c = supervertices.pop()
    u, v, w = vertices_in[c]
    vertices_in[v] = (u, v, w)
    _dismantle(v, vertices_parent, vertices_children, supervertices)
  output = np.zeros(n, dtype=np.int64)
  for u in range(1, n):
    output[u] = vertices_in[u][0]
  return output


@numba.njit
def _dismantle(u: int,
               vertices_parent: numba.int64[:],
               vertices_children: numba.typeof([[1]]),
               supervertices: numba.typeof([1])):
  """Dismantles a cycle that was constructed in Tarjan phase 1."""
  while vertices_parent[u] != u:
    for v in vertices_children[vertices_parent[u]]:
      if v == u:
        continue
      vertices_parent[v] = v
      # pylint: disable=g-explicit-length-test
      if len(vertices_children[v]) > 0:
        supervertices.append(v)
    u = vertices_parent[u]


@numba.njit
def _arcmax(log_potentials: NPArray) -> NPArray:
  n = log_potentials.shape[-1]-1
  proposal = np.zeros(n+1, dtype=np.int64)
  for i in range(1, n+1):
    proposal[i] = np.argmax(log_potentials[:, i])
  return proposal


@numba.njit
def _parse(log_potentials: NPArray, single_root_edge: bool) -> NPArray:
  """Applies ArcMax and Reweighting tricks before calling Tarjan's algorithm."""
  proposal = _arcmax(log_potentials)
  root_count = np.count_nonzero(proposal[1:] == 0)
  if is_tree(proposal) and (not single_root_edge or root_count == 1):
    result = proposal
  else:
    if single_root_edge:
      log_potentials = _reweighting(log_potentials)
    result = _tarjan(log_potentials)
  return result


@numba.guvectorize("(n,n),(),()->(n)", nopython=True)
def _vectorized_mst(log_potentials, length, single_root_edge, res):
  res[:length] = _parse(log_potentials[:length, :length], single_root_edge)
  res[length:] = length


def vectorized_mst(log_potentials: NPArray, lengths: Optional[NPArray],
                   single_root_edge: bool) -> NPArray:
  """Numpy implementation of MST that supports batch dimension."""
  if lengths is None:
    lengths = np.full(log_potentials.shape[:-2], log_potentials.shape[-1])
  single_root_edge_expanded = np.full(
      log_potentials.shape[:-2], single_root_edge, dtype=np.int64)
  assert log_potentials.shape[:-2] == lengths.shape
  out = np.full(log_potentials.shape[:-1], -2, dtype=np.int64)
  log_potentials = log_potentials.astype(np.float64)
  lengths = lengths.astype(np.int64)
  with np.errstate(invalid="ignore"):
    _vectorized_mst(log_potentials, lengths, single_root_edge_expanded, out)
  return out
