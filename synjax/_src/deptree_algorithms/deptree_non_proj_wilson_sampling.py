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

"""Algorithms for random walk sampling of dependency trees from Stanojević 2022.

Stanojević, 2022: https://aclanthology.org/2022.emnlp-main.110.pdf
"""
import numba
import numpy as np

from synjax._src import constants

EPS = constants.EPS
MTT_LOG_EPS = constants.MTT_LOG_EPS


@numba.njit
def _construct_laplacian_hat(log_potentials: np.ndarray, single_root: bool
                             ) -> np.ndarray:
  """Computes a graph laplacian matrix.

  Args:
    log_potentials: Weight matrix with log-potential entries.
    single_root: Whether to use a single-root constraint
  Returns:
    Laplacian matrix.
  """
  potentials = np.exp(np.logaddexp(log_potentials, MTT_LOG_EPS))
  potentials[..., 0] = 0  # Removing root-entering edges.
  potentials *= (1-np.eye(potentials.shape[-1]))  # Removing self-edges
  def laplacian(x):  # Standard complete Laplacian matrix.
    return np.expand_dims(np.sum(x, axis=-2), axis=-2) * np.eye(x.shape[-1]) - x
  def cut(x):  # Removes 0th row and 0th column.
    return x[..., 1:, 1:]
  if single_root:
    l = laplacian(cut(potentials))  # (..., n-1, n-1)
    l[..., 0, :] = potentials[..., 0, 1:]
  else:
    l = cut(laplacian(potentials))  # (..., n-1, n-1)
  return l


@numba.njit
def _marginals_with_given_laplacian_invt(
    log_potentials: np.ndarray, laplacian_invt: np.ndarray,
    single_root: bool) -> np.ndarray:
  """Computes marginals in cases where the inverse of the Laplacian is provided.

  Based on the presentation in Koo et al 2007
  https://aclanthology.org/D07-1015.pdf

  Args:
    log_potentials: Weight matrix with log-potential entries.
    laplacian_invt: Inverse-transpose of the Laplacian-hat matrix.
    single_root: Whether to use a single-root constraint.
  Returns:
    Matrix of marginals.
  """
  potentials = np.exp(np.logaddexp(log_potentials, MTT_LOG_EPS))
  marginals = np.zeros(potentials.shape)

  x = np.diag(laplacian_invt).copy()  # Extract diagonal of laplacian inverse.
  if single_root:
    x[0] = 0
  x_matrix = x.reshape(1, -1)  # (1, n)

  y_matrix = laplacian_invt.copy()
  if single_root:
    y_matrix[0] = 0

  marginals[1:, 1:] = potentials[1:, 1:] * (x_matrix - y_matrix)
  if single_root:
    marginals[0, 1:] = potentials[0, 1:] * laplacian_invt[0]
  else:
    marginals[0, 1:] = potentials[0, 1:] * np.diag(laplacian_invt)
  marginals = np.where(np.isnan(marginals) | (marginals < 0), 0, marginals)
  return marginals


@numba.njit
def _marginals(log_potentials: np.ndarray, single_root):
  laplacian = _construct_laplacian_hat(log_potentials, single_root)
  return _marginals_with_given_laplacian_invt(
      log_potentials, np.linalg.inv(laplacian).T, single_root)


@numba.njit
def _sample_wilson_multi_root(log_potentials: np.ndarray):
  """Sampling rooted spanning trees from directed graphs using Wilson algorithm.

  Args:
    log_potentials: Log-potentials from which to get a sample.
  Returns:
    Single sample that may contain multiple root edges.
  """
  n = log_potentials.shape[0]-1
  t = np.zeros(n+1, dtype=np.int64)
  visited = np.zeros(n+1, dtype=np.int64)
  visited[0] = 1
  for i in range(1, n+1):
    u: int = i
    loop_count = 0
    max_loop_count = n * 100  # Needed to prevent infinite loops in some graphs.
    while not visited[u] and loop_count < max_loop_count:
      loop_count += 0
      noise = np.random.gumbel(0, 1, n+1)
      v = np.argmax(log_potentials[:, u] + noise)
      t[u] = v
      u = v
    u = i
    while not visited[u]:
      visited[u] = 1
      u = t[u]
  return t


@numba.njit
def _sample_generalized_wilson(log_potentials: np.ndarray, single_root: bool
                               ) -> np.ndarray:
  """Returns only a single sample spanning tree."""
  if single_root:
    log_potentials = log_potentials.copy()
    marginals = _marginals(log_potentials, single_root)
    root_log_marginals = np.log(np.maximum(marginals, 0.0001))[0]
    root_node = np.argmax(root_log_marginals +
                          np.random.gumbel(0, 1, log_potentials.shape[-1]))
    log_potentials[0] = -np.inf
    log_potentials[0, root_node] = 0
  return _sample_wilson_multi_root(log_potentials)


@numba.guvectorize("(n,n),(),()->(n)", nopython=True)
def _vectorized_sample_wilson(log_potentials, length, single_root, res):
  res[:length] = _sample_generalized_wilson(
      log_potentials[:length, :length], single_root)
  res[length:] = length


def vectorized_sample_wilson(log_potentials, lengths, single_root):
  """Vectorized version of wilson algorithm that returns a single sample."""
  single_root_extended = np.full(log_potentials.shape[:-2], single_root,
                                 dtype=np.int64)
  if lengths is None:
    lengths = np.full(log_potentials.shape[:-2], log_potentials.shape[-1])
  out = np.zeros(log_potentials.shape[:-1], dtype=np.int64)
  log_potentials = log_potentials.astype(np.float64)
  lengths = lengths.astype(np.int64)
  _vectorized_sample_wilson(log_potentials, lengths, single_root_extended, out)
  return out
