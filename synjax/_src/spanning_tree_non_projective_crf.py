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

"""Distribution representing non-projective dependency trees."""
from __future__ import annotations
# pylint: disable=g-long-lambda
# pylint: disable=g-multiple-import, g-importing-member
import dataclasses
from functools import partial
from typing import Literal, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int32
import numpy as np
from synjax._src.config import get_config
from synjax._src.constants import EPS, MTT_LOG_EPS
from synjax._src.deptree_algorithms.deptree_padding import pad_log_potentials, directed_tree_mask
from synjax._src.distribution import Distribution
from synjax._src.typing import Shape, Key, typed
from synjax._src.utils import autoregressive_decoding
from synjax._src.utils import special


SamplingAlgorithmName = Literal["colbourn", "wilson"]


@typed
def _optionally_shift_log_potentials(
    log_potentials: Float[Array, "*batch n n"], single_root_edge: bool
    ) -> Tuple[Float[Array, "*batch n n"], Float[Array, "*batch"]]:
  """Makes log-potentials numerically more stable.

  Modifies log_potentials to be more numerically stable without having
  any impact on the tree distribution. Inspired by see Section D.2 from
  Paulus et al (2020). This implementation is more stable than
  Paulus et al because max normalization it is applied column-wise and in that
  way guarantees that the maximum score of a tree is not bigger than 0, but it
  breaks the symmetry if there was any.

  References:
    Paulus et al 2020 - Section D2: https://arxiv.org/pdf/2006.08063.pdf#page=26

  Args:
    log_potentials: Log-potentials of the graph.
    single_root_edge: Whether to renormalize the root outgoing edges which is
                      valid only if single-root constraint is used.

  Returns:
    New log potentials with correction for log-partition.
  """
  cfg = get_config()
  if cfg.mtt_shift_log_potentials:
    c_matrix = jnp.max(log_potentials, axis=-2, keepdims=True)
    correction = jnp.sum(c_matrix[..., 0, 1:], -1)
    if single_root_edge:
      c_root = jnp.max(
          log_potentials*jax.nn.one_hot(0, log_potentials.shape[-1])[:, None],
          axis=-1, keepdims=True)
      c_matrix += c_root
      correction += c_root[..., 0, -1]
    log_potentials -= jax.lax.stop_gradient(c_matrix.at[..., :, 0].set(0))
  else:
    correction = jnp.zeros(log_potentials.shape[:-2])
  return log_potentials, jax.lax.stop_gradient(correction)


@typed
def _custom_slog_det(
    x: Float[Array, "*batch n n"]
    ) -> Tuple[Float[Array, "*batch"], Float[Array, "*batch"]]:
  cfg = get_config()
  return special.safe_slogdet(x, logdet_method=cfg.mtt_logdet_method,
                              inv_method=cfg.mtt_inv_method,
                              matmul_precision=cfg.mtt_inv_matmul_precision,
                              test_invertability=False)


class SpanningTreeNonProjectiveCRF(Distribution):
  """Distribution representing non-projective dependency trees."""

  single_root_edge: bool = eqx.static_field()
  lengths: Int32[Array, "*batch"]

  @typed
  def __init__(self,
               log_potentials: Float[Array, "*batch n n"],
               *,
               single_root_edge: bool,
               lengths: Optional[Int32[Array, "*batch"]] = None,
               **kwargs):
    super().__init__(log_potentials=log_potentials, **kwargs)
    self.single_root_edge = single_root_edge
    if lengths is None:
      batch_shape = log_potentials.shape[:-2]
      lengths = jnp.full(batch_shape, log_potentials.shape[-1])
    self.lengths = lengths

  @property
  def event_shape(self) -> Shape:
    return self.log_potentials.shape[-2:]

  @property
  def max_nodes(self) -> int:
    # Maximal number of nodes including ROOT node at position 0.
    return self.log_potentials.shape[-1]

  @property
  def _typical_number_of_parts_per_event(self) -> Int32[Array, "*batch"]:
    return self.lengths-1

  @typed
  def argmax(self, **ignored_args) -> Float[Array, "*batch n n"]:
    return mst_numpy_callback(self._padded_log_potentials, self.lengths,
                              self.single_root_edge).astype(jnp.float32)

  @typed
  def argmax_and_max(self, **kwargs) -> Tuple[Float[Array, "*batch n n"],
                                              Float[Array, "*batch"]]:
    best = self.argmax(**kwargs)
    score = self.unnormalized_log_prob(best)
    return best, score

  @typed
  def top_k(self, k: int, approximate: bool = True
            ) -> Tuple[Float[Array, "k *batch n n"], Float[Array, "k *batch"]]:
    """This is an approximate top-k by using beam search over marginals.

    Args:
      k: The number of trees to return.
      approximate: Use the approximate top-k algorithm.
    Returns:
      A tuple of trees (represented as adjecency matrices) and their logprobs.
    """
    if k <= 0:
      raise ValueError("k must be a strictly positive integer")
    elif k == 1:
      # This is a shortcut optimization for a special case.
      best, score = self.argmax_and_max()
      return best[None], score[None]
    else:
      if not approximate:
        raise NotImplementedError("Non-Projective trees distribution supports"
                                  "only 'approximate' top_k_algorithm.")

      beam_state, _ = special.vmap_ndim(
          lambda lp: autoregressive_decoding.beam_search(
              init_state=State.initial(lp,
                                       single_root_edge=self.single_root_edge),
              max_length=self.max_nodes-1,
              k=k), self.batch_ndim)(self._padded_log_potentials)
      trees = beam_state.sample
      trees = jnp.moveaxis(trees, self.batch_ndim, 0)
      matrices = _to_adjacency_matrix(trees, self.lengths)
      return matrices, self.unnormalized_log_prob(matrices)

  @property
  def _padded_log_potentials(self) -> Float[Array, "*batch n n"]:
    return pad_log_potentials(self.log_potentials, self.lengths)

  @typed
  def log_partition(self) -> Float[Array, "*batch"]:
    log_potentials, correction = _optionally_shift_log_potentials(
        self._padded_log_potentials, self.single_root_edge)
    laplacian_hat = _construct_laplacian_hat(log_potentials,
                                             self.single_root_edge)
    return correction + _custom_slog_det(laplacian_hat)[1]

  def marginals_for_template_variables(self, **kwargs):
    # Default marginals_for_template_vars with an addition of removing padding.
    marginal = super().marginals_for_template_variables(**kwargs).log_potentials
    mask = directed_tree_mask(self.event_shape[-1], self.lengths)
    return dataclasses.replace(self, log_potentials=mask * marginal)

  @typed
  def sample_without_replacement(
      self, key: Key, k: int) -> Tuple[Float[Array, "k *batch n n"],
                                       Float[Array, "k *batch"],
                                       Float[Array, "k *batch"]]:
    """Sampling without replacement from Stanojević (2022).

    References:
      Stanojević, 2022: https://aclanthology.org/2022.emnlp-main.110.pdf

    Args:
      key: Sampling key.
      k: Number of swor samples.
    Returns:
      Tuple of (samples, logprobs, gumbel perturbed logprobs)
    """
    beam_state, logprobs, gumbels = special.vmap_ndim(
        lambda rng, lp: autoregressive_decoding.stochastic_beam_search(
            key=rng,
            init_state=State.initial(lp,
                                     single_root_edge=self.single_root_edge),
            max_length=self.max_nodes-1,
            k=k), self.batch_ndim
        )(special.split_key_for_shape(key, self.batch_shape),
          self._padded_log_potentials)
    sampled_trees = beam_state.sample
    move = lambda x: jnp.moveaxis(x, self.batch_ndim, 0)
    sampled_matrices = _to_adjacency_matrix(move(sampled_trees), self.lengths)
    return sampled_matrices, move(logprobs), move(gumbels)

  @typed
  def _single_sample(self, key: Key,
                     algorithm: SamplingAlgorithmName = "colbourn"
                     ) -> Float[Array, "*batch n n"]:
    if algorithm == "colbourn":
      final_states: State
      final_states, _, _ = special.vmap_ndim(
          lambda rng, lp: autoregressive_decoding.single_ancestral_sample(
              init_state=State.initial(lp,
                                       single_root_edge=self.single_root_edge),
              key=rng, max_length=self.max_nodes-1, unroll=1),
          self.batch_ndim
          )(special.split_key_for_shape(key, self.batch_shape),
            self._padded_log_potentials)
      sampled_trees = final_states.sample
      sampled_matrices = _to_adjacency_matrix(sampled_trees, self.lengths)
    elif algorithm == "wilson":
      sampled_matrices = sample_wilson_numpy_callback(
          self._padded_log_potentials, self.lengths, self.single_root_edge
          ).astype(jnp.float32)
    else:
      raise NotImplementedError(
          f"sampling_algorithm {algorithm:r} not supported")
    return sampled_matrices


@typed
def _to_adjacency_matrix(
    tree: Int32[Array, "*batch n"], lengths: Int32[Array, "*#batch"]
    ) -> Float[Array, "*batch n n"]:
  tree = jnp.where(jnp.arange(tree.shape[-1]) < lengths[..., None], tree, -1)
  return jax.nn.one_hot(tree, tree.shape[-1], dtype=jnp.float32, axis=-2
                        ).at[..., 0].set(0)


@typed
class State(autoregressive_decoding.State):
  """Implements a state of a Colbourn sampler for spanning trees.

  Original algorithm presented in Colbourn et al (1996) for spanning trees.
  Zmigrod et al (2021) adapt the algorithm to single-root dependency trees.
  Here we use presentation from Stanojević (2022) that is easier to adapt for
  more compelex use cases such as sampling without replacement.

  References:
    Stanojević, 2022: https://aclanthology.org/2022.emnlp-main.110.pdf
    Zmigrod et al, 2021: https://aclanthology.org/2021.emnlp-main.824v2.pdf
    Colbourn et al, 1996: https://www.sciencedirect.com/science/article/pii/S0196677496900140
  """  # pylint: disable=line-too-long

  potentials: Float[Array, "n n"]
  laplacian: Float[Array, "n-1 n-1"]
  laplacian_invt: Float[Array, "n-1 n-1"]
  j: Int32[Array, ""]
  sample: Int32[Array, "n"]
  single_root_edge: bool = eqx.static_field()

  @typed
  def logprobs(self) -> Float[Array, "n"]:
    marginals = _marginals_with_given_laplacian_invt(
        jnp.log(self.potentials), self.laplacian_invt,
        single_root_edge=self.single_root_edge)
    return special.safe_log(marginals[:, self.j])

  @typed
  def apply_transition(self, a: Int32[Array, ""]) -> State:
    potentials, laplacian, laplacian_invt = self._constrain_graph(a)
    sample = self.sample.at[self.j].set(a)
    state = State(potentials=potentials, laplacian=laplacian,
                  laplacian_invt=laplacian_invt, j=self.j + 1, sample=sample,
                  single_root_edge=self.single_root_edge)
    return state

  @typed
  def _constrain_graph(self, i: Int32[Array, ""]
                       ) -> Tuple[Float[Array, "n n"], Float[Array, "n-1 n-1"],
                                  Float[Array, "n-1 n-1"]]:
    potentials_old, laplacian_old, laplacian_invt_old, j = (
        self.potentials, self.laplacian, self.laplacian_invt, self.j)
    constrained_incoming = jax.nn.one_hot(i, potentials_old.shape[-1])
    potentials = potentials_old.at[..., j].set(constrained_incoming)

    laplacian = _construct_laplacian_hat(jnp.log(potentials), self.single_root_edge)

    uj = laplacian[:, j - 1] - laplacian_old[:, j - 1]
    bj = laplacian_invt_old[:, j - 1]

    den = 1 + uj.T @ bj

    # Application of Sherman-Morrison formula.
    update = jnp.outer(bj, uj.T @ laplacian_invt_old)/jnp.where(den, den, EPS)
    laplacian_invt = laplacian_invt_old - update

    return potentials, laplacian, laplacian_invt

  @staticmethod
  @typed
  def initial(log_potentials: Float[Array, "n n"], single_root_edge: bool) -> State:
    empty_sample = jnp.empty(log_potentials.shape[-1], dtype=jnp.int32)
    laplacian = _construct_laplacian_hat(log_potentials,
                                         single_root_edge=single_root_edge)
    laplacian_invt = jnp.linalg.inv(laplacian).T
    return State(potentials=jnp.exp(log_potentials), laplacian=laplacian,
                 laplacian_invt=laplacian_invt, j=jnp.int32(1),
                 single_root_edge=single_root_edge, sample=empty_sample)


@typed
def _construct_laplacian_hat(
    log_potentials: Float[Array, "*batch n n"], single_root_edge: bool
    ) -> Float[Array, "*batch n-1 n-1"]:
  """Computes a graph Laplacian-hat matrix as in Koo et al (2007).

  This is not a Laplacian matrix, but Laplacian-hat matrix. It is constructed by
  applying the right modification to a regular Laplacian matrix so that a
  determinant of Laplacian-hat gives partition function of all spanning trees.

  References:
    Koo et al, 2007: https://aclanthology.org/D07-1015.pdf
  Args:
    log_potentials: Weight matrix with log-potential entries.
    single_root_edge: Whether to use a single-root constraint
  Returns:
    Laplacian matrix.
  """
  potentials = jnp.exp(jnp.logaddexp(log_potentials, MTT_LOG_EPS))
  potentials *= 1-jnp.eye(potentials.shape[-1])  # Removing self-edges
  laplacian = lambda x: x.sum(axis=-2, keepdims=True) * jnp.eye(x.shape[-1]) - x
  cut = lambda x: x[..., 1:, 1:]
  if single_root_edge:
    return laplacian(cut(potentials)).at[..., 0, :].set(potentials[..., 0, 1:])
  else:
    return cut(laplacian(potentials))


@typed
def _marginals_with_given_laplacian_invt(
    log_potentials: Float[Array, "*batch n n"],
    laplacian_invt: Float[Array, "*batch n-1 n-1"], single_root_edge: bool
    ) -> Float[Array, "*batch n n"]:
  """Computes marginals in cases where the inverse of the Laplacian is provided.

  This implementation exploits automatic differantiation concise implementation.
  This function is vector-Jacobian product of a function that constructs
  laplacian-hat matrix where primals are log-potentials and tangents come from
  inverse-transpose of the Laplacian-hat.
  For the explicit definition that doesn't use automatic differentation see
  Section 3.2 of Koo et al (2007) or NumPy implementation of this function
  within SynJax.

  References:
    Koo et al, 2007: https://aclanthology.org/D07-1015.pdf#page=5
  Args:
    log_potentials: Weight matrix with log-potential entries.
    laplacian_invt: Inverse-transpose of the Laplacian-hat matrix.
    single_root_edge: Whether to use a single-root constraint.
  Returns:
    Matrix of marginals.
  """
  _, vjp = jax.vjp(partial(_construct_laplacian_hat,
                           single_root_edge=single_root_edge), log_potentials)
  return vjp(laplacian_invt)[0]


@jax.custom_gradient
def sample_wilson_numpy_callback(
    log_potentials: jax.Array, lengths: jax.Array, single_root_edge: bool
    ) -> jax.Array:
  """JAX-to-Numba callback for vectorized sampling of spanning trees."""
  # The import is located here so that if users do not
  # call Numba code the Numba compilation won't be triggered and potential
  # irrelevant compilation errors won't appear.
  # pylint: disable=g-import-not-at-top
  # pylint: disable=import-outside-toplevel
  from synjax._src.deptree_algorithms import deptree_non_proj_wilson_sampling
  result_shape = jax.ShapeDtypeStruct(log_potentials.shape[:-1], jnp.int32)
  # pylint: disable=g-long-lambda
  f = lambda *x: deptree_non_proj_wilson_sampling.vectorized_sample_wilson(
      *jax.tree.map(np.asarray, x)).astype(jnp.int32)
  trees = jax.pure_callback(f, result_shape, log_potentials, lengths,
                            single_root_edge, vectorized=True)
  # pytype: disable=bad-return-type
  return (_to_adjacency_matrix(trees, lengths),
          lambda g: (jnp.zeros_like(log_potentials), None, None, None))
  # pytype: enable=bad-return-type


@jax.custom_gradient
def mst_numpy_callback(log_potentials: jax.Array, lengths: jax.Array,
                       single_root_edge: bool) -> jax.Array:
  """JAX-to-Numba callback for vectorized Tarjan's maximum spanning tree."""
  # The import is located here so that if users do not call Numba code the
  # Numba compilation won't be triggered and potential irrelevant
  # compilation errors won't appear.
  # pylint: disable=g-import-not-at-top
  # pylint: disable=import-outside-toplevel
  from synjax._src.deptree_algorithms import deptree_non_proj_argmax
  result_shape = jax.ShapeDtypeStruct(log_potentials.shape[:-1], jnp.int32)
  trees = jax.pure_callback(
      lambda *x: deptree_non_proj_argmax.vectorized_mst(
          *jax.tree.map(np.asarray, x)).astype(jnp.int32),
      result_shape, log_potentials, lengths, single_root_edge, vectorized=True)
  # pytype: disable=bad-return-type
  return (_to_adjacency_matrix(trees, lengths),
          lambda g: (jnp.zeros_like(log_potentials), None, None))
  # pytype: enable=bad-return-type
