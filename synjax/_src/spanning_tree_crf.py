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

"""Distribution of spanning trees."""
# pylint: disable=g-multiple-import,g-importing-member
from __future__ import annotations
from typing import cast, Optional, Union, Tuple

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float, Int32
from synjax._src.distribution import Distribution
from synjax._src.spanning_tree_non_projective_crf import SpanningTreeNonProjectiveCRF
from synjax._src.spanning_tree_projective_crf import SpanningTreeProjectiveCRF
from synjax._src.typing import Shape, Key, typed


class SpanningTreeCRF(Distribution):
  """Unified interface to all spanning tree distributions."""

  directed: bool = eqx.static_field()
  projective: bool = eqx.static_field()
  single_root_edge: bool = eqx.static_field()
  _dist: Union[SpanningTreeNonProjectiveCRF, SpanningTreeProjectiveCRF]

  @typed
  def __init__(self,
               log_potentials: Float[Array, "*batch n n"],
               *,
               directed: bool,
               projective: bool,
               single_root_edge: bool,
               lengths: Optional[Int32[Array, "*batch"]] = None):
    """Spanning Tree Conditional-Random Field distribution.

    This distribution is used for modeling spanning trees of graphs with n
    nodes. In case it is a directed spanning tree (i.e. arborescence) The 0th
    nodes is treated as the special root node by convention. In case the graph
    is undirected, the log-potentials will be symmetrized (this is a noop if
    they are already symmetric). The implementation optionally allows for
    a constraint that enforces the spanning trees to have only one edge coming
    out of the root node. It also optionally allows for constraining trees to be
    projective, oftentimes useful in dependency parsing of natural language.
    See Koo et al (2007) for a standard version of this model.

    References:
      Koo et al, 2007: https://aclanthology.org/D07-1015.pdf
      Stanojević, 2022: https://aclanthology.org/2022.emnlp-main.110.pdf
      Stanojević and Cohen, 2021: https://aclanthology.org/2021.emnlp-main.823.pdf
      Zmigrod et al, 2021: https://aclanthology.org/2021.emnlp-main.824v2.pdf
      Zmigrod et al, 2021: https://arxiv.org/pdf/2008.12988.pdf
      Colbourn et al, 1996: https://www.sciencedirect.com/science/article/pii/S0196677496900140
      Kuhlmann et al, 2011: https://aclanthology.org/P11-1068.pdf
      Eisner, 2000: https://www.cs.jhu.edu/~jason/papers/eisner.iwptbook00.pdf
    Args:
      log_potentials: jax.Array of shape (..., n, n). If graph is directed, 0th
                      column will be ignored because 0th node will be treated as
                      the root node. If graph is undirected log-potentials will
                      be symmetrized -- this is a noop if they are already
                      symmetric.
      directed: Boolean flag signifying if the tree is directed (arborescence).
      projective: Boolean flag signifying if the tree should be projective,
                  i.e. if there should be no crossing tree branches when nodes
                  are positioned on a single line with their canonical order
                  (see Eisner, 2020).
      single_root_edge: Boolean flag signifying if the number of arcs leaving
                        root node (node at position 0) should be exactly 1.
      lengths: Optional array providing the length of non-root nodes in the
               graphs. The "non-root" part is important. This array,
               if provided, will be used for automatic padding.
    """  # pylint: disable=line-too-long
    super().__init__(log_potentials=None)
    self.directed = directed
    self.projective = projective
    self.single_root_edge = single_root_edge
    if not directed:
      # Symmetrize log_potentials.
      log_potentials = (log_potentials + jnp.swapaxes(log_potentials, -2, -1))/2
    cls = (SpanningTreeProjectiveCRF if projective
           else SpanningTreeNonProjectiveCRF)
    self._dist = cls(log_potentials, lengths=lengths,
                     single_root_edge=single_root_edge)

  @property
  def event_shape(self) -> Shape:
    return self._dist.event_shape

  @property
  def batch_shape(self) -> Shape:
    return self._dist.batch_shape

  @property
  def lengths(self) -> Array:
    return self._dist.lengths

  @typed
  def _remove_padding(self, event: Float[Array, "*xy n n"]
                      ) -> Float[Array, "*xy n n"]:
    """Removes padding elements introduced for computing log-partition."""
    x = jnp.arange(event.shape[-1]) < self.lengths[..., None]
    mask = x[..., None, :] & x[..., None]
    return jnp.where(mask, event, 0)

  @typed
  def sample_without_replacement(self, key: Key, k: int
                                 ) -> Tuple[Float[Array, "k *batch n n"],
                                            Float[Array, "k *batch"],
                                            Float[Array, "k *batch"]]:
    """Sampling without replacement from Stanojević (2022).

    References:
      Stanojević, 2022: https://aclanthology.org/2022.emnlp-main.110.pdf
    Args:
      key: Sampling key.
      k: The number of required samples without replacement.
    Returns:
      Tuple of (samples, logprobs, gumbel perturbed logprobs)
    """
    if self.projective:
      raise NotImplementedError("There is no implementation of sampling "
                                "without replacement for projective trees.")
    dist = cast(SpanningTreeNonProjectiveCRF, self._dist)
    samples, logprobs, gumbel_logprobs = dist.sample_without_replacement(key, k)
    if not self.directed:
      samples = samples + jnp.swapaxes(samples, -2, -1)
    samples = self._remove_padding(samples)
    return samples, logprobs, gumbel_logprobs

  @typed
  def sample(self, key: Key, sample_shape: Union[Shape, int] = (), **kwargs
             ) -> Float[Array, "... n n"]:
    samples = self._dist.sample(key=key, sample_shape=sample_shape, **kwargs)
    if not self.directed:
      samples = samples + jnp.swapaxes(samples, -2, -1)
    samples = self._remove_padding(samples)
    return samples

  @typed
  def differentiable_sample(self, **kwargs,) -> Float[Array, "... n n"]:
    samples = self._dist.differentiable_sample(**kwargs)
    if not self.directed:
      samples = samples + jnp.swapaxes(samples, -2, -1)
    samples = self._remove_padding(samples)
    return samples

  @typed
  def normalize_log_probs(self, scores: Float[Array, "*b"]
                          ) -> Float[Array, "*b"]:
    return self._dist.normalize_log_probs(scores)

  @typed
  def log_prob(self, event: Float[Array, "*b n n"], **kwargs
               ) -> Float[Array, "*b"]:
    event = self._remove_padding(event)
    if not self.directed:
      event = jnp.triu(event)
    return self._dist.log_prob(event, **kwargs)

  @typed
  def unnormalized_log_prob(self, event: Float[Array, "*b n n"], **kwargs
                            ) -> Float[Array, "*b"]:
    event = self._remove_padding(event)
    if not self.directed:
      event = jnp.triu(event)
    return self._dist.unnormalized_log_prob(event, **kwargs)

  @typed
  def log_partition(self, **kwargs) -> Float[Array, "*batch"]:
    return self._dist.log_partition(**kwargs)

  @typed
  def marginals_for_template_variables(self, **kwargs
                                       ) -> Float[Array, "*batch n n"]:
    return self._dist.marginals_for_template_variables(**kwargs)

  @typed
  def marginals(self, **kwargs) -> Float[Array, "*batch n n"]:
    m = self._dist.marginals(**kwargs)
    if not self.directed:
      m = m + jnp.swapaxes(m, -2, -1)
    m = self._remove_padding(m)
    return m

  @typed
  def argmax(self, **kwargs) -> Float[Array, "*batch n n"]:
    tree = self._dist.argmax(**kwargs)
    if not self.directed:
      tree = tree + jnp.swapaxes(tree, -2, -1)
    tree = self._remove_padding(tree)
    return tree

  @typed
  def argmax_and_max(self, **kwargs) -> Tuple[Float[Array, "*batch n n"],
                                              Float[Array, "*batch"]]:
    tree, score = self._dist.argmax_and_max(**kwargs)
    if not self.directed:
      tree = tree + jnp.swapaxes(tree, -2, -1)
    tree = self._remove_padding(tree)
    return tree, score

  @typed
  def top_k(self, k: int, **kwargs) -> Tuple[Float[Array, "k *batch n n"],
                                             Float[Array, "k *batch"]]:
    trees, scores = self._dist.top_k(k, **kwargs)
    if not self.directed:
      trees = trees + jnp.swapaxes(trees, -2, -1)
    trees = self._remove_padding(trees)
    return trees, scores

  @typed
  def entropy(self, **kwargs) -> Float[Array, "*batch"]:
    return self._dist.entropy(**kwargs)

  @typed
  def cross_entropy(self, other: SpanningTreeCRF, **kwargs
                    ) -> Float[Array, "*batch"]:
    if self.directed != other.directed:
      raise ValueError("Cross entropy cannot be computed between directed and"
                       "undirected spanning tree distributions.")
    # pylint: disable=protected-access
    return self._dist.cross_entropy(other._dist, **kwargs)

  @typed
  def kl_divergence(self, other: SpanningTreeCRF, **kwargs
                    ) -> Float[Array, "*batch"]:
    if self.directed != other.directed:
      raise ValueError("Cross entropy cannot be computed between directed and"
                       "undirected spanning tree distributions.")
    # pylint: disable=protected-access
    return self._dist.kl_divergence(other._dist, **kwargs)
