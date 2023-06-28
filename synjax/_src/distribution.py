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

"""Definition of the interface of all SynJax distributions."""
from __future__ import annotations
# pylint: disable=g-multiple-import
# pylint: disable=g-long-lambda
# pylint: disable=g-importing-member
# pylint: disable=protected-access
import functools
from typing import TypeVar, cast, Optional, Union, Tuple
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Num, PyTree
from synjax._src.constants import INF
from synjax._src.typing import Key, Shape, typed
from synjax._src.utils import semirings
from synjax._src.utils import special


Self = TypeVar("Self")
Event = PyTree[Float[Array, "..."]]
SoftEvent = PyTree[Float[Array, "..."]]

vmap_ndim = special.vmap_ndim
grad_ndim = special.grad_ndim
partial = functools.partial
tree_leaves = jax.tree_util.tree_leaves

prob_clip = partial(jax.tree_map, lambda x: jnp.clip(jnp.nan_to_num(x), 0))
tlog = partial(jax.tree_map, special.safe_log)
tmul = partial(jax.tree_map, jnp.multiply)
tadd = partial(jax.tree_map, jnp.add)
tsub = partial(jax.tree_map, jnp.subtract)
tsum_all = lambda x: functools.reduce(jnp.add, map(jnp.sum, tree_leaves(x)))
is_shape = lambda x: isinstance(x, tuple) and all(isinstance(y, int) for y in x)


@typed
class Distribution(eqx.Module):
  """Abstract base class for all distributions."""

  log_potentials: Optional[PyTree[Float[Array, "..."]]]

  @typed
  def log_count(self, **kwargs) -> Float[Array, "*batch"]:
    """Log of the count of structures in the support."""
    replace_fn = lambda x: jnp.where(x <= -INF, -INF, 0)
    safe_replace_fn = lambda x: replace_fn(x) if eqx.is_inexact_array(x) else x
    return jax.tree_map(safe_replace_fn, self).log_partition(**kwargs)

  @property
  def event_shape(self) -> PyTree[int]:
    """PyTree of shapes of the event arrays."""
    raise NotImplementedError

  @property
  def batch_shape(self):
    """Shape of the batch."""
    if isinstance(self.log_potentials, Array) and is_shape(self.event_shape):
      return cast(Array, self.log_potentials).shape[:-len(self.event_shape)]
    else:
      raise NotImplementedError

  @property
  def batch_ndim(self):
    return len(self.batch_shape)

  @typed
  def _single_sample(self, key: Key, **kwargs) -> Event:
    raise NotImplementedError

  @typed
  def sample(self, key: Key, sample_shape: Union[int, Shape] = (), **kwargs
             ) -> Event:
    """Samples an event.

    Args:
      key: KeyArray key or integer seed.
      sample_shape: Additional leading dimensions for sample.
      **kwargs: Additional distribution specific kwargs.
    Returns:
      A sample of shape `sample_shape` + `batch_shape` + `event_shape`.
    """
    sample_shape = special.asshape(sample_shape)
    keys = special.split_key_for_shape(key, sample_shape)
    fn = lambda key: self._single_sample(key=key, **kwargs)
    return vmap_ndim(fn, len(sample_shape))(keys)

  @typed
  def log_prob(self, event: Event, **kwargs) -> Float[Array, "*batch"]:
    """Normalized Log probability of an event."""
    scores = self.unnormalized_log_prob(event, **kwargs)
    return scores - self.log_partition(**kwargs)

  @typed
  def unnormalized_log_prob(self, event: Event) -> Float[Array, "..."]:
    r"""Unnormalized probability of an event."""
    bcast_ndim = self._bcast_ndim(event)
    f = lambda a, b: jnp.sum(a*b, range(bcast_ndim+self.batch_ndim, a.ndim))
    leaf_sums = jax.tree_map(f, event, self.log_potentials)
    return vmap_ndim(tsum_all, bcast_ndim+self.batch_ndim)(leaf_sums)

  def _bcast_ndim(self, event: Event) -> int:
    leaf0 = lambda x: tree_leaves(x, is_leaf=is_shape)[0]
    return leaf0(event).ndim - len(leaf0(self.event_shape)) - self.batch_ndim

  @typed
  def log_partition(self, **kwargs) -> Float[Array, "*batch"]:
    """Log-partition function."""
    raise NotImplementedError

  @typed
  def marginals(self, **kwargs) -> SoftEvent:
    """Marginal probability of structure's parts."""
    def f(dist):
      return dist.log_partition(**kwargs)
    m = grad_ndim(f, self.batch_ndim)(self).log_potentials
    return prob_clip(m)

  @typed
  def log_marginals(self, **kwargs) -> SoftEvent:
    """Logs of marginal probability of structure's parts."""
    return tlog(self.marginals(**kwargs))

  @typed
  def argmax(self, **kwargs) -> Event:
    """Finds the highest scoring structure.

    Args:
      **kwargs: Keyword arguments for the underlying distribution.
    Returns:
      The highest scoring structure and its score. In case of ties some
      distributions return fractional structures (i.e. edges may not be only 0
      and 1 but any number in between). Those distributions support strict_max
      parameter that will arbitrarily break the ties and remove fractional
      structures at a price of needing more compute.
    """
    return self.argmax_and_max(**kwargs)[0]

  @typed
  def argmax_and_max(self, **kwargs) -> Tuple[Event, Float[Array, "*batch"]]:
    """Finds the highest scoring structure and its unnormalized score.

    Args:
      **kwargs: Keyword arguments for the underlying distribution.
    Returns:
      The highest scoring structure and its score. In case of ties some
      distributions return fractional structures (i.e. edges may not be only 0
      and 1 but any number in between). Those distributions support strict_max
      parameter that will arbitrarily break the ties and remove fractional
      structures at a price of needing more compute.
    """
    raise NotImplementedError

  @typed
  def top_k(self, k: int, approximate: bool = False, **kwargs
            ) -> Tuple[PyTree[Num[Array, "k ..."]], Float[Array, "k ..."]]:
    """Finds top-k structures and their scores."""
    raise NotImplementedError

  @typed
  def entropy(self, **kwargs) -> Float[Array, "*batch"]:
    """Calculates the Shannon entropy (in nats).

    Based on Li and Eisner (2009). Similar statements are appear in
    Martins et al (2010) and Zmigrod et al (2021).

    References:
      Li and Eisner, 2009 - Section 6.1: https://aclanthology.org/D09-1005.pdf#page=9
      Martins et al, 2010 - Equation 9: https://aclanthology.org/D10-1004.pdf#page=4
      Zmigrod et al, 2021 - Section 6.2: https://aclanthology.org/2021.tacl-1.41.pdf#page=10

    Args:
      **kwargs: Additional arguments for computation of marginals.
    Returns:
      Entropy value.
    """  # pylint: disable=line-too-long
    m = self.marginals(**kwargs)
    return -cast(Array, vmap_ndim(tsum_all, self.batch_ndim)(tmul(m, tlog(m))))

  @typed
  def cross_entropy(self: Self, other: Self, **kwargs
                    ) -> Float[Array, "*batch"]:
    """Calculates the cross entropy to another distribution (in nats).

    References:
      Li and Eisner, 2009 - Section 6.1: https://aclanthology.org/D09-1005.pdf#page=9

    Args:
      other: A compatible distribution.
      **kwargs: Additional arguments for computation of marginals.
    Returns:
      The cross entropy `H(self || other_dist)`.
    """  # pylint: disable=line-too-long
    log_q = other.log_marginals(**kwargs)
    p = self.marginals(**kwargs)
    return -cast(Array, vmap_ndim(tsum_all, self.batch_ndim)(tmul(p, log_q)))

  @typed
  def kl_divergence(self: Self, other: Self, **kwargs
                    ) -> Float[Array, "*batch"]:
    """Calculates the KL divergence to another distribution.

    References:
      Li and Eisner, 2009 - Section 6.1: https://aclanthology.org/D09-1005.pdf#page=9

    Args:
      other: A compatible distribution
      **kwargs: Additional arguments for computation of marginals.
    Returns:
      The KL divergence `KL(self || other)`.
    """  # pylint: disable=line-too-long
    log_q = other.log_marginals(**kwargs)
    p = self.marginals(**kwargs)
    return vmap_ndim(tsum_all, self.batch_ndim)(tmul(p, tsub(tlog(p), log_q)))

  def __getitem__(self: Self, i) -> Self:
    """If distribution is batched, indexes sub-distribution from the batch."""
    return jax.tree_map(lambda x: x[i], self)


class SemiringDistribution(Distribution):
  """Abstract class representing structured distributions based on semirings."""

  struct_is_isomorphic_to_params: bool = eqx.static_field(default=True)

  @typed
  def unnormalized_log_prob(self: Self, event: Event, **kwargs
                            ) -> Float[Array, "..."]:
    r"""Unnormalized score of an event.

    Args:
      event: Structures that distribution can broadcast over.
      **kwargs: Additional keyword arguments that are passed to
                log-partition function.
    Returns:
      Unnormalized log-probs for each sample.
    """
    if self.struct_is_isomorphic_to_params:
      return super().unnormalized_log_prob(event)
    else:
      # This is useful mostly for distributions like PCFG where parameters
      # (in PCFG case that is the grammar) are not of the same form as marginals
      # (in PCFG case that is a chart).
      sr = semirings.LogSemiring()
      key = jax.random.PRNGKey(0)
      def f_single_sample_single_batch(
          base_struct: SoftEvent, dist: Self) -> Float[Array, ""]:
        return sr.unwrap(dist._structure_forward(
            jax.tree_map(lambda x: jnp.where(x, 0, -INF), base_struct),
            sr, key=key, **kwargs))
      def f_single_sample_multi_batch(base_struct: SoftEvent
                                      ) -> Float[Array, "*batch"]:
        return vmap_ndim(f_single_sample_single_batch, self.batch_ndim
                         )(base_struct, self)
      def f_multi_sample_multi_batch(base_struct: SoftEvent
                                     ) -> Float[Array, "*sample_batch"]:
        return vmap_ndim(f_single_sample_multi_batch, self._bcast_ndim(event)
                         )(base_struct)
      log_probs = f_multi_sample_multi_batch(event)
      return log_probs

  @typed
  def log_partition(self, **kwargs) -> Float[Array, "*batch"]:
    """Compute the log-partition function."""
    sr = semirings.LogSemiring()
    def f(dist, base):
      return dist._structure_forward(base, sr, jax.random.PRNGKey(0), **kwargs)
    result = vmap_ndim(f, self.batch_ndim)(self, self._batched_base_structure())
    return sr.unwrap(jnp.moveaxis(result, -1, 0))

  @typed
  def argmax_and_max(self, strict_max: Optional[bool] = None, **kwargs
                     ) -> Tuple[Event, Float[Array, "*batch"]]:
    """Calculates the argmax and max."""
    sr = semirings.MaxSemiring(strict_max=strict_max)
    def f(base_struct, dist):
      max_score = dist._structure_forward(
          base_struct, sr, key=jax.random.PRNGKey(0), **kwargs)
      max_score = sr.unwrap(max_score)
      return max_score, max_score
    max_structs, max_scores = grad_ndim(f, self.batch_ndim, has_aux=True
                                        )(self._batched_base_structure(), self)
    return max_structs, max_scores

  @typed
  def marginals(self: Self, **kwargs) -> SoftEvent:
    """Marginal probability of structure's parts."""
    sr = semirings.LogSemiring()
    def f(base_struct: SoftEvent, dist: Self) -> Float[Array, "*batch"]:
      return sr.unwrap(dist._structure_forward(
          base_struct, sr, key=jax.random.PRNGKey(0), **kwargs))
    m = grad_ndim(f, self.batch_ndim)(self._batched_base_structure(), self)
    return prob_clip(m)

  @typed
  def _single_sample(self, key: Key, **kwargs) -> Event:
    """Finds a single sample per each batched distribution.

    Args:
      key: KeyArray to use for sampling. It is a single key that will be
           split for each batch element.
      **kwargs: Any additional arguments needed for forward pass
    Returns:
      Single sample for each distribution in the batch.
    """
    keys = special.split_key_for_shape(key, self.batch_shape)
    sr = semirings.SamplingSemiring()
    def f(base_struct, dist, akey):
      return sr.unwrap(dist._structure_forward(base_struct, sr, akey, **kwargs))
    samples = grad_ndim(f, self.batch_ndim
                        )(self._batched_base_structure(), self, keys)
    return samples

  @typed
  def top_k(self, k: int, approximate: bool = False, **kwargs
            ) -> Tuple[PyTree[Num[Array, "k ..."]], Float[Array, "k ..."]]:
    """Finds top_k structures.

    Args:
      k: Number of top elements.
      approximate: Should k-best be approximate.
      **kwargs: Additional kwargs for the distribution specific forward method.
    Returns:
      A tuple where first element is an array of top k structures and second
      element is an array of their scores that are unnormalized.
    """
    if k <= 0:
      raise ValueError("k must be a strictly positive integer")
    if k == 1:
      # This is a shortcut optimization for a special case.
      best, score = self.argmax_and_max()
      expand = partial(jax.tree_map, lambda x: x[None])
      return expand(best), expand(score)
    def kbest_forward(base_struct, dist):
      kbest_scores = dist._structure_forward(
          base_struct, semirings.KBestSemiring(k, approximate=approximate),
          key=jax.random.PRNGKey(0), **kwargs)
      return kbest_scores, kbest_scores
    def kbest_per_dist(base_struct, dist):
      return jax.jacrev(kbest_forward, has_aux=True)(base_struct, dist)
    kbest_structs, kbest_scores = vmap_ndim(kbest_per_dist, self.batch_ndim)(
        self._batched_base_structure(), self)
    move = lambda x: jnp.moveaxis(x, self.batch_ndim, 0)
    kbest_structs = jax.tree_map(move, kbest_structs)
    kbest_scores = move(kbest_scores)
    return kbest_structs, kbest_scores

  @typed
  def _batched_base_structure(self) -> SoftEvent:
    leaves_shapes, defs = jax.tree_util.tree_flatten(self.event_shape, is_shape)
    leaves = [jnp.zeros(self.batch_shape+shape) for shape in leaves_shapes]
    return jax.tree_util.tree_unflatten(defs, leaves)

  @typed
  def _structure_forward(
      self, base_struct: SoftEvent,
      semiring: semirings.Semiring, key: Key, **kwargs) -> Float[Array, "s"]:
    """Computes partition under a semiring for single instance."""
    raise NotImplementedError
