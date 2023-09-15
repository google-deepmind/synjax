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
from dataclasses import replace
from functools import partial
from typing import TypeVar, cast, Optional, Union, Tuple, Literal, Callable
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import Array, ArrayLike, Float, Num, PyTree, Int32
from synjax._src.config import get_config
from synjax._src.constants import INF
from synjax._src.typing import Key, Shape, typed
from synjax._src.utils import perturbation_utils
from synjax._src.utils import semirings
from synjax._src.utils import special

vmap_ndim = special.vmap_ndim
grad_ndim = special.grad_ndim
is_shape = special.is_shape

Self = TypeVar("Self")
Event = PyTree[Float[Array, "..."]]
SoftEvent = PyTree[Float[Array, "..."]]


@typed
class Distribution(eqx.Module):
  """Abstract base class for all distributions."""

  log_potentials: Optional[PyTree[Float[Array, "..."]]]
  struct_is_isomorphic_to_params: bool = eqx.static_field(default=True)

  @typed
  def log_count(self, **kwargs) -> Float[Array, "*batch"]:
    """Log of the count of structures in the support."""
    params, non_params = eqx.partition(self, eqx.is_inexact_array)
    params = jax.tree_map(lambda x: jnp.where(x <= -INF, -INF, 0), params)
    return eqx.combine(params, non_params).log_partition(**kwargs)

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
  def _typical_number_of_parts_per_event(self) -> Int32[Array, "*batch"]:
    raise NotImplementedError

  @property
  def batch_ndim(self):
    return len(self.batch_shape)

  @typed
  def _single_sample(self, key: Key, **kwargs) -> Event:
    raise NotImplementedError

  @typed
  def sample(self, key: Key, sample_shape: Union[int, Shape] = (),
             temperature: Float[ArrayLike, ""] = 1., **kwargs) -> Event:
    """Unbiased sampling of a structure.

    Args:
      key: Key array.
      sample_shape: Additional leading dimensions for sample.
      temperature: Sampling temperature.
      **kwargs: Additional distribution specific kwargs.
    Returns:
      A sample of shape `sample_shape` + `batch_shape` + `event_shape`.
    """
    sample_shape = special.asshape(sample_shape)
    keys = special.split_key_for_shape(key, sample_shape)
    dist = self.with_temperature(temperature)
    fn = lambda key: dist._single_sample(key=key, **kwargs)
    return vmap_ndim(fn, len(sample_shape))(keys)

  def with_temperature(self: Self, temperature: Float[ArrayLike, ""] = 1.
                       ) -> Self:
    return special.tscale_inexact_arrays(1/temperature, self)

  @typed
  def with_perturbation(
      self: Self,
      key: Key,
      noise: Union[Literal["Sum-of-Gamma", "Gumbel", "None"],
                   Callable[..., Array]],
      temperature: Float[ArrayLike, ""]) -> Self:
    eta = perturbation_utils.noise_for_pytree(key, self._noise_fn(noise), self)
    return special.tadd(self, eta).with_temperature(temperature)

  def _noise_fn(
      self, noise: Union[Literal["Sum-of-Gamma", "Gumbel", "None"],
                         Callable[..., Array]]) -> Callable[..., Array]:
    if noise == "Sum-of-Gamma":
      return partial(perturbation_utils.sample_sum_of_gamma,
                     k=self._typical_number_of_parts_per_event,
                     s=get_config().sum_of_gamma_s)
    elif noise == "Gumbel":
      return jax.random.gumbel
    elif noise == "None":
      return lambda key, shape: jnp.zeros(shape)
    elif isinstance(noise, Callable):
      return noise
    else:
      raise NotImplementedError

  @typed
  def differentiable_sample(
      self, key: Key,
      *,
      method: Literal["Perturb-and-Marginals", "Perturb-and-SoftmaxDP",
                      "Perturb-and-SparsemaxDP", "Perturb-and-MAP-Implicit-MLE",
                      "Gumbel-CRF"],
      noise: Union[Literal["Sum-of-Gamma", "Gumbel", "None"],
                   Callable[..., Array]] = "Sum-of-Gamma",
      temperature: Float[ArrayLike, ""] = 1.,
      sample_shape: Union[int, Shape] = (),
      straight_through: bool = False,
      implicit_MLE_lr: Float[ArrayLike, ""] = 1.,  # pylint: disable=invalid-name
      ) -> SoftEvent:
    """Biased differentiable sampling of a structure.

    With this method, with a price of having some bias, we can
    get differentiable samples. Most of the supported sampling methods are based
    on some form of perturbation that is followed by a custom decoding method.

    Args:
      key: PRNGKey for sampling.
      method: Which of the available methods should be used for producing a
              differentiable sample. Currently supported are
              - Perturb-and-Marginals -- the same as Stochastic-Softmax-Tricks
                from (Paulus et al, 2021) which applies weight perturbations
                before computing marginals.
              - Perturb-and-SoftmaxDP (Corro and Titov, 2019) is using
                perturbation of weights followed by a smoothed dynamic
                programming by (Mensch and Blondel, 2018) with negative-entropy
                smoothing which is the same as replacing argmax with softmax.
              - Perturb-and-SparsemaxDP -- essentially the same as
                Perturb-and-SoftmaxDP but with squared-L2 smoothing from
                (Mensch and Blondel, 2018) which amounts to using sparsemax from
                (Martins and Astudillo, 2016) as an approximate argmax.
              - Perturb-and-MAP-Implicit-MLE (Niepert et al, 2021) does
                perturbed argmax decoding with a custom backward step that
                projects structure's gradient to a new structure.
              - Gumbel-CRF is from (Fu et al 2020) and reparametrizes the
                backward step of Forward-Filtering Backward-Sampling by using
                Gumbel-Softmax trick.
      noise: Noise type to use for perturbation. The available options are
             Sum-of-Gamma, Gumbel and None. A sum of a k Sum-of-Gamma samples
             approximates a single sample of Gumbel better than a sum of
             k Gumbel samples. Therefore it is is interesting to use for
             structures that decompose into k parts. Sum-of-Gramma is described
             in (Niepert et al, 2021).
             If noise is set to "None" it can produce deterministic
             structures, e.g. Perturb-and-Marginals with "None" for noise is
             equivalent to Structured-Attention from (Kim et al, 2017).
             Noise argument can also be any Callable that takes key and shape
             arguments.
      temperature: Temperature for perturbation. Lower temperature implies being
                   closer to a peaked distribution.
      sample_shape: Additional leading dimensions for sample in case multiple
                    samples are needed.
      straight_through: If the returned sample should be discretized by passing
                        through the gradients using straight-through estimator.
                        This influences Gumbel-CRF and Perturb-and-SoftmaxDP.
      implicit_MLE_lr: Internal learning rate used for
                       Perturb-and-MAP-Implicit-MLE update.
                       Niepert et al. (2021) refer to it as lambda parameter.
    Returns:
      A sample of shape `sample_shape` + `batch_shape` + `event_shape`.
    References:
      Niepert et al, 2021: https://arxiv.org/pdf/2106.01798.pdf
      Paulus et al, 2021: https://arxiv.org/pdf/2006.08063.pdf
      Fu et al, 2020: https://proceedings.neurips.cc/paper/2020/file/ea119a40c1592979f51819b0bd38d39d-Paper.pdf
      Corro and Titov, 2019: https://openreview.net/pdf?id=BJlgNh0qKQ
      Mensch and Blondel, 2018: https://arxiv.org/pdf/1802.03676.pdf
      Kim et al, 2017: https://openreview.net/pdf?id=HkE0Nvqlg
      Martins and Astudillo, 2016: http://proceedings.mlr.press/v48/martins16.pdf
    """  # pylint: disable=line-too-long
    bcast = lambda x: jnp.broadcast_to(x, special.asshape(sample_shape)+x.shape)
    dist = jax.tree_map(bcast, self)

    if method in ["Perturb-and-SoftmaxDP", "Perturb-and-SparsemaxDP",
                  "Gumbel-CRF"] and not isinstance(self, SemiringDistribution):
      raise ValueError(f"Method {method} can be applied only to distributions "
                       "that use dynamic programming.")

    if method == "Perturb-and-Marginals":
      return dist.with_perturbation(key, noise, temperature).marginals()
    elif method == "Perturb-and-SoftmaxDP":
      smoothing = "st-softmax" if straight_through else "softmax"
      return dist.with_perturbation(key, noise, temperature
                                    ).argmax(smoothing=smoothing)
    elif method == "Perturb-and-SparsemaxDP":
      return dist.with_perturbation(key, noise, temperature
                                    ).argmax(smoothing="sparsemax")
    elif method == "Gumbel-CRF":
      relaxation = "ST-Gumbel-Softmax" if straight_through else "Gumbel-Softmax"
      return self.sample(key, sample_shape=sample_shape,
                         temperature=temperature, relaxation=relaxation)
    elif method == "Perturb-and-MAP-Implicit-MLE":
      # Check 1
      if not dist.struct_is_isomorphic_to_params:
        raise ValueError("Implicit MLE cannot be used with distributions that"
                         " not isomorphic to with their parameters.")
      # Check 2
      if not hasattr(dist, "log_potentials") or dist.log_potentials is None:
        raise ValueError(
            "Implicit MLE requires distribution with log_potentials attribute.")
      # Call Implicit-MLE
      sampling_fn = perturbation_utils.implicit_mle(
          noise_fn=dist._noise_fn(noise),
          argmax_fn=lambda lp, dst: replace(dst, log_potentials=lp).argmax(),
          internal_learning_rate=implicit_MLE_lr, temperature=temperature)
      return sampling_fn(key, dist.log_potentials, dist)
    else:
      raise NotImplementedError

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
    return vmap_ndim(special.tsum_all, bcast_ndim+self.batch_ndim)(leaf_sums)

  def _bcast_ndim(self, event: Event) -> int:
    leaf0 = lambda x: jtu.tree_leaves(x, is_leaf=is_shape)[0]
    return leaf0(event).ndim - len(leaf0(self.event_shape)) - self.batch_ndim

  @typed
  def log_partition(self, **kwargs) -> Float[Array, "*batch"]:
    """Log-partition function."""
    raise NotImplementedError

  @typed
  def marginals_for_template_variables(self: Self, **kwargs) -> Self:
    """Marginal prob. of template parts (e.g. PCFG rules instead tree nodes)."""
    grad_f = grad_ndim(lambda x: x.log_partition(**kwargs), self.batch_ndim)
    return jax.tree_map(jnp.nan_to_num, grad_f(self))

  @typed
  def marginals(self, **kwargs) -> SoftEvent:
    """Marginal probability of structure's parts."""
    return self.marginals_for_template_variables(**kwargs).log_potentials

  @typed
  def log_marginals(self, **kwargs) -> SoftEvent:
    """Logs of marginal probability of structure's parts."""
    return jax.tree_map(special.safe_log, self.marginals(**kwargs))

  @typed
  def argmax(self, **kwargs) -> Event:
    """Finds the highest scoring structure.

    Args:
      **kwargs: Keyword arguments for the underlying distribution.
    Returns:
      The highest scoring structure and its score. In case of ties some
      distributions return fractional structures (i.e. edges may not be only 0
      and 1 but any number in between).
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
      and 1 but any number in between).
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
    return self.cross_entropy(self, **kwargs)

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
    def param_leaves(x):
      return [y for y in jtu.tree_leaves(x) if eqx.is_inexact_array(y)]
    p_marginals = param_leaves(self.marginals_for_template_variables(**kwargs))
    q_log_potentials = param_leaves(other)
    q_log_z = other.log_partition(**kwargs)
    return q_log_z - vmap_ndim(special.tsum_all, self.batch_ndim
                               )(special.tmul(p_marginals, q_log_potentials))

  @typed
  def kl_divergence(self: Self, other: Self, **kwargs
                    ) -> Float[Array, "*batch"]:
    """Calculates the KL divergence to another distribution (in nats).

    References:
      Li and Eisner, 2009 - Section 6.1: https://aclanthology.org/D09-1005.pdf#page=9

    Args:
      other: A compatible distribution
      **kwargs: Additional arguments for computation of marginals.
    Returns:
      The KL divergence `KL(self || other)`.
    """  # pylint: disable=line-too-long
    return self.cross_entropy(other, **kwargs) - self.entropy(**kwargs)

  def __getitem__(self: Self, i) -> Self:
    """If distribution is batched, indexes sub-distribution from the batch."""
    return jax.tree_map(lambda x: x[i], self)


class SemiringDistribution(Distribution):
  """Abstract class representing structured distributions based on semirings."""

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
  def argmax_and_max(
      self,
      smoothing: Optional[Literal["softmax", "st-softmax", "sparsemax"]] = None,
      temperature: float = 1.,
      **kwargs) -> Tuple[Event, Float[Array, "*batch"]]:
    """Calculates the argmax and max."""
    sr = semirings.MaxSemiring(smoothing=smoothing, temperature=temperature)
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
    return jax.tree_map(jnp.nan_to_num, m)

  @typed
  def _single_sample(
      self, key: Key,
      relaxation: Optional[Literal["Gumbel-Softmax", "ST-Gumbel-Softmax"]
                           ] = None,
      **kwargs) -> Event:
    """Finds a single sample per each batched distribution.

    Args:
      key: Key array to use for sampling. It is a single key that will be
           split for each batch element.
      relaxation: Biased relaxation, if any.
      **kwargs: Any additional arguments needed for forward pass
    Returns:
      Single sample for each distribution in the batch.
    """
    keys = special.split_key_for_shape(key, self.batch_shape)
    sr = semirings.SamplingSemiring(relaxation=relaxation)
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
