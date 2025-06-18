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

"""Distribution of alignments between two sequences."""
from __future__ import annotations
# pylint: disable=g-multiple-import, g-importing-member
from functools import partial
from typing import Optional, Tuple, Literal, Union
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int32
import numpy as np
import scipy
from synjax._src.alignment_monotone_general import GeneralMonotoneAlignmentCRF
from synjax._src.constants import INF
from synjax._src.distribution import Distribution
from synjax._src.typing import Shape, Key, typed


def _numpy_non_monotone_align(log_potentials):
  vectorized_linear_sum_assignment = np.vectorize(
      partial(scipy.optimize.linear_sum_assignment, maximize=True),
      signature="(n, n)->(n),(n)")
  i, j = vectorized_linear_sum_assignment(log_potentials)
  one_hot = lambda x: np.arange(log_potentials.shape[-1]) == x[..., None]
  return np.sum(one_hot(i)[..., :, None]*one_hot(j)[..., None, :], axis=-3,
                dtype=np.float32)


@jax.custom_gradient
@typed
def _jax_non_monotone_align_callback(log_potentials: Float[Array, "*b n n"],
                                     lenghts: Int32[Array, "*b"]):
  """Computes non-monotone alignment using JAX pure callback to NumPy."""
  n = log_potentials.shape[-1]
  mask = jnp.arange(n) < lenghts[..., None]
  mask = mask[..., None] * mask[..., None, :]
  log_potentials = jnp.where(mask, log_potentials, -INF)
  diag_mask = (jnp.arange(n) >= lenghts[..., None, None]) * jnp.eye(n)
  log_potentials = jnp.where(diag_mask, INF, log_potentials)
  alignments = jax.pure_callback(
      _numpy_non_monotone_align, jax.ShapeDtypeStruct(log_potentials.shape,
                                                      jnp.float32),
      log_potentials)
  return alignments*mask, lambda g: (jnp.zeros_like(log_potentials), None)


class AlignmentCRF(Distribution):
  """Simple alignment CRF that covers most use-cases.

  This alignment class provides both monotone and non-monotone alignment, but
  restricts all alignment scores to be defined per cell, i.e. the score does not
  depend on the direction of the path trough the alignment table but only on the
  actual cells it visits. For a more general type of monotone alignment see
  GeneralMonotoneAligmentCRF class.
  """

  lengths_rows: Int32[Array, "*batch"]
  lengths_cols: Int32[Array, "*batch"]
  _dist: Optional[GeneralMonotoneAlignmentCRF]
  alignment_type: str = eqx.field(static=True)

  @typed
  def __init__(self, log_potentials: Float[Array, "*batch row col"], *,
               lengths_rows: Optional[Int32[Array, "*batch"]] = None,
               lengths_cols: Optional[Int32[Array, "*batch"]] = None,
               alignment_type: Literal["monotone_one_to_many",
                                       "monotone_many_to_many",
                                       "non_monotone_one_to_one"], **kwargs):
    if "_dist" in kwargs:
      # Ignore _dist kwarg if provided via dataclasses.replace
      del kwargs["_dist"]

    super().__init__(log_potentials=log_potentials,
                     **(dict(struct_is_isomorphic_to_params=True) | kwargs))
    if lengths_rows is not None:
      self.lengths_rows = lengths_rows
    else:
      self.lengths_rows = jnp.full(self.log_potentials.shape[:-2],
                                   self.log_potentials.shape[-2])

    if lengths_cols is not None:
      self.lengths_cols = lengths_rows
    else:
      self.lengths_cols = jnp.full(self.log_potentials.shape[:-2],
                                   self.log_potentials.shape[-1])

    self.alignment_type = alignment_type
    if (lengths_cols is None and lengths_rows is None
        and alignment_type == "monotone_one_to_many"
        and log_potentials.shape[-2] >= log_potentials.shape[-1]):
      raise ValueError("This is a useless distribution because there is "
                       "less than two alignment possible.")

    if alignment_type == "monotone_one_to_many":
      self._dist = GeneralMonotoneAlignmentCRF(
          log_potentials_horizontal=(log_potentials, log_potentials),
          log_potentials_vertical=None,
          lengths_rows=lengths_rows, lengths_cols=lengths_cols)
    elif alignment_type == "monotone_many_to_many":
      self._dist = GeneralMonotoneAlignmentCRF(
          log_potentials_horizontal=(log_potentials, log_potentials),
          log_potentials_vertical=log_potentials,
          lengths_rows=lengths_rows, lengths_cols=lengths_cols)
    elif alignment_type == "non_monotone_one_to_one":
      self._dist = None
      if lengths_cols is not None:
        raise ValueError("Non-monotone alignment requires only lengths_rows.")
      if log_potentials.shape[-1] != log_potentials.shape[-2]:
        raise ValueError("Non-monotone alignment requires square matrix.")
    else:
      raise ValueError(f"Unknown alignment type: {alignment_type}")

  @property
  def event_shape(self) -> Shape:
    return self.log_potentials.shape[-2:]

  @property
  def batch_shape(self) -> Shape:
    return self.log_potentials.shape[:-2]

  @property
  def _typical_number_of_parts_per_event(self) -> Int32[Array, "*batch"]:
    return self.lengths_rows

  @typed
  def sample(self, key: Key, sample_shape: Union[Shape, int] = (), **kwargs
             ) -> Float[Array, "... n m"]:
    if self.alignment_type == "non_monotone_one_to_one":
      raise NotImplementedError(
          "Non-monotone distribution doesn't support sampling.\n"
          "Instead, you can try differentiable sampling with "
          "Perturb-and-MAP-Implicit-MLE.")
    else:
      return self._dist.sample(key=key, sample_shape=sample_shape, **kwargs)

  @typed
  def differentiable_sample(self, **kwargs) -> Float[Array, "*batch n m"]:
    return super().differentiable_sample(**kwargs)

  @typed
  def normalize_log_probs(self, scores: Float[Array, "*b"]
                          ) -> Float[Array, "*b"]:
    if self.alignment_type == "non_monotone_one_to_one":
      raise NotImplementedError(
          "Non-monotone distribution doesn't support normalization.")
    else:
      return self._dist.normalize_log_probs(scores)

  @typed
  def log_prob(self, event: Float[Array, "*b n m"], **kwargs
               ) -> Float[Array, "*b"]:
    if self.alignment_type == "non_monotone_one_to_one":
      raise NotImplementedError(
          "Non-monotone alignment doesn't support normalized log-probs.")
    else:
      return self._dist.log_prob(event, **kwargs)

  @typed
  def unnormalized_log_prob(self, event: Float[Array, "*b n m"], **kwargs
                            ) -> Float[Array, "*b"]:
    if self.alignment_type == "non_monotone_one_to_one":
      return jnp.einsum("...ij,...ij->...", event, self.log_potentials)
    else:
      return self._dist.unnormalized_log_prob(event, **kwargs)

  @typed
  def log_partition(self, **kwargs) -> Float[Array, "*batch"]:
    if self.alignment_type == "non_monotone_one_to_one":
      raise NotImplementedError(
          "Non-monotone alignment doesn't support log-partition.")
    else:
      return self._dist.log_partition(**kwargs)

  @typed
  def marginals_for_template_variables(self, **kwargs
                                       ) -> Float[Array, "*batch n m"]:
    if self.alignment_type == "non_monotone_one_to_one":
      raise NotImplementedError(
          "Non-monotone alignment doesn't support marginals.")
    else:
      return self._dist.marginals_for_template_variables(**kwargs)

  @typed
  def marginals(self, **kwargs) -> Float[Array, "*batch n m"]:
    if self.alignment_type == "non_monotone_one_to_one":
      raise NotImplementedError(
          "Non-monotone alignment doesn't support marginals.")
    else:
      return self._dist.marginals(**kwargs)

  @typed
  def argmax(self, **kwargs) -> Float[Array, "*batch n m"]:
    if self.alignment_type == "non_monotone_one_to_one":
      return _jax_non_monotone_align_callback(self.log_potentials,
                                              self.lengths_rows)
    else:
      return self._dist.argmax(**kwargs)

  @typed
  def argmax_and_max(self, **kwargs) -> Tuple[Float[Array, "*batch n m"],
                                              Float[Array, "*batch"]]:
    event = self.argmax(**kwargs), self
    return event, self.unnormalized_log_prob(event, **kwargs)

  @typed
  def top_k(self, k: int, **kwargs) -> Tuple[Float[Array, "k *batch n m"],
                                             Float[Array, "k *batch"]]:
    if self.alignment_type == "non_monotone_one_to_one":
      raise NotImplementedError(
          "Non-monotone alignment doesn't support top-k.")
    else:
      return self._dist.top_k(k, **kwargs)

  @typed
  def entropy(self, **kwargs) -> Float[Array, "*batch"]:
    if self.alignment_type == "non_monotone_one_to_one":
      raise NotImplementedError(
          "Non-monotone alignment doesn't support entropy.")
    else:
      return self._dist.entropy(**kwargs)

  @typed
  def cross_entropy(self, other: AlignmentCRF, **kwargs
                    ) -> Float[Array, "*batch"]:
    # pylint: disable=protected-access
    if self.alignment_type == "non_monotone_one_to_one":
      raise NotImplementedError(
          "Non-monotone alignment doesn't support cross-entropy.")
    else:
      return self._dist.cross_entropy(other._dist, **kwargs)

  @typed
  def kl_divergence(self, other: AlignmentCRF, **kwargs
                    ) -> Float[Array, "*batch"]:
    # pylint: disable=protected-access
    if self.alignment_type == "non_monotone_one_to_one":
      raise NotImplementedError(
          "Non-monotone alignment doesn't support KL divergence.")
    else:
      return self._dist.kl_divergence(other._dist, **kwargs)
