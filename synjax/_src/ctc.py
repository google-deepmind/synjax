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

"""Distribution over CTC alignments."""
from typing import Optional
import jax
import jax.numpy as jnp
# pylint: disable=g-multiple-import, g-importing-member
from jaxtyping import Array, Float, Int32, Num
from synjax._src.alignment_monotone_general import GeneralMonotoneAlignmentCRF
from synjax._src.config import get_config
from synjax._src.constants import INF
from synjax._src.distribution import SemiringDistribution
from synjax._src.typing import Key, Shape, typed
from synjax._src.utils.semirings import Semiring


class CTC(SemiringDistribution):
  """Distribution over CTC alignments.

  References:
    Graves et al, 2006: https://www.cs.toronto.edu/~graves/icml_2006.pdf
    Hannun, 2017: https://distill.pub/2017/ctc/
  """

  log_potentials: Float[Array, "*batch n vocab"]
  labels_extended: Int32[Array, "*batch 2*labels+1"]
  label_lengths: Int32[Array, "*batch"]
  input_lengths: Int32[Array, "*batch"]

  @typed
  def __init__(
      self, log_potentials: Float[Array, "*batch n vocab"],
      labels: Int32[Array, "*batch labels"], *,
      label_lengths: Optional[Int32[Array, "*batch"]] = None,
      input_lengths: Optional[Int32[Array, "*batch"]] = None,
      blank_id: int = 0):
    super().__init__(log_potentials=log_potentials,
                     struct_is_isomorphic_to_params=False)

    # Inserts blank_id as first symbol, last symbol, and in between all words.
    self.labels_extended = jnp.full(
        labels.shape[:-1]+(labels.shape[-1]*2+1,), blank_id
        ).at[..., 1::2].set(labels.astype(jnp.int32))
    self.log_potentials = jax.nn.log_softmax(log_potentials, axis=-1)

    *batch_shape, cols, _ = self.log_potentials.shape
    rows = self.labels_extended.shape[-1]

    if label_lengths is None:
      self.label_lengths = jnp.full(batch_shape, rows)
    else:
      self.label_lengths = 2*label_lengths+1

    if input_lengths is None:
      self.input_lengths = jnp.full(batch_shape, cols+2)
    else:
      self.input_lengths = input_lengths+2

  @property
  def batch_shape(self) -> Shape:
    return self.log_potentials.shape[:-2]

  @property
  def event_shape(self) -> Shape:
    return self.labels_extended.shape[-1], self.log_potentials.shape[-2]

  @typed
  def _structure_forward(
      self, base_struct: Float[Array, "labels n"], semiring: Semiring,
      key: Key) -> Float[Array, "s"]:
    labels_extended = self.labels_extended
    voc = self.log_potentials.shape[-1]

    table = jnp.einsum("...nv,...lv->...ln", self.log_potentials,
                       jax.nn.one_hot(labels_extended, voc))
    table += base_struct
    # Insert one extra column in beginning and end to account for
    # the possibility of two beginning and two ending states (see Distill blog).
    extra_col = jnp.zeros(table.shape[:-1]+(1,))
    table = jnp.concatenate((extra_col, table, extra_col), axis=-1)
    table = table.at[1:, 0].set(-INF)  # first artificial state
    table = table.at[2:, 1].set(-INF)  # first two states are valid

    step_0 = step_1 = table

    non_repetitions = labels_extended != jnp.roll(labels_extended, 2, axis=-1)
    step_2 = jnp.where(non_repetitions[..., None], table, -INF)
    step_2 = step_2.at[..., 1].set(-INF)
    step_2 = jnp.where(jnp.arange(step_2.shape[-1]) == self.input_lengths-1,
                       -INF, step_2)

    dist = GeneralMonotoneAlignmentCRF(
        (step_0, step_1, step_2), None,
        lengths_rows=self.label_lengths, lengths_cols=self.input_lengths)
    # pylint: disable=protected-access
    return dist._structure_forward(jnp.zeros(dist.event_shape), semiring, key)

  @typed
  def log_partition(self, use_optax: Optional[bool] = None
                    ) -> Float[Array, "*batch"]:
    if use_optax is None:
      use_optax = get_config().ctc_use_optax
    if use_optax:
      n = self.log_potentials.shape[-2]
      l = self.labels_extended.shape[-1] // 2
      logit_paddings = jnp.arange(n) >= self.input_lengths[:, None]
      label_paddings = jnp.arange(l) >= (self.label_lengths[:, None]//2)
      labels = self.labels_extended[..., 1::2]
      logits = self.log_potentials
      # pylint: disable=g-import-not-at-top
      # pylint: disable=import-outside-toplevel
      import optax
      return -optax.ctc_loss(logits, logit_paddings, labels, label_paddings)
    else:
      return super().log_partition()

  @typed
  def marginals_for_template_variables(self, **kwargs) -> "CTC":
    # This override is needed because Optax internally does normalization.
    return super().marginals_for_template_variables(use_optax=False)

  @typed
  def log_count(self) -> Float[Array, "*batch"]:
    """Log of the count of structures in the support."""
    # This override is needed because Optax internally does normalization.
    return super().log_count(use_optax=False)

  @typed
  def loss(self, use_optax: Optional[bool] = None) -> Float[Array, "*batch"]:
    return -self.log_partition(use_optax=use_optax)

  @typed
  def alignment_to_labels(self, alignment: Num[Array, "*batch labels n"]
                          ) -> Num[Array, "*batch n"]:
    return jnp.einsum("...ln,...l->...ln", alignment, self.labels_extended
                      ).max(-2).astype(int)

  @typed
  def log_prob_labels(self, labels: Int32[Array, "*batch n"]
                      ) -> Float[Array, "*batch"]:
    n, voc = self.log_potentials.shape[-2:]
    scores_per_col = jnp.einsum(
        "...nv,...nv->...n", jax.nn.one_hot(labels, voc), self.log_potentials)
    mask = jnp.arange(n) < self.input_lengths[..., None]
    return jnp.sum(scores_per_col*mask, axis=-1)
