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

"""Distribution of monotone alignments between two sequences."""
# pylint: disable=g-multiple-import, g-importing-member
from typing import Optional, Tuple, List
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int32, PyTree
from synjax._src.config import get_config
from synjax._src.constants import INF
from synjax._src.distribution import SemiringDistribution
from synjax._src.typing import Shape, Key, typed
from synjax._src.utils import semirings


class GeneralMonotoneAlignmentCRF(SemiringDistribution):
  """Distribution of monotone alignments between elements of two sequences.

  It is similar to String-Edit Distance CRF from McCallum et al (2005),
  but more powerful in some aspects because it can represent alignments with
  bigger diagonal jumps that are needed for distributions like CTC in case of
  blank symbols.

  References:
    McCallum et al, 2005: https://people.cs.umass.edu/~mccallum/papers/crfstredit-uai05.pdf
  """  # pylint: disable=line-too-long

  log_potentials_horizontal: Tuple[Float[Array, "*batch row col"], ...]
  log_potentials_vertical: Optional[Float[Array, "*batch row col"]]
  lengths_rows: Int32[Array, "*batch"]
  lengths_cols: Int32[Array, "*batch"]

  @typed
  def __init__(
      self,
      log_potentials_horizontal: PyTree[Float[Array, "*batch row col"]],
      # NOTE The type of log_potentials_horizontal here is PyTree instead of
      # tuple because jaxtyping/typeguard sometimes fails in checking shapes
      # otherwise. Tuple type is check explicitly later with isinstance.
      log_potentials_vertical: Optional[Float[Array, "*batch row col"]], *,
      lengths_rows: Optional[Int32[Array, "*batch"]] = None,
      lengths_cols: Optional[Int32[Array, "*batch"]] = None):
    """Creates an AlignmentCRF distribution.

    Args:
      log_potentials_horizontal:
        Tuple of jax.Arrays that specifies the lot-potentials for making one
        horizontal move + i numbers of vertical moves where i is the position
        of the array in the tuple. For example, the array
        log_potentials_horizontal[0] specifies strictly horizontal moves
        (1 horizontal + 0 vertical), log_potentials_horizontal[1] specifies
        strictly diagonal moves (1 horizontal + 1 vertical), log_potentials[2]
        specifies even more tilted diagonal moves (1 horizontal + 2 vertical)
        etc. The number of arrays has to be at least 1, in case
        log_potentials_vertical is also provided, or at least 2 in case there is
        not log_potentials_vertical.
      log_potentials_vertical:
        Optional jax.Array that specifies the log-potentials for moving
        vertically in the alignment matrix.
      lengths_rows: Optional jax.Array with the number of rows in each instance.
      lengths_cols: Optional jax.Array with the number of columns in
                    each instance.
    """
    super().__init__(log_potentials=None, struct_is_isomorphic_to_params=False)
    if not isinstance(log_potentials_horizontal, tuple):
      raise ValueError("log_potentials_horizontal must be a tuple.")
    if len(log_potentials_horizontal)+(log_potentials_vertical is not None) < 2:
      # Explicit check needed here because jaxtyping checks fail sometimes.
      raise ValueError("Arguments log-potentials must have the same shape.")
    rows, cols = log_potentials_horizontal[0].shape[-2:]
    if (log_potentials_vertical is None and lengths_cols is None
        and lengths_rows is None and rows >= cols):
      raise ValueError("This is a useless distribution because there is "
                       "less than two alignment possible.")
    batch_shape = log_potentials_horizontal[0].shape[:-2]
    if lengths_rows is None:
      lengths_rows = jnp.full(batch_shape, rows)
    if lengths_cols is None:
      lengths_cols = jnp.full(batch_shape, cols)
    self.log_potentials_horizontal = log_potentials_horizontal
    self.log_potentials_vertical = log_potentials_vertical
    self.lengths_rows = lengths_rows
    self.lengths_cols = lengths_cols

  @property
  def batch_shape(self) -> Shape:
    return self.log_potentials_horizontal[0].shape[:-2]

  @property
  def event_shape(self) -> Shape:
    return self.log_potentials_horizontal[0].shape[-2:]

  @typed
  def unnormalized_log_prob(self, event: Float[Array, "*samples_batch row col"],
                            **kwargs) -> Float[Array, "*samples_batch"]:
    lp, lp_vert = self._masked_params()
    scores = jnp.zeros(event.shape[:-2])
    if lp_vert is not None:
      is_vertical = (event * jnp.roll(event, 1, -2)).at[..., 0, :].set(0)
      scores += jnp.sum(lp_vert * is_vertical, (-1, -2))
    event_leaving = (jnp.argmax(jnp.cumsum(event, -2), -2, keepdims=True)
                     == jnp.arange(event.shape[-2])[:, None]).astype(jnp.int32)
    event_leaving_shifted = jnp.roll(event_leaving, 1, -1).at[..., 0].set(0)
    event_entering = (jnp.argmax(event, -2, keepdims=True)
                      == jnp.arange(event.shape[-2])[:, None]).astype(jnp.int32)
    for i, lp_i in enumerate(lp):
      is_active = event_entering * jnp.roll(event_leaving_shifted, i, -2)
      scores += jnp.sum(lp_i * is_active, (-1, -2))
    return scores

  @typed
  def _structure_forward(
      self, base_struct: Float[Array, "row col"], semiring: semirings.Semiring,
      key: Key) -> Float[Array, "s"]:
    rows, cols = self.event_shape
    init_state = semiring.wrap(jnp.full(rows, -INF).at[0].set(0))
    keys = jax.random.split(key, cols*len(self.log_potentials_horizontal)
                            ).reshape(cols, -1, 2)

    def loop(state, inp):
      scores, scores_vert, keys = inp
      out_state = semiring.mul(state, scores[0])
      for shift, score in enumerate(scores[1:], 1):
        transition = semiring.mul(score, jnp.roll(state, shift=shift, axis=-1))
        out_state = semiring.add(out_state, transition, key=keys[shift])
      if scores_vert is not None:
        def vertical_loop(up, inp2):
          curr, weight, key = inp2
          curr = semiring.add(curr, semiring.mul(weight, up), key=key)
          return curr, curr
        subkeys = jax.random.split(keys[0], rows)
        out_state = jax.lax.scan(vertical_loop, semiring.zero(),
                                 (out_state.T, scores_vert.T, subkeys))[1].T
      return out_state, out_state
    if get_config().checkpoint_loops:
      loop = jax.checkpoint(loop)
    lp, lp_vert = self._masked_params()
    lp = [jnp.moveaxis(semiring.wrap(x+base_struct), -1, 0) for x in lp]
    if lp_vert is not None:
      lp_vert = jnp.moveaxis(semiring.wrap(lp_vert+base_struct), -1, 0)
    _, outputs = jax.lax.scan(loop, init_state, (lp, lp_vert, keys))
    return outputs[self.lengths_cols-1, :, self.lengths_rows-1]

  @typed
  def _masked_params(self) -> Tuple[List[Float[Array, "*batch row col"]],
                                    Optional[Float[Array, "*batch row col"]]]:
    rows, _ = self.event_shape
    lp_h, lp_v = self.log_potentials_horizontal, self.log_potentials_vertical
    lp_h = [x.at[..., 0].set(-INF * (jnp.arange(rows) > 0)) for x in lp_h]
    lp_h = [jnp.where(jnp.arange(rows)[:, None] >= i, x, -INF)
            for i, x in enumerate(lp_h)]
    if lp_v is not None:
      lp_v = lp_v.at[..., 0, :].set(-INF)
    return lp_h, lp_v

