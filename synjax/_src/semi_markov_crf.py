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

"""Distribution representing Semi-Markov CRF for linear chains."""
from typing import Optional
import jax
import jax.numpy as jnp
# pylint: disable=g-multiple-import, g-importing-member
from jaxtyping import Array, Float, Int32, Num
from synjax._src.config import get_config
from synjax._src.constants import INF
from synjax._src.distribution import SemiringDistribution
from synjax._src.typing import Shape, Key, typed
from synjax._src.utils import special
from synjax._src.utils.semirings import Semiring


class SemiMarkovCRF(SemiringDistribution):
  """Distribution representing semi-Markov CRFs.

  Semi-Markov CRF was defined by Sarawagi and Cohen (2004). Similar model was
  used in speech recognition under the name of Segmental CRF
  (Abdel-Hamid et al, 2013; Lu et al, 2016). The main difference is that
  Segmental CRF predicts label independently of each other which is a special
  case of Semi-Markov CRF.

  References:
    Sarawagi and Cohen, 2004: https://proceedings.neurips.cc/paper/2004/file/eb06b9db06012a7a4179b8f3cb5384d3-Paper.pdf
    Abdel-Hamid et al, 2013: https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/SegmentalNN.pdf
    Lu et al, 2016: https://www.isca-speech.org/archive_v0/Interspeech_2016/pdfs/0040.PDF
  """  # pylint: disable=line-too-long

  log_potentials: Float[Array, "*batch n skip state state"]
  lengths: Int32[Array, "*batch"]

  @typed
  def __init__(self, log_potentials: Float[Array, "*batch n skip state state"],
               *, lengths: Optional[Int32[Array, "*batch"]] = None):
    """Constructs Semi-Markov CRF distribution.

    References:
      Sarawagi and Cohen, 2004: https://proceedings.neurips.cc/paper/2004/file/eb06b9db06012a7a4179b8f3cb5384d3-Paper.pdf
      Abdel-Hamid et al, 2013: https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/SegmentalNN.pdf
      Lu et al, 2016: https://www.isca-speech.org/archive_v0/Interspeech_2016/pdfs/0040.PDF
    Args:
      log_potentials:
        For a sentence of n words log_potentials will have shape
        (..., n, m, t, t). The entry of log_potentials[i, j, t1, t2] represents
        the log_potential of an edge (i-j, t1) -> (i, t2). In other words
        log_potentials[i, j] are representing edges entering word at position i
        by jumping from word at position i-j-1.
        Zero-th transition matrix shows transitions from initial state
        (non-word state) into the first word at position 0.
      lengths:
        Lengths of each entry in the batch. It has the same shape as the batch
        and dtype of jnp.int32. If it's not passed, the maximal length will be
        assumed based on the log_potentials.shape[-4].
    """  # pylint: disable=line-too-long
    super().__init__(log_potentials=log_potentials)
    if lengths is None:
      lengths = jnp.full(self.batch_shape, self.event_shape[0])
    self.lengths = lengths

  @property
  def event_shape(self) -> Shape:
    return self.log_potentials.shape[-4:]

  @typed
  def _structure_forward(
      self, base_struct: Float[Array, "n skip state state"],
      semiring: Semiring, key: Key) -> Float[Array, "s"]:
    """Forward algorithm with complexity O(n m t^2)."""
    n, m, t = self.log_potentials.shape[-4:-1]
    def loop(state, inp):
      transitions, key = inp
      out = semiring.einsum("smi,smij->sj", state, transitions, key=key)
      state = jnp.roll(state, 1, axis=-2).at[:, 0].set(out)
      return state, out
    if get_config().checkpoint_loops:
      loop = jax.checkpoint(loop)
    keys = jax.random.split(key, n+1)  # (n+1, 2)
    seq = semiring.wrap(base_struct + self.log_potentials)  # (s, n, m, t, t)
    seq = jnp.swapaxes(seq, 0, 1)  # (n, s, m, t, t)
    state = semiring.wrap(jnp.full((m, t), -INF).at[0, 0].set(0))
    _, states = jax.lax.scan(loop, state, (seq, keys[:-1]))
    state = states[self.lengths-1]
    return semiring.sum(state, axis=-1, key=keys[-1])

  @classmethod
  @typed
  def convert_sample_to_element_labels(
      cls, sample: Num[Array, "*xs n skip label label"]
      ) -> Num[Array, "*xs n label"]:
    """Converts samples from standard edge shape to a sequence of labels.

    Args:
      sample: Array of shape (..., n, m, t) where n is the sequence length,
      m is the skip size and t is the label.
    Returns:
      Array of shape (..., n, t) where each element (..., a, b) is 1 if
      sample has an arc covering position a with label b, otherwise it is 0.
    """
    # This function is exposed as a classmethod in order for it to be accessible
    # from the public interface of SynJax.
    n, m = sample.shape[-4:-2]
    labels = sample.sum(-2)  # (..., n, m, t)
    labels2 = jax.lax.cumsum(labels, -2 % labels.ndim, reverse=True)
    def roll_and_mask(x, step):
      mask = jnp.arange(n)[:, None] < n-step
      return jnp.where(mask, special.roll(x, -step, -2), 0)
    labels3 = jax.vmap(roll_and_mask, in_axes=(-2, 0), out_axes=-2
                      )(labels2, jnp.arange(m))
    labels4 = labels3.sum(-2)  # (..., n, t)
    labels4.sum(-1)
    return labels4
