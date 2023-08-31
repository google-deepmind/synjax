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

"""Distribution representing linear chain CRF."""
# pylint: disable=g-multiple-import, g-importing-member
import math
from typing import Literal, Optional
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int32
from synjax._src.config import get_config
from synjax._src.constants import INF
from synjax._src.distribution import SemiringDistribution
from synjax._src.typing import Key, typed
from synjax._src.utils.semirings import Semiring


@typed
class LinearChainCRF(SemiringDistribution):
  """Distribution representing linear chain CRF.

  References:
    Lafferty et al, 2001: https://repository.upenn.edu/cgi/viewcontent.cgi?article=1162&context=cis_papers
    Sutton and McCallum, 2012: https://homepages.inf.ed.ac.uk/csutton/publications/crftutv2.pdf
    Collins notes 2011 -- http://www.cs.columbia.edu/~mcollins/crf.pdf
  """  # pylint: disable=line-too-long

  log_potentials: Float[Array, "*batch n t t"]
  lengths: Int32[Array, "*batch"]

  def __init__(self, log_potentials: Float[Array, "*batch n t t"],
               lengths: Optional[Int32[Array, "*batch"]] = None, **kwargs):
    """Linear Chain CRFs for a sequence of length n with t states.

    References:
      Lafferty et al, 2001: https://repository.upenn.edu/cgi/viewcontent.cgi?article=1162&context=cis_papers
      Sutton and McCallum, 2012: https://homepages.inf.ed.ac.uk/csutton/publications/crftutv2.pdf
      Collins notes 2011 -- http://www.cs.columbia.edu/~mcollins/crf.pdf
    Args:
      log_potentials:
        For a sentence of n words log_potentials will have shape
        (..., n, t, t). The entry of log_potentials[i, t1, t2] represents
        the log_potential of an edge (i-1, t1) -> (i, t2). In other words
        log_potentials[i] are representing edges entering word at position i.
        Zero-th transition matrix shows transitions from initial state
        (non-word state) into the first word at position 0. This means that in
        the 0th transition matrix all rows except for the 0th one are ignored.
      lengths:
        Lengths of each entry in the batch. It has the same shape as the
        batch and dtype of jnp.int32. If it's not passed, the maximal length
        will be assumed based on the log_potentials.shape[-3].
      **kwargs: Additional optional args to pass to superclass constructors.
    """
    super().__init__(log_potentials=log_potentials,
                     **(dict(struct_is_isomorphic_to_params=True) | kwargs))
    if lengths is None:
      lengths = jnp.full(self.batch_shape, self.event_shape[0])
    self.lengths = lengths

  @property
  def event_shape(self):
    return self.log_potentials.shape[-3:]

  @property
  def _typical_number_of_parts_per_event(self) -> Int32[Array, "*batch"]:
    return self.lengths

  @typed
  def _structure_forward(
      self, base_struct: Float[Array, "n t t"], semiring: Semiring, key: Key,
      forward_algorithm: Optional[Literal["sequential", "parallel"]] = None
      ) -> Float[Array, "s"]:
    if forward_algorithm is None:
      forward_algorithm = get_config().linear_chain_crf_forward_algorithm

    if forward_algorithm == "sequential":
      return self._structure_forward_sequential(base_struct, semiring, key)
    elif forward_algorithm == "parallel":
      return self._structure_forward_parallel(base_struct, semiring, key)
    else:
      raise NotImplementedError

  @typed
  def _structure_forward_sequential(
      self, base_struct: Float[Array, "n t t"], semiring: Semiring,
      key: Key) -> Float[Array, "s"]:
    """Forward algorithm with complexity O(n t^2)."""
    base_struct = base_struct.at[0, 1:].set(-INF)
    n, t = self.log_potentials.shape[-3:-1]
    def loop(state, inp):
      matrix, key = inp
      state = semiring.einsum("si,sij->sj", state, matrix, key=key)
      return state, state
    if get_config().checkpoint_loops:
      loop = jax.checkpoint(loop)
    keys = jax.random.split(key, n+1)  # (n+1, 2)
    seq = semiring.wrap(base_struct+self.log_potentials)  # (s, n, t, t)
    seq = jnp.swapaxes(seq, 0, 1)  # (n, s, t, t)
    state = semiring.one(t)
    _, states = jax.lax.scan(loop, state, (seq, keys[:-1]))
    state = states[self.lengths-1]
    return semiring.sum(state, axis=-1, key=keys[-1])

  @typed
  def _structure_forward_parallel(
      self, base_struct: Float[Array, "n t t"], semiring: Semiring,
      key: Key) -> Float[Array, "s"]:
    """Forward algorithm with parallel complexity O(log(n) t^3).

    This is inspired by the algorithm of Hassan et al (2021) and used by
    Rush (2020). This algorithm reduces parallel time with respect to length
    dimension n but it increases it with respect to t. It additionally increases
    sequential complexity to O(n log(n) t^3). In most cases sequential algorithm
    (the default one) should be faster, but in some extreme lengths parallel
    algorithm may be more numerically stable -- for the same reason as pairwise
    summation can be more accurate than sequential summation (Higham, 1993).

    References:
      Rush, 2020 - Section 6a: https://arxiv.org/pdf/2002.00876.pdf
      Hassan et al, 2021: https://arxiv.org/pdf/2102.05743.pdf
      Higham, 1993: https://doi.org/10.1137/0914050

    Args:
      base_struct: Dummy structure that will be used to track gradients.
      semiring: Semiring used for the computation.
      key: Key that will be used if semiring is a sampling semiring.
    Returns:
      Log-partition under a given semiring.
    """
    base_struct = base_struct.at[0, 1:].set(-INF)
    real_n, t = self.log_potentials.shape[-3: -1]
    log_n = math.ceil(math.log(real_n, 2))

    extension_shape = (int(2**log_n)-real_n, t, t)
    base_struct_extended = jnp.concatenate(
        (base_struct, jnp.zeros(extension_shape)), axis=-3)

    base_struct_extended = _mask_out_base_struct(base_struct_extended,
                                                 self.lengths)
    log_potentials = jnp.concatenate(
        (self.log_potentials, jnp.zeros(extension_shape)), axis=-3)

    seq = semiring.wrap(base_struct_extended+log_potentials)  # (s, real_n, t, t)

    keys = jax.random.split(key, log_n+1)
    def loop(aseq, akey):
      left = aseq[:, 0::2, :, :]
      right = aseq[:, 1::2, :, :]
      return semiring.einsum("snij,snjk->snik", left, right, key=akey)
    if get_config().checkpoint_loops:
      loop = jax.checkpoint(loop)
    for step in range(log_n):
      seq = loop(seq, keys[step])
    matrix = seq.squeeze(-3)
    return semiring.sum(matrix, axis=(-1, -2), key=keys[-1])


@typed
def _mask_out_base_struct(base_struct: Float[Array, "*batch n t t"],
                          lengths: Int32[Array, "*batch"]
                          ) -> Float[Array, "*batch n t t"]:
  """Masks-out parts of base-struct that don't fit seqence length."""
  n, t = base_struct.shape[-3:-1]

  padding_mask = jnp.arange(0, n) >= lengths[..., None]
  padding_mask = jnp.broadcast_to(padding_mask[..., None, None],
                                  (*padding_mask.shape, t, t))
  padding_mask = padding_mask != padding_mask.at[..., 0].set(False)

  potentials_mask = jnp.arange(0, n) < lengths[..., None]
  potentials_mask = jnp.broadcast_to(potentials_mask[..., None, None],
                                     (*potentials_mask.shape, t, t))

  return jnp.where(potentials_mask,
                   base_struct, jnp.where(padding_mask, 0, -INF))
