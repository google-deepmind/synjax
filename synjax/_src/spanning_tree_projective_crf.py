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

"""Distribution representing projective dependency trees."""
# pylint: disable=g-multiple-import, g-importing-member
import dataclasses
from typing import Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int32
from synjax._src.config import get_config
from synjax._src.deptree_algorithms.deptree_padding import pad_log_potentials, directed_tree_mask
from synjax._src.distribution import SemiringDistribution
from synjax._src.typing import Shape, Key, typed
from synjax._src.utils import chart_struct
from synjax._src.utils.semirings import Semiring


Chart = chart_struct.Chart


class SpanningTreeProjectiveCRF(SemiringDistribution):
  """Distribution representing projective dependency trees."""

  single_root_edge: bool = eqx.static_field()
  lengths: Int32[Array, "*batch"]

  @typed
  def __init__(self, log_potentials: Float[Array, "*batch n n"],
               *,
               single_root_edge: bool,
               lengths: Optional[Int32[Array, "*batch"]] = None,
               **kwargs):
    super().__init__(log_potentials=log_potentials, **kwargs)
    self.single_root_edge = single_root_edge
    if lengths is None:
      lengths = jnp.full(log_potentials.shape[:-2], log_potentials.shape[-1])
    self.lengths = lengths

  @property
  def event_shape(self) -> Shape:
    return self.log_potentials.shape[-2:]

  @property
  def _typical_number_of_parts_per_event(self) -> Int32[Array, "*batch"]:
    return self.lengths-1

  @property
  def _padded_log_potentials(self) -> Float[Array, "*batch n n"]:
    return pad_log_potentials(self.log_potentials, self.lengths)

  def argmax_and_max(self, *args, **kwargs) -> Float[Array, "*batch n n"]:
    # Default argmax_and_max with an addition of removing padding.
    struct, score = super().argmax_and_max(*args, **kwargs)
    return struct * directed_tree_mask(struct.shape[-1], self.lengths), score

  def sample(self, *args, **kwargs):
    # Default sample with an addition of removing padding.
    struct = super().sample(*args, **kwargs)
    return struct * directed_tree_mask(struct.shape[-1], self.lengths)

  def marginals_for_template_variables(self, **kwargs):
    # Default marginals_for_template_vars with an addition of removing padding.
    marginal = super().marginals_for_template_variables(**kwargs).log_potentials
    mask = directed_tree_mask(self.event_shape[-1], self.lengths)
    return dataclasses.replace(self, log_potentials=mask * marginal)

  def marginals(self, *args, **kwargs):
    return self.marginals_for_template_variables(*args, **kwargs).log_potentials

  def top_k(self, *args, **kwargs):
    # Default top-k with an addition of removing padding.
    struct, score = super().top_k(*args, **kwargs)
    return struct * directed_tree_mask(struct.shape[-1], self.lengths), score

  @typed
  def _structure_forward(
      self, base_struct: Float[Array, "n n"], semiring: Semiring, key: Key
      ) -> Float[Array, "s"]:
    """Eisner's parsing algorithm.

    References:
      Eisner, 2000: https://www.cs.jhu.edu/~jason/papers/eisner.iwptbook00.pdf
      Chen et al, 2014 - slide 20: http://ir.hit.edu.cn/~lzh/papers/coling14-tutorial-dependency-parsing-a.pdf#page=20
    Args:
      base_struct: Zero tensor for tracking gradients by being glued to the
                   structure of the computation.
      semiring: Used semiring.
      key: Random key.
    Returns:
      Partition function with the provided semiring.
    """  # pylint: disable=line-too-long
    params = base_struct+self._padded_log_potentials
    lr_arcs = chart_struct.from_cky_table(semiring.wrap(params))
    rl_arcs = chart_struct.from_cky_table(semiring.wrap(params.T))
    chart_left_incomp = chart_struct.from_cky_table(
        semiring.one(self.event_shape))
    chart_right_incomp = chart_struct.from_cky_table(
        semiring.one(self.event_shape))
    chart_left_comp = chart_struct.from_cky_table(
        semiring.one(self.event_shape))
    chart_right_comp = chart_struct.from_cky_table(
        semiring.one(self.event_shape))

    n = base_struct.shape[-1]
    keys = jax.random.split(key, n+2)
    state = (chart_left_incomp, chart_right_incomp,
             chart_left_comp, chart_right_comp)
    def loop(state: Tuple[Chart, Chart, Chart, Chart], d):
      (chart_left_incomp, chart_right_incomp,
       chart_left_comp, chart_right_comp) = state
      akeys = jax.random.split(keys[d], 4)

      content = semiring.einsum("sij,sij->si",
                                chart_left_comp.left(),
                                chart_right_comp.right(d, semiring),
                                key=akeys[0])
      chart_left_incomp = chart_left_incomp.set_entries(
          d, semiring.mul(content, lr_arcs.get_entries(d)))
      chart_right_incomp = chart_right_incomp.set_entries(
          d, semiring.mul(content, rl_arcs.get_entries(d)))

      content = semiring.einsum("sij,sij->si",
                                chart_left_incomp.left_non_empty(),
                                chart_left_comp.right(d, semiring),
                                key=akeys[1])
      chart_left_comp = chart_left_comp.set_entries(d, content)

      content = semiring.einsum("sij,sij->si",
                                chart_right_comp.left(),
                                chart_right_incomp.right_non_empty(d, semiring),
                                key=akeys[2])
      chart_right_comp = chart_right_comp.set_entries(d, content)
      state = (chart_left_incomp, chart_right_incomp,
               chart_left_comp, chart_right_comp)
      return state, None
    if get_config().checkpoint_loops:
      loop = jax.checkpoint(loop)
    state, _ = jax.lax.scan(loop, state, jnp.arange(2, n+1))
    (chart_left_incomp, chart_right_incomp,
     chart_left_comp, chart_right_comp) = state
    del chart_left_incomp, chart_right_incomp  # These are not used later.

    if self.single_root_edge:
      left = chart_right_comp.left()[:, 1, :-1]
      right = chart_left_comp.right_unmasked_non_empty(n-1)[:, 1, :-1]
      arcs = semiring.wrap(params[0, 1:])
      result = semiring.einsum("si,si,si->s", left, right, arcs, key=keys[n])
    else:
      result = chart_left_comp.get_entries(n)[:, 0]

    return result
