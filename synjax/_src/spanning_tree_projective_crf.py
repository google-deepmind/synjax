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
from typing import Literal, Optional, Tuple
import warnings

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int32
from synjax._src.config import get_config
from synjax._src.constants import INF
from synjax._src.deptree_algorithms import deptree_utils
from synjax._src.distribution import SemiringDistribution
from synjax._src.typing import Shape, Key, typed
from synjax._src.utils import chart_struct
from synjax._src.utils.semirings import Semiring, MaxSemiring


Chart = chart_struct.Chart
AlgorithmName = Literal["Kuhlmann", "Eisner"]


class SpanningTreeProjectiveCRF(SemiringDistribution):
  """Distribution representing projective dependency trees."""

  single_root: bool = eqx.static_field()
  lengths: Int32[Array, "*batch"]

  @typed
  def __init__(self, log_potentials: Float[Array, "*batch n n"],
               *,
               single_root: bool,
               lengths: Optional[Int32[Array, "*batch"]] = None):
    self.single_root = single_root
    if lengths is None:
      lengths = jnp.full(log_potentials.shape[:-2], log_potentials.shape[-1])
    self.lengths = lengths
    super().__init__(log_potentials=deptree_utils.pad_log_potentials(
        log_potentials, self.lengths))

  @property
  def event_shape(self) -> Shape:
    return self.log_potentials.shape[-2:]

  @typed
  def _structure_forward(
      self, base_struct: Float[Array, "n n"], semiring: Semiring, key: Key,
      algorithm: Optional[AlgorithmName] = None) -> Float[Array, "s"]:
    if (algorithm is None and isinstance(semiring, MaxSemiring)) or (
        algorithm == "Kuhlmann"):
      if not isinstance(semiring, MaxSemiring):
        warnings.warn(
            "Kuhlmann's arc-hybrid algorithm does not provide correct results"
            " for any semiring except MaxSemiring due to spurious ambiguity.")
      return self._structure_forward_Kuhlmann_arc_hybrid(
          base_struct, semiring, key)
    elif algorithm is None or algorithm == "Eisner":
      return self._structure_forward_Eisner(base_struct, semiring, key)
    else:
      raise ValueError(f"Unknown algorithm {algorithm}.")

  @typed
  def _structure_forward_Kuhlmann_arc_hybrid(  # pylint: disable=invalid-name
      self, base_struct: Float[Array, "n n"], semiring: Semiring, key: Key
      ) -> Float[Array, "s"]:
    """Kuhlmann et al (2011) arc-hybrid parsing algorithm.

    Fast in practice, but should be used only with MaxSemiring because it has
    multiiple derivations for the same tree causing the partition function
    computation of other semirings to be incorrect. Simple visual depiction of
    the algorithm is present in Shi et al (2017).

    References:
      Shi et al, 2017 - Figure 1b: https://aclanthology.org/D17-1002.pdf#page=5
      Kuhlmann et al, 2011: https://aclanthology.org/P11-1068.pdf
    Args:
      base_struct: Zero tensor for tracking gradients by being glued to the
                   structure of the computation.
      semiring: Used semiring.
      key: Random key.
    Returns:
      Partition function with the provided semiring.
    """
    # See Figure 1b in Shi et al 2017 for good illustration of the algorithm.
    n = self.log_potentials.shape[-1]-1  # Number of words excluding ROOT node.
    if self.single_root:
      # Apply Reweighting algorithm from Stanojević and Cohen 2021.
      # https://aclanthology.org/2021.emnlp-main.823.pdf
      lp = jnp.clip(self.log_potentials, -INF/100)
      c = jax.lax.stop_gradient(n*(jnp.max(lp) - jnp.min(lp))+1)
    else:
      c = 0

    params = base_struct+self.log_potentials.at[0].add(-c)
    params_extended = jnp.full((n+2, n+2), -INF).at[:n+1, :n+1].set(params)
    lr_arcs = chart_struct.from_cky_table(semiring.wrap(params_extended))
    rl_arcs = chart_struct.from_cky_table(semiring.wrap(params_extended.T))
    init_chart = chart_struct.from_cky_table(semiring.wrap(
        INF*(jnp.eye(n+2, k=1)-1)))
    keys = jax.random.split(key, 2*n)
    def loop(chart: chart_struct.Chart, d):
      lr = lr_arcs.left()
      rl = rl_arcs.right_non_empty(d, semiring)
      left = chart.left()
      right = chart.right_non_empty(d, semiring)
      score = semiring.add(lr, rl, key=keys[2*d])
      entries = semiring.einsum("sij,sij,sij->si", left, right, score,
                                key=keys[2*d+1])
      return chart.set_entries(d, entries), None
    if get_config().checkpoint_loops:
      loop = jax.checkpoint(loop)
    chart, _ = jax.lax.scan(loop, init_chart, jnp.arange(3, n+3))
    return chart.get_entries(n+2)[:, 0] + c

  @typed
  def _structure_forward_Eisner(  # pylint: disable=invalid-name
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
    params = base_struct+self.log_potentials
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

    if self.single_root:
      left = chart_right_comp.left()[:, 1, :-1]
      right = chart_left_comp.right_unmasked_non_empty(n-1)[:, 1, :-1]
      arcs = semiring.wrap(params[0, 1:])
      result = semiring.einsum("si,si,si->s", left, right, arcs, key=keys[n])
    else:
      result = chart_left_comp.get_entries(n)[:, 0]

    return result
