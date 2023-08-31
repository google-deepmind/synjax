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

"""Implementation of Tree CRF that models constituency trees.

References:
  Stern et al 2017 -- https://aclanthology.org/P17-1076.pdf
"""
from typing import Optional
import jax
import jax.numpy as jnp
# pylint: disable=g-multiple-import, g-importing-member
from jaxtyping import Array, Float, Int32
from synjax._src.config import get_config
from synjax._src.distribution import SemiringDistribution
from synjax._src.typing import Shape, Key, typed
from synjax._src.utils import chart_struct
from synjax._src.utils.semirings import Semiring


class TreeCRF(SemiringDistribution):
  """Globally normalized istribution over binary constituency trees.

  The model structure is very similar to Stern et al (2017) except SynJax
  additionally supports properly normalizing the distribution.

  References:
    Stern et al 2017 -- https://aclanthology.org/P17-1076.pdf
  """

  log_potentials: Float[Array, "*batch n n label"]
  lengths: Int32[Array, "*batch"]

  @typed
  def __init__(self,
               log_potentials: Float[Array, "*batch n n label"],
               *,
               lengths: Optional[Int32[Array, "*batch"]] = None, **kwargs):
    super().__init__(log_potentials=log_potentials,
                     **(dict(struct_is_isomorphic_to_params=True) | kwargs))
    if lengths is None:
      *batch_shape, n = log_potentials.shape[:-2]
      lengths = jnp.full(batch_shape, n)
    self.lengths = lengths

  @property
  def event_shape(self) -> Shape:
    return self.log_potentials.shape[-3:]

  @property
  def _typical_number_of_parts_per_event(self) -> Int32[Array, "*batch"]:
    return 2*self.lengths-1

  @typed
  def _structure_forward(self, base_struct: Float[Array, "n n label"],
                         semiring: Semiring, key: Key) -> Float[Array, "s"]:
    n = self.event_shape[0]
    keys = jax.random.split(key, n+1)

    param = semiring.wrap(base_struct+self.log_potentials)
    chart = chart_struct.from_cky_table(semiring.sum(param, axis=3,
                                                     key=keys[1]))

    def loop(chart: chart_struct.Chart, d: Array):
      new = semiring.einsum("sij,sij,si->si", chart.left(),
                            chart.right(d, semiring), chart.get_entries(d),
                            key=keys[d])
      return chart.set_entries(d, new), None
    if get_config().checkpoint_loops:
      loop = jax.checkpoint(loop)
    chart, _ = jax.lax.scan(loop, chart, jnp.arange(2, n+1))
    return chart.pick_length(self.lengths)
