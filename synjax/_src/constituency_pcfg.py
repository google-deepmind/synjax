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

"""Implementation of PCFG."""
# pylint: disable=g-multiple-import, g-importing-member
# pylint: disable=invalid-name
from typing import NamedTuple, Optional, Union
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int32
from synjax._src import distribution
from synjax._src.config import get_config
from synjax._src.typing import typed, Key, Shape
from synjax._src.utils import chart_struct
from synjax._src.utils.semirings import Semiring
from synjax._src.utils.special import roll


class Event(NamedTuple):

  chart: Union[Float[Array, "*batch n n nt"], Shape]
  tags: Union[Float[Array, "*batch n pt"], Shape]


class GeneralizedPCFG(distribution.SemiringDistribution):
  """Probabilistic Context-Free Grammar.

  Note that this is a conditional PCFG, i.e. it is a distribution over trees
  provided by PCFG conditioned by a provided sentence. Because of that calling
  dist.log_probability(tree) returns a p(tree | sentence; pcfg). To get a
  joint probability of a tree and a sentence p(tree, sentence ; pcfg) call
  dist.unnormalized_log_probability(tree). For a short description of
  normalization see Eisner (2016, S7.1), for long description see
  Nederhof and Satta (2003).

  References:
    Eisner 2016 - Section 7.1: https://aclanthology.org/W16-5901.pdf#page=7
    Nederhof and Satta 2003: https://aclanthology.org/W03-3016.pdf
  """

  preterminal_scores: Float[Array, "*batch n pt"]
  root: Float[Array, "*batch nt"]
  rule: Float[Array, "*batch nt nt+pt nt+pt"]
  lengths: Int32[Array, "*batch"]

  @typed
  def __init__(
      self, *,
      preterminal_scores: Float[Array, "*batch n pt"],
      root: Float[Array, "*batch nt"],
      rule: Float[Array, "*batch nt nt+pt nt+pt"],
      lengths: Optional[Int32[Array, "*batch"]] = None, **kwargs):
    super().__init__(log_potentials=None,
                     **(dict(struct_is_isomorphic_to_params=False) | kwargs))
    self.root = jax.nn.log_softmax(root, -1)
    self.rule = jax.nn.log_softmax(rule, (-1, -2))
    self.preterminal_scores = preterminal_scores

    if lengths is None:
      lengths = jnp.full(self.batch_shape, self.size_sentence)
    self.lengths = lengths

  @property
  def size_sentence(self) -> int:
    return self.preterminal_scores.shape[-2]

  @property
  def size_nonterminals(self) -> int:
    return self.rule.shape[-3]

  @property
  def size_preterminals(self) -> int:
    return self.rule.shape[-2] - self.rule.shape[-3]

  @property
  def event_shape(self) -> Event:
    chart_shape = self.size_sentence, self.size_sentence, self.size_nonterminals
    preterm_shape = self.size_sentence, self.size_preterminals
    return Event(chart_shape, preterm_shape)

  @property
  def batch_shape(self) -> Shape:
    return self.rule.shape[:-3]

  @property
  def _typical_number_of_parts_per_event(self) -> Int32[Array, "*batch"]:
    return 2*self.lengths-1

  @typed
  def _structure_forward(self, base_struct: Event, semiring: Semiring, key: Key
                         ) -> Float[Array, "s"]:
    base_chart, base_preterm = base_struct
    keys = jax.random.split(key, self.size_sentence+2)
    rule_x_y1_z1 = semiring.wrap(
        self.rule[:, self.size_nonterminals:, self.size_nonterminals:])
    rule_x_y1_z = semiring.wrap(
        self.rule[:, self.size_nonterminals:, :self.size_nonterminals])
    rule_x_y_z1 = semiring.wrap(
        self.rule[:, :self.size_nonterminals, self.size_nonterminals:])
    rule_x_y_z = semiring.wrap(
        self.rule[:, :self.size_nonterminals, :self.size_nonterminals])

    ##########################################################################
    ############################ SPAN SIZE 1 ### START #######################
    # This establishes the connection between the term entries and
    # the diagonal in the base_struct. This info is implicitly used by
    # grad to put gradients in two places.
    term = semiring.wrap(self.preterminal_scores + base_preterm)  # s n pt
    ##########################################################################

    # Here a chart is constructed by cutting of the useless parts of base_struct
    chart = chart_struct.from_cky_table(semiring.wrap(base_chart))

    ##########################################################################
    ############################ SPAN SIZE 2 ### START #######################
    # The binary rule that has only terminals on the RHS is used
    # if and only if span is of size 2.

    x = semiring.einsum("siy,siz,sxyz->six",
                        term, roll(term, -1, axis=1), rule_x_y1_z1, key=keys[2])

    chart = chart.set_entries(2, semiring.mul(chart.get_entries(2), x))
    ##########################################################################
    def loop(chart: chart_struct.Chart, d: Array):
      akey = jax.random.split(keys[d], 4)
      ############################### X -> Y Z #################################
      y = chart.left()                                       # S,N,N,NT
      z = chart.right(d, semiring, exclude_word_nodes=True)  # S,N,N,NT
      xc = semiring.einsum("sijy,sijz,sxyz->six", y, z, rule_x_y_z, key=akey[0])
      ############################### X -> Y1 Z  ###############################
      y1 = term                                         # S,N,PT
      z = roll(chart.get_entries(d-1), -1, axis=1)      # S,N,NT
      xb = semiring.einsum("siy,siz,sxyz->six", y1, z, rule_x_y1_z, key=akey[1])
      ############################### X -> Y  Z1 ###############################
      y = chart.get_entries(d-1)               # S,N,NT
      z1 = roll(term, -d+1, axis=1)            # S,N,PT
      xa = semiring.einsum("siy,siz,sxyz->six", y, z1, rule_x_y_z1, key=akey[2])

      ######################### combine all variations #########################
      x = semiring.add(xa, xb, xc, key=akey[3])
      return chart.set_entries(d, semiring.mul(chart.get_entries(d), x)), None
    if get_config().checkpoint_loops:
      loop = jax.checkpoint(loop)
    chart, _ = jax.lax.scan(loop, chart, jnp.arange(3, self.size_sentence+1))

    ############################ ROOT NODE ### START #######################
    lengths = self.lengths if self.lengths is not None else self.size_sentence
    x = chart.pick_length(lengths)         # S,NT
    root = semiring.wrap(self.root)        # S,NT
    x = semiring.sum(semiring.mul(x, root), key=keys[-1], axis=-1)  # S
    ############################ ROOT NODE ### END   #######################
    return x


class PCFG(GeneralizedPCFG):

  __doc__ = GeneralizedPCFG.__doc__

  word_ids: Int32[Array, "*batch n"]
  emission: Float[Array, "*batch pt voc"]

  @typed
  def __init__(self, emission: Float[Array, "*batch pt voc"],
               root: Float[Array, "*batch nt"],
               rule: Float[Array, "*batch nt nt+pt nt+pt"],
               word_ids: Int32[Array, "*batch n"],
               lengths: Optional[Int32[Array, "*batch"]] = None, **kwargs):
    self.word_ids = word_ids
    self.emission = emission
    emission = jax.nn.log_softmax(emission, -1)
    preterm_scores = jnp.take_along_axis(emission, word_ids[..., None, :], -1)
    preterm_scores = jnp.swapaxes(preterm_scores, -1, -2)
    super().__init__(root=root, rule=rule, lengths=lengths,
                     preterminal_scores=preterm_scores, **kwargs)
