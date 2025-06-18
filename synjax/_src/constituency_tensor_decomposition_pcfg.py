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

"""Tensor-Decomposition PCFG based on Yang et al (2022) and Cohen et al 2013.

References:
  Yang et al, 2022: https://aclanthology.org/2022.naacl-main.353.pdf
  Cohen et al, 2013: https://aclanthology.org/N13-1052.pdf
"""
# pylint: disable=g-multiple-import, g-importing-member
# pylint: disable=invalid-name
import functools
from typing import NamedTuple, Optional, Union
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int32
from synjax._src.config import get_config
from synjax._src.constituency_tree_crf import TreeCRF
from synjax._src.distribution import SemiringDistribution
from synjax._src.typing import Shape, Key, typed
from synjax._src.utils import chart_struct
from synjax._src.utils.semirings import Semiring, LogSemiring
from synjax._src.utils.special import max_one_hot


class Event(NamedTuple):

  chart: Union[Float[Array, "*batch n n rank"], Shape]
  tags: Union[Float[Array, "*batch n pt"], Shape]


class GeneralizedTensorDecompositionPCFG(SemiringDistribution):
  """Tensor-Decomposition Probabilistic Context-Free Grammar.

  Cohen et al (2013) showed that PCFG with large number of non-terminals can be
  approximated using CPD tensor decomposition. Yang et al (2022) used this to
  do efficient grammar induction with large number of non-terminals and
  relatively small rank dimesnion. They avoid tensor-decomposition step by
  keeping all parameters always in the rank space and enforcing all decomposed
  rules to be normalized. This is the same as "unary trick" decompositon of
  n-ary rules from Stanojević and Sima'an (2015).

  Note that this is a conditional TD-PCFG, i.e. it is a distribution over trees
  provided by TD-PCFG conditioned by a provided sentence. Because of that
  calling dist.log_probability(tree) returns a p(tree | sentence; td-pcfg).
  To get a joint probability of a tree and a sentence
  p(tree, sentence ; td-pcfg) call dist.unnormalized_log_probability(tree).
  For a short description of normalization see Eisner (2016, S7.1), for long
  description see Nederhof and Satta (2003).

  References:
    Yang et al, 2022 - Section 4.2: https://aclanthology.org/2022.naacl-main.353.pdf
    Cohen et al, 2013 - Section 7.1: https://aclanthology.org/N13-1052.pdf#page=8
    Stanojević and Sima'an, 2015 - Section 2: https://aclanthology.org/D15-1005.pdf#page=3
    Eisner 2016 - Section 7.1: https://aclanthology.org/W16-5901.pdf#page=7
    Nederhof and Satta 2003: https://aclanthology.org/W03-3016.pdf
  """  # pylint: disable=line-too-long

  size_sentence: int = eqx.field(static=True)
  size_nonterminals: int = eqx.field(static=True)
  size_preterminals: int = eqx.field(static=True)
  size_rank: int = eqx.field(static=True)

  preterminal_scores: Float[Array, "*batch n pt"]
  root: Float[Array, "*batch nt"]
  nt_to_rank: Float[Array, "*batch nt rank"]
  rank_to_left_nt: Float[Array, "*batch rank nt+pt"]
  rank_to_right_nt: Float[Array, "*batch rank nt+pt"]

  lengths: Int32[Array, "*batch"]

  @typed
  def __init__(self,
               *,
               preterminal_scores: Float[Array, "*batch n pt"],
               root: Float[Array, "*batch nt"],
               nt_to_rank: Float[Array, "*batch nt rank"],
               rank_to_left_nt: Float[Array, "*batch rank nt+pt"],
               rank_to_right_nt: Float[Array, "*batch rank nt+pt"],
               lengths: Optional[Int32[Array, "*batch"]] = None, **kwargs):
    super().__init__(log_potentials=None,
                     **(dict(struct_is_isomorphic_to_params=False) | kwargs))
    normalize = functools.partial(jax.nn.log_softmax, axis=-1)
    self.preterminal_scores = preterminal_scores
    self.root = normalize(root)
    self.nt_to_rank = normalize(nt_to_rank)
    self.rank_to_left_nt = normalize(rank_to_left_nt)
    self.rank_to_right_nt = normalize(rank_to_right_nt)

    self.size_sentence = preterminal_scores.shape[-2]
    self.size_nonterminals = root.shape[-1]
    self.size_preterminals = rank_to_left_nt.shape[-1] - self.size_nonterminals
    self.size_rank = nt_to_rank.shape[-1]

    if lengths is None:
      lengths = jnp.full(preterminal_scores.shape[:-2], self.size_sentence)
    self.lengths = lengths

  @property
  def event_shape(self) -> Event:
    chart_shape = self.size_sentence, self.size_sentence, self.size_rank
    preterm_shape = self.size_sentence, self.size_preterminals
    return Event(chart_shape, preterm_shape)

  @property
  def batch_shape(self) -> Shape:
    return self.root.shape[:-1]

  @property
  def _typical_number_of_parts_per_event(self) -> Int32[Array, "*batch"]:
    return 2*self.lengths-1

  @typed
  def _structure_forward(self, base_struct: Event, semiring: Semiring, key: Key
                         ) -> Float[Array, "s"]:
    base_chart, base_preterm = base_struct
    sr = semiring  # Simple renaming because semiring is used frequently here.
    if not isinstance(sr, LogSemiring):
      raise NotImplementedError("This distribution supports only LogSemiring.")
    n = self.size_sentence
    nt = self.size_nonterminals

    # These rules go bottom-up because that is the way CKY parsing works.
    left_unary = sr.einsum("srx,sxf->srf",   # s rank_binary(r)->rank_left(f)
                           sr.wrap(self.rank_to_left_nt[:, :nt]),
                           sr.wrap(self.nt_to_rank))
    right_unary = sr.einsum("srx,sxf->srf",  # s rank_binary->rank_right
                            sr.wrap(self.rank_to_right_nt[:, :nt]),
                            sr.wrap(self.nt_to_rank))
    root_unary = sr.einsum("sx,sxr->sr",     # s rank_of_binary_root_node
                           sr.wrap(self.root),
                           sr.wrap(self.nt_to_rank))

    base_chart = chart_struct.from_cky_table(sr.wrap(base_chart))
    left_chart = chart_struct.from_cky_table(sr.one((n, n, self.size_rank)))
    right_chart = chart_struct.from_cky_table(sr.one((n, n, self.size_rank)))

    keys = jax.random.split(key, 3*n+3)

    # Span size 1
    preterminal_scores = sr.wrap(self.preterminal_scores + base_preterm)
    left_chart = left_chart.set_entries(
        1, sr.einsum("snp,srp->snr",
                     preterminal_scores, sr.wrap(self.rank_to_left_nt[:, nt:]),
                     key=keys[2]))
    right_chart = right_chart.set_entries(
        1, sr.einsum("snp,srp->snr",
                     preterminal_scores, sr.wrap(self.rank_to_right_nt[:, nt:]),
                     key=keys[3]))

    def loop(state, d):
      left_chart, right_chart = state
      rank_state = sr.einsum("sir,sijr,sijr->sir",
                             base_chart.get_entries(d),
                             left_chart.left(),
                             right_chart.right(d, sr), key=keys[3*d])

      left_chart = left_chart.set_entries(d, sr.einsum(
          "sfr,sir->sif", left_unary, rank_state, key=keys[3*d+1]))
      right_chart = right_chart.set_entries(d, sr.einsum(
          "sfr,sir->sif", right_unary, rank_state, key=keys[3*d+2]))
      return (left_chart, right_chart), rank_state[:, 0]
    if get_config().checkpoint_loops:
      loop = jax.checkpoint(loop)
    _, rank_states = jax.lax.scan(loop, (left_chart, right_chart),
                                  jnp.arange(2, n+1))
    rank_state = rank_states[self.lengths-2]  # s r
    return sr.einsum("sr,sr->s", rank_state, root_unary, key=keys[0])

  @typed
  def mbr(self, *, marginalize_labels: bool, **kwargs) -> Event:
    """Minimum-Bayes Risk decoding.

    Args:
      marginalize_labels: Flag that controls if metric that is used by MBR is
      labelled or unlabled span recall, whereby labels are ranks,
      not non-terminals.
      **kwargs: Other optional kwargs that will be used by TreeCRF for decoding.
    Returns:
      The decoded structure. If marginalize_labels is the last axis, the one
      reserved for rank, will be of size 1.
    """
    chart_log_marginals, preterm_log_marginals = self.log_marginals()
    chart_log_marginals *= 1 - jnp.eye(self.size_sentence)[:, :, None]
    if marginalize_labels:
      chart_log_marginals = jax.nn.logsumexp(
          chart_log_marginals, axis=-1, keepdims=True)
    tree = TreeCRF(chart_log_marginals, lengths=self.lengths).argmax(**kwargs)
    tree = jnp.where(jnp.eye(self.size_sentence)[:, :, None], 0, tree)
    return Event(tree, max_one_hot(preterm_log_marginals, -1))


class TensorDecompositionPCFG(GeneralizedTensorDecompositionPCFG):

  __doc__ = GeneralizedTensorDecompositionPCFG.__doc__

  word_ids: Int32[Array, "*batch n"]
  emission: Float[Array, "*batch pt voc"]

  @typed
  def __init__(self, emission: Float[Array, "*batch pt voc"],
               root: Float[Array, "*batch nt"],
               nt_to_rank: Float[Array, "*batch nt rank"],
               rank_to_left_nt: Float[Array, "*batch rank nt+pt"],
               rank_to_right_nt: Float[Array, "*batch rank nt+pt"],
               word_ids: Int32[Array, "*batch n"],
               lengths: Optional[Int32[Array, "*batch"]] = None, **kwargs):
    """Constructs standard version of Tensor-Decomposition PCFG."""
    self.word_ids = word_ids
    self.emission = emission
    emission = jax.nn.log_softmax(emission, -1)
    preterm_scores = jnp.take_along_axis(emission, word_ids[..., None, :], -1)
    preterm_scores = jnp.swapaxes(preterm_scores, -1, -2)
    super().__init__(
        root=root, nt_to_rank=nt_to_rank, rank_to_left_nt=rank_to_left_nt,
        rank_to_right_nt=rank_to_right_nt, lengths=lengths,
        preterminal_scores=preterm_scores, **kwargs)
