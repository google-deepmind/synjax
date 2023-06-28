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

"""SynJax is a library for structured prediction."""
# pylint: disable=g-multiple-import
from synjax import semirings
from synjax import special
from synjax._src.alignment_crf import AlignmentCRF
from synjax._src.config import get_config, set_config
from synjax._src.constituency_pcfg import PCFG, GeneralizedPCFG
from synjax._src.constituency_tensor_decomposition_pcfg import TensorDecompositionPCFG, GeneralizedTensorDecompositionPCFG
from synjax._src.constituency_tree_crf import TreeCRF
from synjax._src.ctc import CTC
from synjax._src.distribution import Distribution, SemiringDistribution
from synjax._src.hmm import HMM
from synjax._src.linear_chain_crf import LinearChainCRF
from synjax._src.semi_markov_crf import SemiMarkovCRF
from synjax._src.spanning_tree_crf import SpanningTreeCRF


__version__ = "0.1.4"

__all__ = (
    "Distribution",
    "SemiringDistribution",
    "AlignmentCRF",
    "CTC",
    "SpanningTreeCRF",
    "LinearChainCRF",
    "SemiMarkovCRF",
    "HMM",
    "TreeCRF",
    "PCFG",
    "GeneralizedPCFG",
    "TensorDecompositionPCFG",
    "GeneralizedTensorDecompositionPCFG",
    "get_config",
    "set_config",
    "semirings",
    "special",
)


#  _________________________________________
# / Please don't use symbols in `_src` they \
# \ are not part of the SynJax public API. /
#  -----------------------------------------
#         \   ^__^
#          \  (oo)\_______
#             (__)\       )\/\
#                 ||----w |
#                 ||     ||
#
