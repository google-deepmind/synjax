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

"""Generally useful small functions."""

# pylint: disable=g-importing-member
# pylint: disable=g-multiple-import
from synjax._src.utils.special import inv, safe_slogdet, safe_log, sparsemax, safe_clip, safe_log_softmax, vmap_ndim, sample_one_hot, max_one_hot, log_comb, log_catalan, log_delannoy, straight_through_replace, zgr, zgr_binary, gumbel_rao


__all__ = [
    "safe_slogdet",
    "safe_log",
    "inv",
    "sparsemax",
    "safe_clip",
    "safe_log_softmax",
    "vmap_ndim",
    "sample_one_hot",
    "gumbel_rao",
    "zgr",
    "zgr_binary",
    "max_one_hot",
    "log_comb",
    "log_catalan",
    "log_delannoy",
    "straight_through_replace",
]
