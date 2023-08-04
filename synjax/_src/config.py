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

"""Manages global SynJax configuration."""
import contextlib
import dataclasses
import functools
from typing import Literal
from synjax._src.typing import typed  # pylint: disable=g-importing-member


@typed
@functools.partial(dataclasses.dataclass, frozen=True)
class SynJaxConfig():
  """SynJax configuration."""

  use_strict_max: bool = False
  checkpoint_loops: bool = True
  checkpoint_semiring_einsum: bool = True
  # Matrix-Tree Theorem settings
  mtt_shift_log_potentials: bool = True
  mtt_logdet_method: Literal["lu", "qr"] = "lu"
  mtt_inv_method: Literal["solve", "qr"] = "solve"
  mtt_inv_matmul_precision: Literal["default", "high", "highest"] = "default"
  # CTC settings
  ctc_use_optax: bool = False
  # Projective Spanning Trees settings
  projective_argmax_algorithm: Literal["Kuhlmann", "Eisner"] = "Kuhlmann"
  # Linear-Chain CRF settings
  linear_chain_crf_forward_algorithm: Literal["sequential", "parallel"] = (
      "sequential")


_config = SynJaxConfig()


def get_config() -> SynJaxConfig:
  return _config


def set_config(**settings) -> None:
  global _config
  _config = dataclasses.replace(_config, **settings)


@contextlib.contextmanager
def config_context(**settings):
  prior_settings = dataclasses.asdict(get_config())
  set_config(**settings)
  yield get_config()
  set_config(**prior_settings)
