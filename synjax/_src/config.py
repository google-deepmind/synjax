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
import dataclasses
import functools


@functools.partial(dataclasses.dataclass, frozen=True)
class SynJaxConfig():

  use_strict_max: bool = False
  checkpoint_loops: bool = True
  checkpoint_semiring_einsum: bool = True
  # Matrix-Tree Theorem settings
  mtt_shift_log_potentials: bool = True
  mtt_logdet_method: str = "qr"
  mtt_inv_method: str = "qr"
  mtt_inv_matmul_precision: str = "highest"


_config = SynJaxConfig()


def get_config() -> SynJaxConfig:
  return _config


def set_config(**kwargs) -> None:
  global _config
  _config = dataclasses.replace(_config, **kwargs)
