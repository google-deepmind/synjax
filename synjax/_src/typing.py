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

"""Common types used in SynJax."""
from typing import Tuple
import jaxtyping
import typeguard


Shape = Tuple[int, ...]
Key = jaxtyping.UInt32[jaxtyping.Array, "2"]
typed = lambda fn: jaxtyping.jaxtyped(typeguard.typechecked(fn))
