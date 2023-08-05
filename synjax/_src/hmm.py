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

"""Distribution representing Hidden Markov Models."""
from typing import Optional, Union
import jax
import jax.numpy as jnp
# pylint: disable=g-multiple-import, g-importing-member
from jaxtyping import Array, Float, Int32
from synjax._src import linear_chain_crf
from synjax._src.typing import typed


@typed
def _expand_simple_sequence_model(
    init_score: Float[Array, "*batch state"],
    transition_score: Float[Array, "*batch state state"],
    emission_score: Float[Array, "*batch n state"]
    ) -> Float[Array, "*batch n state state"]:
  """Takes simple HMM-like input and expands it into generalized input for CRF.

  Args:
    init_score: Array of shape (..., t) where t is the number of states,
    transition_score: Array of shape (..., t, t),
    emission_score:
      Array of shape (..., n, t) where n is the length of sequence.
  Returns:
    Expanded Array of shape (..., n, t, t)
  """
  scores = transition_score[..., None, :, :] + emission_score[..., None, :]
  scores = scores.at[..., 0, 0, :].set(init_score + emission_score[..., 0, :])
  return scores


# pylint: disable=invalid-name
@typed
def HMM(init_logits: Float[Array, "*batch state"],
        transition_logits: Float[Array, "*batch state state"],
        emission_dist,
        observations: Union[Int32[Array, "*batch n"],
                            Float[Array, "*batch n d"]],
        *,
        lengths: Optional[Int32[Array, "*batch"]] = None
        ) -> linear_chain_crf.LinearChainCRF:
  """Builds HMM distribution with t states over n observations.

  Note that this is a conditional HMM, i.e. it is a distribution over state
  sequences provided by HMM conditioned by a provided input observations.
  Because of that calling dist.log_probability(state_sequence) returns a
  p(state_sequence | input_sequence; hmm). To get a joint probability of a
  state sequence and an input sequence p(state_sequence, input_sequence ; hmm)
  call dist.unnormalized_log_probability(state_squence).

  Args:
    init_logits: Array of shape (..., t)
    transition_logits: Array of shape (..., t, t)
    emission_dist:
      Array of shape (..., t, v) in case of categorical output or a
      continuous distribution with Distrax or TensorFlow Probability interface
      that has batch of shape (..., t).
    observations: Array of shape (..., n) of type jnp.int32 in case of
                  categorical output or (..., n, d) in case of d-dimensional
                  vector output.
    lengths:
      Lengths of each entry in the batch. It has the same shape as the batch
      and dtype of jnp.int32. If it's not passed, the maximal length will be
      assumed based on the log_potentials.shape[-3].
  Returns:
    Distribution that is in fact LinearChainCRF but is parametrized in such a
    way so that it behaves just the same as if it was an HMM.
  """
  if isinstance(observations, Float[Array, "*batch n d"]) and (
      hasattr(emission_dist, "log_prob")):
    # This is a distrax continuous distribution.
    x = jnp.moveaxis(observations, -2, 0)[..., None, :]  # (n, *batch_shape,1,d)
    lp = emission_dist.log_prob(x)  # (n, *batch_shape, state)
    emission_scores = jnp.moveaxis(lp, 0, -2)  # (*batch_shape, n, state)
  elif isinstance(observations, Int32[Array, "*batch n"]) and (
      isinstance(emission_dist, Float[Array, "*batch state voc"])):
    # This is a categorical distribution.
    emission_dist = jax.nn.log_softmax(emission_dist, -1)
    x = jnp.take_along_axis(emission_dist, observations[..., None, :], -1)
    emission_scores = jnp.swapaxes(x, -1, -2)  # (*batch_shape, n, state)
  else:
    raise ValueError("Arguments for emission_dist and observations do not fit.")

  return linear_chain_crf.LinearChainCRF(
      log_potentials=_expand_simple_sequence_model(
          jax.nn.log_softmax(init_logits),
          jax.nn.log_softmax(transition_logits),
          emission_scores),
      lengths=lengths)
