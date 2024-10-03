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

"""Implements general sampling utils for MC and SWOR sampling with states."""
from __future__ import annotations
import functools
from typing import Any, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from synjax._src.constants import INF  # pylint: disable=g-importing-member


Array = jax.Array
KeyArray = jax.Array


class State(eqx.Module):
  """A class for representing a state of an ancestral sampler."""

  def logprobs(self) -> Array:
    raise NotImplementedError

  def apply_transition(self, a: Array) -> State:
    raise NotImplementedError

  def is_finished(self) -> Array:
    raise NotImplementedError


def ancestral_sampling(key: KeyArray, init_state: State, max_length: int,
                       k: int, unroll: int = 1) -> Tuple[Array, Array, Array]:
  f = functools.partial(single_ancestral_sample, init_state=init_state,
                        max_length=max_length, unroll=unroll)
  keys = jax.random.split(key, k)
  return jax.vmap(f)(keys)


def single_ancestral_sample(key: KeyArray, init_state: State, max_length: int,
                            unroll: int) -> Tuple[Array, Array, Array]:
  """Expands the init_state to produce a single sample of length max_length.

  Args:
    key: KeyArray for producing a sample.
    init_state: Initial sampling state.
    max_length: Maximal length of sampling operations.
    unroll: Steps of jax.lax.scan to unroll.

  Returns:
    A triple of final sampler's state, log probability and actual sample.
  """
  def f(carry, key):
    state: State
    state_logprob: Array
    state, state_logprob = carry
    log_p = state.logprobs()
    p = jnp.exp(log_p)
    i = jax.random.choice(key, jnp.arange(len(p)), p=p)
    new_state = state.apply_transition(i)
    new_state_logprob = state_logprob + log_p[i]
    return (new_state, new_state_logprob), i
  keys = jax.random.split(key, max_length)
  (final_state, logprob), sample = jax.lax.scan(
      f, (init_state, jnp.zeros([])), keys, unroll=unroll)
  return final_state, logprob, sample


def beam_search(init_state: State, max_length: int, k: int,
                unroll: int = 1) -> Tuple[State, Array]:
  a = jax.eval_shape(init_state.logprobs).shape[-1]
  sbs = _GeneralStochasticBeamSearch(k=k, a=a, is_regular_beam_search=True,
                                     unroll=unroll)
  beam_state, logprobs, _ = sbs.sample(  # pytype: disable=wrong-arg-types
      key=None, init_state=init_state, max_length=max_length)
  return beam_state, logprobs


def stochastic_beam_search(
    key: KeyArray, init_state: State, max_length: int, k: int, unroll: int = 1
    ) -> Tuple[State, Array, Array]:
  a = init_state.logprobs().shape[-1]
  sbs = _GeneralStochasticBeamSearch(k=k, a=a, is_regular_beam_search=False,
                                     unroll=unroll)
  return sbs.sample(key=key, init_state=init_state, max_length=max_length)


class _GeneralStochasticBeamSearch:
  """Class containing logic for Stochastic Beam Search."""

  def __init__(self,
               k: int,
               a: int,
               *,
               is_regular_beam_search: bool = False,
               unroll: int = 1):
    self.k = k  # samples
    self.a = a  # actions
    self.is_regular_beam_search = is_regular_beam_search
    self.unroll = unroll

  def _expand_initial_state_to_beam(self,
                                    init_state) -> Tuple[State, Array, Array]:
    beam_state = jax.tree_util.tree_map(
        lambda x: jax.lax.broadcast(x, (self.k,)), init_state)
    return (beam_state,
            jnp.full(self.k, -jnp.inf).at[0].set(0),
            jnp.full(self.k, -jnp.inf).at[0].set(0))

  def _beam_state_subselect(self, beam_state: State, indices: Array) -> State:
    to_gather = indices//self.a
    to_apply = indices % self.a
    beam_state = jax.tree_util.tree_map(lambda x: x[to_gather], beam_state)
    beam_state = jax.vmap(beam_state.__class__.apply_transition)(beam_state,
                                                                 to_apply)
    return beam_state

  # pylint: disable=missing-function-docstring
  def _stochastic_beam_search_loop_body(self, beam, key: KeyArray
                                        ) -> Tuple[Tuple[State, Any, Any], Any]:
    beam_state: State
    beam_state, phis, gs = beam
    phis = (jnp.expand_dims(phis, -1) +
            jax.vmap(beam_state.__class__.logprobs)(beam_state))  # (k, a)
    if self.is_regular_beam_search:
      gs = phis
    else:
      gs = _gumbel_with_maximum(key=key,
                                location=phis,
                                target_max=jnp.expand_dims(gs, -1))
    _, best_indices = jax.lax.top_k(gs.reshape(-1), self.k)
    new_beam = (
        self._beam_state_subselect(beam[0], best_indices),
        phis.reshape(-1)[best_indices],
        gs.reshape(-1)[best_indices]
    )
    return new_beam, None  # None is output that will be ignored by scan.

  def sample(self, key: KeyArray, init_state,
             max_length: int) -> Tuple[State, Array, Array]:
    if self.is_regular_beam_search:
      split_key = jnp.zeros(max_length)
    else:
      if key is None:
        raise ValueError("SBS needs KeyArray")
      split_key = jax.random.split(key, max_length)
    return jax.lax.scan(self._stochastic_beam_search_loop_body,
                        self._expand_initial_state_to_beam(init_state),
                        split_key, unroll=self.unroll)[0]


def _gumbel_with_maximum(key: KeyArray,
                         location: Array,
                         target_max: Array,
                         axis: int = -1) -> Array:
  """Samples a set of gumbels with a given maximum and location.

  Note:
    This function implements the numericaly stable version of the truncated
    Gumbel distribution from appendix B.3 of Kool et al (2019).

  References:
    Kool et al, 2019 - Appendix B.3: https://arxiv.org/pdf/1903.06059.pdf#page=12

  Args:
    key: a KeyArray used as the random key.
    location: the location of gumbel distribution, e.g. log probabilities
      of partial sequence.
    target_max: The desired maximum sampled Gumbel, e.g. the previous perturbed
      log probabilities of a partial seq.
    axis: The dimesion with the maximum values.

  Returns:
    The sampled gumbels.
  """  # pylint: disable=line-too-long
  # Gumbels with location (e.g. `log_probabilities`, G_\phi_{S`} in the paper).
  gumbels = location + jax.random.gumbel(key, location.shape)
  gumbels = jnp.where(jnp.isneginf(gumbels), -INF, gumbels)

  # pylint: disable=invalid-name
  # Use equations (23) and (24) in Appendix B.3.
  T = target_max  # G_\phi_{S}, previous perturbed log_probs of partial seq.
  Z = jnp.max(gumbels, axis=axis, keepdims=True)  # current maximums
  # pylint: enable=invalid-name

  # Shift gumbels.
  v = T - gumbels + jnp.log1p(-jnp.exp(gumbels - Z))
  return T - jax.nn.relu(v) - jnp.log1p(jnp.exp(-jnp.abs(v)))
