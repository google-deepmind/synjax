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

"""Tests for CTC distribution."""

from absl.testing import absltest

import jax
import jax.numpy as jnp

from synjax._src import ctc
from synjax._src import distribution_test


class CtcTest(distribution_test.DistributionTest):

  def create_random_batched_dists(self, key: jax.random.KeyArray):
    b, n, v, l = 2, 12, 400, 6
    key1, key2 = jax.random.split(key, 2)
    log_potentials = jax.random.uniform(key1, (b, n, v))
    labels = jax.random.randint(key2, (b, l), 1, v)
    label_lengths = jnp.full(b, l)
    input_lengths = jnp.full(b, n)
    blank_id = 0

    dists = [ctc.CTC(log_potentials, labels, label_lengths=label_lengths,
                     input_lengths=input_lengths, blank_id=blank_id)]
    return dists

  def create_symmetric_batched_dists(self):
    b, n, l = 2, 6, 6
    v = n
    log_potentials = jnp.zeros((b, n, v))
    labels = jnp.arange(1, n+1) * jnp.ones((b, 1), dtype=jnp.int32)
    label_lengths = jnp.full(b, l)
    input_lengths = jnp.full(b, n)
    blank_id = 0

    dists = [ctc.CTC(log_potentials, labels, label_lengths=label_lengths,
                     input_lengths=input_lengths, blank_id=blank_id)]
    return dists

  def create_invalid_shape_distribution(self):
    b, n, l = 2, 6, 6
    v = n
    log_potentials = jnp.zeros((b, 1, n, v))
    labels = jnp.arange(1, n+1) * jnp.ones((b, 1), dtype=jnp.int32)
    label_lengths = jnp.full(b, l)
    input_lengths = jnp.full(b, n)
    blank_id = 0

    dists = [ctc.CTC(log_potentials, labels, label_lengths=label_lengths,
                     input_lengths=input_lengths, blank_id=blank_id)]
    return dists

  def assert_is_symmetric(self, dist, marginals) -> bool:
    self.assert_all(marginals >= 0)

  def assert_batch_of_valid_samples(self, dist, samples):
    transitions_count = jnp.sum(samples, (-1, -2))
    self.assert_all(transitions_count >= min(*dist.event_shape))

  def assert_valid_marginals(self, dist, marginals):
    self.assert_allclose(marginals[..., -2:, -1].sum(-1), 1)
    self.assert_allclose(marginals[..., :2, 0].sum(-1), 1)

  def test_CTC_loss_against_optax(self):
    for dist in self.create_random_batched_dists(jax.random.PRNGKey(0)):
      self.assert_allclose(dist.loss(use_optax=True),
                           dist.loss(use_optax=False))
      self.assert_allclose(dist.log_partition(use_optax=True),
                           dist.log_partition(use_optax=False))
      self.assert_all(dist.loss(use_optax=True) > 0)

  def test_alignment_to_labels(self):
    for dist in self.create_random_batched_dists(jax.random.PRNGKey(0)):
      kbest_alignments = dist.top_k(3)[0]
      kbest_labelings = dist.alignment_to_labels(kbest_alignments)
      dist.log_prob_labels(kbest_labelings)


if __name__ == "__main__":
  absltest.main()
