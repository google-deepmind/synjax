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

"""Abstract test template for all distributions."""
import functools
from typing import List
from absl.testing import parameterized

# pylint: disable=g-importing-member
import jax
import jax.numpy as jnp
import numpy as np
from synjax._src import constants
from synjax._src.distribution import Distribution
import typeguard


class DistributionTest(parameterized.TestCase):

  def assert_zeros_and_ones(self, x):
    leaves = jax.tree_util.tree_flatten(x)[0]
    is_close = lambda a, b: jnp.isclose(  # pylint: disable=g-long-lambda
        a, b, rtol=constants.TESTING_RELATIVE_TOLERANCE,
        atol=constants.TESTING_ABSOLUTE_TOLERANCE)
    self.assertTrue(
        all(map(lambda x: jnp.all(is_close(x, 0) | is_close(x, 1)), leaves)),
        msg="Edges must be 0s and 1s only.")

  def assert_all(self, x, *, msg=""):
    self.assertTrue(all(map(jnp.all, jax.tree_util.tree_flatten(x)[0])),
                    msg=msg)

  def assert_no_duplicates_in_first_axis(self, x):
    def tree_equal(a, b):
      a_leaves, _ = jax.tree_util.tree_flatten(a)
      b_leaves, _ = jax.tree_util.tree_flatten(b)
      leaf_matches = [jnp.allclose(a_leaf, b_leaf)
                      for a_leaf, b_leaf in zip(a_leaves, b_leaves)]
      return functools.reduce(jnp.logical_and, leaf_matches)
    def vector_match(y, ys):
      return jax.vmap(tree_equal, in_axes=(None, 0))(y, ys)
    matrix_match = jax.vmap(vector_match, in_axes=(0, None))(x, x)
    self.assert_allclose(matrix_match,
                         jnp.eye(matrix_match.shape[-1], dtype=bool))

  def assert_allclose(self, x, y, *, msg="",
                      rtol=constants.TESTING_RELATIVE_TOLERANCE,
                      atol=constants.TESTING_ABSOLUTE_TOLERANCE):
    # This is different from standard np.test.assert_allclose in that
    # it allows for lower precision by default and it allows pytrees and
    # different shapes as long as they are broadcastable.
    def array_all_close(a, b):
      a = np.asarray(a)
      b = np.asarray(b)
      broadcasting_shape = np.broadcast_shapes(a.shape, b.shape)
      a = np.broadcast_to(a, broadcasting_shape)
      b = np.broadcast_to(b, broadcasting_shape)
      np.testing.assert_allclose(a, b, rtol=rtol, atol=atol, err_msg=msg)
    jax.tree_map(array_all_close, x, y)

  def create_random_batched_dists(self, key: jax.random.KeyArray
                                  ) -> List[Distribution]:
    raise NotImplementedError

  def create_symmetric_batched_dists(self) -> List[Distribution]:
    raise NotImplementedError

  def assert_is_symmetric(self, dist: Distribution, marginals: jax.Array
                          ) -> bool:
    raise NotImplementedError

  def assert_batch_of_valid_samples(self, dist: Distribution, samples: jax.Array
                                    ):
    raise NotImplementedError

  def assert_valid_marginals(self, dist: Distribution, marginals: jax.Array):
    raise NotImplementedError

  def analytic_log_count(self, dist: Distribution) -> jax.Array:
    """Computes log count of structrues analytically."""
    raise NotImplementedError

  def create_invalid_shape_distribution(self) -> Distribution:
    raise NotImplementedError

  def test_crash_on_invalid_shapes(self):
    self.assertRaises(typeguard.TypeCheckError,
                      self.create_invalid_shape_distribution)

  def test_symmetric(self):
    for dist in self.create_symmetric_batched_dists():
      assert dist.batch_shape
      self.assert_is_symmetric(dist, dist.marginals())

  def test_log_count(self):
    for dist in self.create_random_batched_dists(jax.random.PRNGKey(0)):
      log_predicted = dist.log_count()
      self.assert_all(log_predicted >= 0)
      predicted = jnp.exp(log_predicted)
      self.assert_all(jnp.round(predicted)-predicted < 0.1)
      try:
        self.assert_allclose(log_predicted, self.analytic_log_count(dist))
      except NotImplementedError:
        pass

  def test_marginals(self):
    for dist in self.create_random_batched_dists(jax.random.PRNGKey(0)):
      assert dist.batch_shape
      m = dist.marginals()
      self.assert_all(jax.tree_map(lambda x: (0 <= x) & (x <= 1.0001), m),
                      msg="Marginals must be between 0 and 1")
      self.assert_all(
          jax.tree_map(lambda x: jax.vmap(jnp.any)(0 < x), m),
          msg="Some marginals must be > 0")
      self.assert_valid_marginals(dist, m)

  def test_argmax(self):
    for dist in self.create_random_batched_dists(jax.random.PRNGKey(0)):
      assert dist.batch_shape
      best = dist.argmax()
      self.assert_zeros_and_ones(best)
      self.assert_batch_of_valid_samples(dist, best)
      self.assert_valid_marginals(dist, best)

      probs = jnp.exp(dist.log_prob(best))
      self.assertEqual(probs.shape, dist.batch_shape)
      self.assert_all((probs > 0) & (probs <= 1))

  def test_sampling(self):
    for dist in self.create_random_batched_dists(jax.random.PRNGKey(0)):
      assert dist.batch_shape
      k = 5
      samples = dist.sample(jax.random.PRNGKey(0), k)
      self.assert_zeros_and_ones(samples)
      self.assert_batch_of_valid_samples(dist, samples)
      # pylint: disable=cell-var-from-loop
      self.assert_all(jax.tree_map(lambda x: x.shape[0] == k, samples))
      prob = jnp.exp(dist.log_prob(samples))
      self.assert_all((0 < prob) & (prob <= 1))

  def test_entropy_cross_entropy(self):
    for dist, dist2 in zip(
        self.create_random_batched_dists(jax.random.PRNGKey(0)),
        self.create_random_batched_dists(jax.random.PRNGKey(1))):
      assert dist.batch_shape
      assert dist2.batch_shape
      entropy = dist.entropy()
      self_cross_entropy = dist.cross_entropy(dist)
      self_kl_divergence = dist.kl_divergence(dist)
      self.assert_allclose(entropy, self_cross_entropy)
      self.assert_all(entropy > 0)
      self.assert_all(self_cross_entropy > 0)
      self.assert_all(self_kl_divergence == 0)
      self.assert_all(dist2.kl_divergence(dist) > 0)

  def check_top_k_single_dist(self, dist: Distribution,
                              check_prefix_condition: bool = True):
    # pylint: disable=cell-var-from-loop
    assert len(dist.batch_shape) == 1
    k = 4
    best_k, best_k_scores_direct = dist.top_k(k)
    self.assert_zeros_and_ones(best_k)
    self.assert_all(jax.tree_map(lambda x: x.shape[0] == k, best_k))

    best_k_scores = dist.unnormalized_log_prob(best_k)
    self.assert_allclose(best_k_scores, best_k_scores_direct)
    self.assert_all(best_k_scores > -1e5)
    self.assert_all(best_k_scores[:-1] >= best_k_scores[1:])

    # All trees are valid.
    for i in range(k):
      best_i = jax.tree_map(lambda x: x[i], best_k)
      self.assert_valid_marginals(dist, best_i)
      self.assert_batch_of_valid_samples(dist, best_i)

    # All structs are different from each other.
    for i in range(dist.batch_shape[0]):
      self.assert_no_duplicates_in_first_axis(
          jax.tree_map(lambda x: x[:, i], best_k))

    top_1 = jax.tree_map(lambda x: x[None], dist.argmax())
    self.assert_allclose(dist.top_k(1)[0], top_1)
    if check_prefix_condition:
      # Top k-1 is a prefix of top k.
      self.assert_allclose(jax.tree_map(lambda x: x[:k-1], best_k),
                           dist.top_k(k-1)[0])
      self.assert_allclose(jax.tree_map(lambda x: x[:1], best_k), top_1)

  def test_top_k(self):
    for dist in self.create_random_batched_dists(jax.random.PRNGKey(0)):
      self.check_top_k_single_dist(dist)

  def test_argmax_can_be_jitted(self):
    f = jax.jit(lambda x: x.argmax())
    for dist in self.create_random_batched_dists(jax.random.PRNGKey(0)):
      jax.block_until_ready(f(dist))
      jax.block_until_ready(f(dist))

  def test_sampling_can_be_jitted(self):
    key = jax.random.PRNGKey(0)
    f = jax.jit(lambda x: x.sample(key))
    for dist in self.create_random_batched_dists(jax.random.PRNGKey(0)):
      jax.block_until_ready(f(dist))
      jax.block_until_ready(f(dist))
