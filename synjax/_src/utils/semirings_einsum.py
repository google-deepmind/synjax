# Copyright 2023 DeepMind Technologies Limited.
# Copyright 2018 The JAX Authors.
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

"""Generalized einsum that works over any semi-ring structure.

Big part of this code is borrowed from internal JAX codebase (version 0.3.15)
(mostly from jax._src.numpy.lax_numpy) and modified so that instead of calling
standard summation and multiplication operations, it calls the ones provided by
the user. Some code that is not part of the JAX interface was included so that
this module continues working even if internals of JAX change in the future.
"""
import collections
import functools
import inspect
import operator
from typing import Sequence, Tuple, FrozenSet, List, Optional, Any, cast
import jax
from jax import core
import jax.numpy as jnp
import numpy as np
import opt_einsum

__all__ = ["einsum_generalized"]


# Taken from jax._src.lax.lax
def _delta(dtype, shape, axes):
  """This utility function exists for creating Kronecker delta arrays."""
  axes = jax.util.safe_map(int, axes)
  dtype = jax.dtypes.canonicalize_dtype(dtype)
  base_shape = tuple(np.take(shape, axes))  # type: ignore[arg-type]
  iotas = [jax.lax.broadcasted_iota(jnp.uint32, base_shape, i)
           for i in range(len(base_shape))]
  eyes = [jax.lax.eq(i1, i2) for i1, i2 in zip(iotas[:-1], iotas[1:])]
  result = jax.lax.convert_element_type_p.bind(
      functools.reduce(operator.and_, eyes), new_dtype=dtype, weak_type=False)
  return jax.lax.broadcast_in_dim(result, shape, axes)


# Taken from jax._src.numpy.lax_numpy
def _removechars(s, chars):
  return s.translate(str.maketrans(dict.fromkeys(chars)))


# Taken and modified from jax._src.numpy.lax_numpy
def _einsum(operands: List[jnp.ndarray],
            contractions: Sequence[Tuple[Tuple[int, ...], FrozenSet[str], str]],
            precision, sum_fn, mul_op, dot_general, key: jax.Array):
  """This function executes tensor contraction operations.

  It is taken from jax._src.numpy.lax_numpy and modified so that it allows for
  arbitrary summation (not just jnp.sum) and arbitrary multiplication operations
  (not just jnp.multiply).

  Args:
    operands: Tensors that need to be contracted.
    contractions: Sequence of contractions specified by opt_einsum.
    precision: Desired precision if matrix multiplication is used internally.
    sum_fn: Function that does summation and has the same interface as jnp.sum.
    mul_op: Function that does multiplication ans has the same interface
            as jnp.multiply.
    dot_general: Function that optimizes sum_fn and mul_fn in case of
                 generalized dot product, similar to what jax.lax.dot_general
                 does for jnp.sum and jnp.multiply. If this argument is None,
                 dot_general will be constructed automatically, but it won't be
                 optimized for operations that can use cores dedicated for
                 matmul.
    key: Is a jax.Array that is split into sub-keys that are passed to
         sum_fn each time it is called. This is useful for semi-rings that
         require randomness.
  Returns:
    The result of a semi-ring einsum.
  """
  unzip2 = jax.util.unzip2

  # NOTE: In the original implementation from jax._src.numpy.lax_numpy the types
  # are promoted using operands = list(_promote_dtypes(*operands)) where
  # _promote_dtypes comes from jax._src.numpy.util. It is removed here because
  # we do not want to depend on the private parts of JAX.
  # Instead we add the condition below that requires
  # all inputs to have the same dtype so no promotion is required.
  operand_types = [x.dtype for x in operands]
  if any(x != operand_types[0] for x in operand_types[1:]):
    raise NotImplementedError(
        "in generalized einsum all operands have to have the same dtype")

  def sum_uniques(operand, names, uniques, *, key):
    if uniques:
      axes = [names.index(name) for name in uniques]
      operand = sum_fn(operand, axis=axes, key=key)
      names = _removechars(names, uniques)
    return operand, names

  def sum_repeats(operand, names, counts, keep_names, *,
                  key: jax.Array):
    for name, count in counts.items():
      if count > 1:
        axes = [i for i, n in enumerate(names) if n == name]
        eye = _delta(operand.dtype, operand.shape, axes)
        if name not in keep_names:
          operand = sum_fn(mul_op(operand, eye), axis=axes, key=key)
          names = names.replace(name, "")
        else:
          operand = sum_fn(mul_op(operand, eye), axis=axes[:-1], key=key)
          names = names.replace(name, "", count - 1)
    return operand, names

  def filter_singleton_dims(operand, names, other_shape, other_names):
    s = jnp.shape(operand)
    new_shape = []
    new_names = []
    for i, d in enumerate(names):
      other_i = other_names.find(d)
      if not core.symbolic_equal_dim(s[i], 1) or other_i == -1 or (
          core.symbolic_equal_dim(other_shape[other_i], 1)):
        new_shape.append(s[i])
        new_names.append(d)
    return jnp.reshape(operand, tuple(new_shape)), "".join(new_names)

  keys = jax.random.split(key, 5*len(contractions))
  for contraction_i, (operand_indices, contracted_names_set, einstr
                      ) in enumerate(contractions):
    contracted_names = sorted(contracted_names_set)
    input_str, result_names = einstr.split("->")
    input_names = input_str.split(",")

    # switch on the number of operands to be processed in this loop iteration.
    # every case here sets 'operand' and 'names'.
    if len(operand_indices) == 1:
      operand = operands.pop(operand_indices[0])
      names, = input_names
      counts = collections.Counter(names)

      # sum out unique contracted indices with a single reduce-sum
      uniques = [name for name in contracted_names if counts[name] == 1]
      operand, names = sum_uniques(operand, names, uniques,
                                   key=keys[contraction_i*5])

      # for every repeated index, do a contraction against an identity matrix
      operand, names = sum_repeats(operand, names, counts, result_names,
                                   key=keys[contraction_i*5+1])

    elif len(operand_indices) == 2:
      lhs, rhs = map(operands.pop, operand_indices)
      lhs_names, rhs_names = input_names

      # handle cases where one side of a contracting or batch dimension is 1
      # but its counterpart is not.
      lhs, lhs_names = filter_singleton_dims(lhs, lhs_names, jnp.shape(rhs),
                                             rhs_names)
      rhs, rhs_names = filter_singleton_dims(rhs, rhs_names, jnp.shape(lhs),
                                             lhs_names)

      lhs_counts = collections.Counter(lhs_names)
      rhs_counts = collections.Counter(rhs_names)

      # sum out unique contracted indices in lhs and rhs
      lhs_uniques = [name for name in contracted_names
                     if lhs_counts[name] == 1 and rhs_counts[name] == 0]
      lhs, lhs_names = sum_uniques(lhs, lhs_names, lhs_uniques,
                                   key=keys[contraction_i*5])

      rhs_uniques = [name for name in contracted_names
                     if rhs_counts[name] == 1 and lhs_counts[name] == 0]
      rhs, rhs_names = sum_uniques(rhs, rhs_names, rhs_uniques,
                                   key=keys[contraction_i*5+1])

      # for every repeated index, contract against an identity matrix
      lhs, lhs_names = sum_repeats(lhs, lhs_names, lhs_counts,
                                   result_names + rhs_names,
                                   key=keys[contraction_i*5+2])
      rhs, rhs_names = sum_repeats(rhs, rhs_names, rhs_counts,
                                   result_names + lhs_names,
                                   key=keys[contraction_i*5+3])

      lhs_or_rhs_names = set(lhs_names) | set(rhs_names)
      contracted_names = [x for x in contracted_names if x in lhs_or_rhs_names]
      lhs_and_rhs_names = set(lhs_names) & set(rhs_names)
      batch_names = [x for x in result_names if x in lhs_and_rhs_names]

      lhs_batch, rhs_batch = unzip2((lhs_names.find(n), rhs_names.find(n))
                                    for n in batch_names)

      # NOTE(mattjj): this can fail non-deterministically in python3, maybe
      # due to opt_einsum
      assert all(
          name in lhs_names and name in rhs_names and
          lhs.shape[lhs_names.index(name)] == rhs.shape[rhs_names.index(name)]
          for name in contracted_names)

      # contract using dot_general
      batch_names_str = "".join(batch_names)
      lhs_cont, rhs_cont = unzip2((lhs_names.index(n), rhs_names.index(n))
                                  for n in contracted_names)
      deleted_names = batch_names_str + "".join(contracted_names)
      remaining_lhs_names = _removechars(lhs_names, deleted_names)
      remaining_rhs_names = _removechars(rhs_names, deleted_names)
      # Try both orders of lhs and rhs, in the hope that one of them means we
      # don't need an explicit transpose. opt_einsum likes to contract from
      # right to left, so we expect (rhs,lhs) to have the best chance of not
      # needing a transpose.
      names = batch_names_str + remaining_rhs_names + remaining_lhs_names
      if names == result_names:
        dimension_numbers = ((rhs_cont, lhs_cont), (rhs_batch, lhs_batch))
        dot_general_args = (rhs, lhs)
      else:
        names = batch_names_str + remaining_lhs_names + remaining_rhs_names
        dimension_numbers = ((lhs_cont, rhs_cont), (lhs_batch, rhs_batch))
        dot_general_args = (lhs, rhs)
      operand = dot_general(*dot_general_args,
                            dimension_numbers=dimension_numbers,
                            key=keys[contraction_i*5+4],
                            precision=precision)
    else:
      raise NotImplementedError  # if this is actually reachable, open an issue!

    # the resulting 'operand' with axis labels 'names' should be a permutation
    # of the desired result
    assert len(names) == len(result_names) == len(set(names))
    assert set(names) == set(result_names)
    if names != result_names:
      perm = tuple(names.index(name) for name in result_names)
      operand = jax.lax.transpose(operand, perm)
    operands.append(operand)  # used in next iteration

  return operands[0]


# Taken and modified from jax._src.numpy.lax_numpy
def einsum_generalized(*operands, optimize="optimal", precision=None,
                       sum_fn, mul_op, dot_general,
                       key: Optional[jax.Array] = None):
  """Generalized version of einsum that works with arbitrary sum and mul ops.

  It is taken from jax._src.numpy.lax_numpy and modified minimally so that it
  allows for arbitrary summation (not just jnp.sum) and arbitrary multiplication
  (not just jnp.multiply) operations.

  Args:
    *operands: Tensors that need to be contracted.
    optimize: Level of opt_einsum optimization to use.
    precision: Desired precision if matrix multiplication is used internally.
    sum_fn: Function that does summation and has the same interface as jnp.sum.
    mul_op: Function that does multiplication ans has the same interface
            as jnp.multiply.
    dot_general: Function that optimizes sum_fn and mul_fn in case of
                 generalized dot product, similar to what jax.lax.dot_general
                 does for jnp.sum and jnp.multiply. If this argument is None,
                 dot_general will be constructed automatically, but it won't be
                 optimized for operations that can use cores dedicated for
                 matmul.
    key: Is a jax.Array that is split into sub-keys that are passed to
         sum_fn each time it is called. This is useful for semi-rings that
         require randomness.
  Returns:
    The result of a semi-ring einsum.
  """
  def add_key_wrap(fn):
    def fn2(*args, key=None, **kwargs):
      del key
      return fn(*args, **kwargs)
    return fn2
  spec = inspect.getfullargspec(sum_fn)
  if "key" not in (spec.args + spec.kwonlyargs):
    sum_fn = add_key_wrap(sum_fn)

  if key is None:
    key = jax.random.PRNGKey(0)

  # pylint: disable=g-bool-id-comparison
  optimize = "optimal" if optimize is True else optimize
  # using einsum_call=True here is an internal api for opt_einsum

  # Allow handling of shape polymorphism
  # pylint: disable=g-complex-comprehension
  non_constant_dim_types = {
      type(d) for op in operands if not isinstance(op, str)
      for d in jnp.shape(op) if not core.is_constant_dim(d)
  }
  if not non_constant_dim_types:
    einsum_contract_path_fn = opt_einsum.contract_path
  else:
    # NOTE: This branch in the original implementation from
    # jax._src.numpy.lax_numpy calls internal function
    # `_polymorphic_einsum_contract_path_handlers` but it's excluded here
    # because it seems useful only for jax2tf and we want not to depend on the
    # private functions of JAX.
    raise NotImplementedError("generalized version of einsum doesn't support"
                              "polymorphic contact path handlers")
  operands, contractions = einsum_contract_path_fn(
      *operands, einsum_call=True, use_blas=True, optimize=optimize)

  # The line below fixes the wrong return type of opt_einsum.contract_path.
  contractions = cast(List[Tuple[Any, ...]], contractions)

  contractions = tuple((a, frozenset(b), c) for a, b, c, *_ in contractions)
  return _einsum(operands, contractions, precision, sum_fn, mul_op, dot_general,
                 key=key)
