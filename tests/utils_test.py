import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
from absl.testing import absltest, parameterized

from scalax.utils import (
    JaxRNG, get_float_dtype_by_name, float_tensor_to_dtype, float_to_dtype,
    tree_path_to_string, flatten_tree, named_tree_map
)


class JaxRNGTest(parameterized.TestCase):

    @parameterized.parameters(2, 4, 8, 16)
    def test_rng(self, n_rngs):
        rng = jax.random.PRNGKey(42)
        rng_obj = JaxRNG.from_seed(42)
        rng_obj2 = JaxRNG(jax.random.PRNGKey(42))

        assert_equal_fn = lambda x, y: jax.tree_util.tree_map(
            lambda a, b: self.assertTrue(jnp.array_equal(a, b)),
            x, y
        )

        rng, split_rng = jax.random.split(rng)
        assert_equal_fn(split_rng, rng_obj())
        assert_equal_fn(split_rng, rng_obj2())

        split_rngs = jax.random.split(rng, n_rngs + 1)
        rng = split_rngs[0]
        split_rngs = split_rngs[1:]
        assert_equal_fn(split_rngs, rng_obj(n_rngs))
        assert_equal_fn(split_rngs, rng_obj2(n_rngs))

        keys = [str(x) for x in range(n_rngs)]
        split_rngs = jax.random.split(rng, n_rngs + 1)
        rng = split_rngs[0]
        split_rngs = split_rngs[1:]
        split_rngs = {key: val for key, val in zip(keys, split_rngs)}
        assert_equal_fn(split_rngs, rng_obj(keys))
        assert_equal_fn(split_rngs, rng_obj2(keys))


    @parameterized.parameters(2, 4, 8, 16)
    def test_global_rng(self, n_rngs):
        rng = jax.random.PRNGKey(42)
        JaxRNG.init_global_rng(42)

        assert_equal_fn = lambda x, y: jax.tree_util.tree_map(
            lambda a, b: self.assertTrue(jnp.array_equal(a, b)),
            x, y
        )

        rng, split_rng = jax.random.split(rng)
        assert_equal_fn(split_rng, JaxRNG.next_rng())

        split_rngs = jax.random.split(rng, n_rngs + 1)
        rng = split_rngs[0]
        split_rngs = split_rngs[1:]
        assert_equal_fn(split_rngs, JaxRNG.next_rng(n_rngs))

        keys = [str(x) for x in range(n_rngs)]
        split_rngs = jax.random.split(rng, n_rngs + 1)
        rng = split_rngs[0]
        split_rngs = split_rngs[1:]
        split_rngs = {key: val for key, val in zip(keys, split_rngs)}
        assert_equal_fn(split_rngs, JaxRNG.next_rng(keys))


class FloatTensorDTypeTest(parameterized.TestCase):

    def test_get_float_dtype_by_name(self):
        self.assertEqual(get_float_dtype_by_name('bf16'), jnp.bfloat16)
        self.assertEqual(get_float_dtype_by_name('bfloat16'), jnp.bfloat16)
        self.assertEqual(get_float_dtype_by_name('fp16'), jnp.float16)
        self.assertEqual(get_float_dtype_by_name('float16'), jnp.float16)
        self.assertEqual(get_float_dtype_by_name('fp32'), jnp.float32)
        self.assertEqual(get_float_dtype_by_name('float32'), jnp.float32)
        self.assertEqual(get_float_dtype_by_name('fp64'), jnp.float64)
        self.assertEqual(get_float_dtype_by_name('float64'), jnp.float64)

    @parameterized.parameters(
        jnp.bfloat16, jnp.float16, jnp.float32,
        'bf16', 'bfloat16', 'fp16', 'float16', 'fp32', 'float32'
    )
    def test_float_tensor_to_dtype(self, target_dtype):
        if isinstance(target_dtype, str):
            target_dtype_object = get_float_dtype_by_name(target_dtype)
        else:
            target_dtype_object = target_dtype
        for init_dtype in [jnp.bfloat16, jnp.float16, jnp.float32]:
            tensor = jnp.ones((16, 16), dtype=init_dtype)
            self.assertEqual(float_tensor_to_dtype(tensor, target_dtype).dtype, target_dtype_object)

        for init_dtype in [jnp.bool_, jnp.int8, jnp.int16, jnp.int32, jnp.int64, jnp.uint8, jnp.uint16, jnp.uint32, jnp.uint64]:
            tensor = jnp.ones((16, 16), dtype=init_dtype)
            self.assertIs(float_tensor_to_dtype(tensor, target_dtype), tensor)

        x = object()
        self.assertIs(float_tensor_to_dtype(x, target_dtype), x)

    def test_float_tensor_to_dtype_no_dtype(self):
        tensor = jnp.ones((16, 16))
        self.assertIs(float_tensor_to_dtype(tensor, None), tensor)
        self.assertIs(float_tensor_to_dtype(tensor, ''), tensor)

    @parameterized.parameters(
        jnp.bfloat16, jnp.float16, jnp.float32,
        'bf16', 'bfloat16', 'fp16', 'float16', 'fp32', 'float32'
    )
    def test_float_to_dtype(self, target_dtype):
        if isinstance(target_dtype, str):
            target_dtype_object = get_float_dtype_by_name(target_dtype)
        else:
            target_dtype_object = target_dtype

        pytree = {
            'a': jnp.ones((16, 16), dtype=jnp.float16),
            'b': {
                'c': jnp.ones((16, 16), dtype=jnp.float32),
                'd': jnp.ones((16, 16), dtype=jnp.int32),
                'e': jnp.ones(4, dtype=jnp.bfloat16),
            },
            'f': jnp.ones(4, dtype=jnp.int8),
            'g': jnp.ones(4, dtype=jnp.bool_),
        }

        expected_pytree = {
            'a': jnp.ones((16, 16), dtype=target_dtype_object),
            'b': {
                'c': jnp.ones((16, 16), dtype=target_dtype_object),
                'd': jnp.ones((16, 16), dtype=jnp.int32),
                'e': jnp.ones(4, dtype=target_dtype_object),
            },
            'f': jnp.ones(4, dtype=jnp.int8),
            'g': jnp.ones(4, dtype=jnp.bool_),
        }

        assert_equal_fn = lambda x, y: jax.tree_util.tree_map(
            lambda a, b: self.assertTrue(jnp.array_equal(a, b)),
            x, y
        )

        assert_equal_fn(float_to_dtype(pytree, target_dtype), expected_pytree)


class TreePathTest(parameterized.TestCase):

    def test_tree_path_to_string(self):
        path = [
            jax.tree_util.SequenceKey(0),
            jax.tree_util.DictKey('a'),
            jax.tree_util.GetAttrKey('b'),
            jax.tree_util.FlattenedIndexKey(1)
        ]
        self.assertEqual(tree_path_to_string(path), ('0', 'a', 'b', '1'))
        self.assertEqual(tree_path_to_string(path, sep='/'), '0/a/b/1')

    def test_flatten_tree(self):
        assert_equal_fn = lambda x, y: jax.tree_util.tree_map(
            lambda a, b: self.assertTrue(jnp.array_equal(a, b)),
            x, y
        )

        pytree = {
            'a': jnp.ones((16, 16)),
            'b': {
                'c': jnp.ones((16, 16)),
                'd': jnp.ones((16, 16)),
            },
            'e': jnp.ones([]),
        }
        flattened = flatten_tree(pytree, sep='/')
        expected_flattened = {
            'a': jnp.ones((16, 16)),
            'b/c': jnp.ones((16, 16)),
            'b/d': jnp.ones((16, 16)),
            'e': jnp.ones([]),
        }
        assert_equal_fn(flattened, expected_flattened)


        flattened = flatten_tree(pytree)
        expected_flattened = {
            ('a',): jnp.ones((16, 16)),
            ('b', 'c'): jnp.ones((16, 16)),
            ('b', 'd'): jnp.ones((16, 16)),
            ('e',): jnp.ones([]),
        }
        assert_equal_fn(flattened, expected_flattened)


    def test_named_tree_map(self):
        assert_equal_fn = lambda x, y: jax.tree_util.tree_map(
            lambda a, b: self.assertTrue(jnp.array_equal(a, b)),
            x, y
        )
        pytree = {
            'a': jnp.ones(1),
            'b': {
                'c': jnp.ones(2),
                'd': jnp.ones(3),
            },
            'e': jnp.ones(4),
        }
        expected_pytree = {
            'a': jnp.zeros(1),
            'b': {
                'c': jnp.ones(2),
                'd': jnp.zeros(3),
            },
            'e': jnp.ones(4),
        }

        def map_fn(path, value):
            if path == 'a':
                return jnp.zeros_like(value)
            elif path == 'b/c':
                return jnp.ones_like(value)
            elif path == 'b/d':
                return jnp.zeros_like(value)
            elif path == 'e':
                return jnp.ones_like(value)
            else:
                raise ValueError(f'Unexpected path: {path}')

        assert_equal_fn(named_tree_map(map_fn, pytree, sep='/'), expected_pytree)

        expected_pytree = {
            'a': jnp.zeros(1),
            'b': {
                'c': jnp.ones(2),
                'd': jnp.zeros(3),
            },
            'e': jnp.ones(4),
        }

        def map_fn(path, value):
            if path == ('a', ):
                return jnp.zeros_like(value)
            elif path == ('b', 'c'):
                return jnp.ones_like(value)
            elif path == ('b', 'd'):
                return jnp.zeros_like(value)
            elif path == ('e', ):
                return jnp.ones_like(value)
            else:
                raise ValueError(f'Unexpected path: {path}')

        assert_equal_fn = lambda x, y: jax.tree_util.tree_map(
            lambda a, b: self.assertTrue(jnp.array_equal(a, b)),
            x, y
        )
        assert_equal_fn(named_tree_map(map_fn, pytree), expected_pytree)


if __name__ == '__main__':
    absltest.main()
