import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=8'

from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as PS
from jax.sharding import Mesh, NamedSharding
from absl.testing import absltest, parameterized

from scalax.sharding import (
    FSDPShardingRule, TreePathShardingRule, PolicyShardingRule,
    MeshShardingHelper
)


class FSDPShardingRuleTest(parameterized.TestCase):

    @parameterized.parameters(
        (4, 1024),
        (8, 2048),
        (2, 512),
    )
    def test_sharding_rule(self, fsdp_axis_size, min_fsdp_size):
        pytree = {
            'scaler': jnp.ones([]),
            'small_vector': jnp.ones(min_fsdp_size - fsdp_axis_size),
            'large_vector': jnp.ones(min_fsdp_size * 2),
            'small_matrix': jnp.ones((16, 16)),
            'large_matrix': jnp.ones((min_fsdp_size, min_fsdp_size)),
            'weird_matrix': jnp.ones((min_fsdp_size + 1, min_fsdp_size)),
            'really_weird_matrix': jnp.ones((min_fsdp_size + 1, min_fsdp_size + 1)),
        }
        sharding_rule = FSDPShardingRule(
            fsdp_axis_name='fsdp',
            fsdp_axis_size=fsdp_axis_size,
            min_fsdp_size=min_fsdp_size,
        )
        matched_partition_specs = sharding_rule.apply(pytree)

        expected_partition_specs = {
            'scaler': PS(),
            'small_vector': PS(),
            'large_vector': PS('fsdp'),
            'small_matrix': PS(),
            'large_matrix': PS('fsdp', None),
            'weird_matrix': PS(None, 'fsdp'),
            'really_weird_matrix': PS(),
        }
        self.assertEqual(matched_partition_specs, expected_partition_specs)


class TreePathShardingRuleTest(parameterized.TestCase):

    def test_tree_path_sharding_rule(self):
        pytree = {
            'a': jnp.ones((16, 16)),
            'b': {
                'c': jnp.ones((16, 16)),
                'd': jnp.ones((16, 16)),
            },
            'e': jnp.ones([]),
        }
        sharding_rule = TreePathShardingRule(
            ('a', PS('x', 'y')),
            ('b/c', PS('x')),
            ('b/d', PS('y')),
        )
        matched_partition_specs = sharding_rule.apply(pytree)

        expected_partition_specs = {
            'a': PS('x', 'y'),
            'b': {
                'c': PS('x'),
                'd': PS('y'),
            },
            'e': PS(),
        }
        self.assertEqual(matched_partition_specs, expected_partition_specs)

    def test_tree_path_sharding_rule_strict(self):
        """ Test that the sharding rule is strict and raises an error if a
            leaf is not found in the rule patterns.
        """
        pytree = {
            'a': jnp.ones((16, 16)),
            'b': {
                'c': jnp.ones((16, 16)),
                'd': jnp.ones((16, 16)),
            },
        }
        sharding_rule = TreePathShardingRule(
            ('a', PS('x', 'y')),
            ('b/c', PS('x')),
            strict=True,
        )
        with self.assertRaises(ValueError):
            matched_partition_specs = sharding_rule.apply(pytree)


class PolicyShardingRuleTest(parameterized.TestCase):

    def test_policy_sharding_rule(self):
        pytree = {
            'a': jnp.ones((16, 16)),
            'b': {
                'c': jnp.ones((16, 16), dtype=jnp.float32),
                'd': jnp.ones((16, 16), dtype=jnp.int32),
            },
            'e': jnp.ones([]),
        }

        def policy_fn(path, value):
            if path == 'a':
                return PS('x')
            elif value.dtype == jnp.int32:
                return PS('y')
            elif len(value.shape) == 0:
                return PS('z')
            else:
                return PS()

        sharding_rule = PolicyShardingRule(policy_fn)
        matched_partition_specs = sharding_rule.apply(pytree)

        expected_partition_specs = {
            'a': PS('x'),
            'b': {
                'c': PS(),
                'd': PS('y'),
            },
            'e': PS('z'),
        }
        self.assertEqual(matched_partition_specs, expected_partition_specs)


class MeshShardingHelperTest(parameterized.TestCase):

    @parameterized.parameters(32, 64, 192)
    def test_sjit_out_shardings(self, dim):
        mesh = MeshShardingHelper(
            axis_dims=(2, 4),
            axis_names=('x', 'y'),
        )

        sharding_rule = TreePathShardingRule(
            ('a', PS('x', 'y')),
            ('b', PS('y', 'x')),
        )

        @partial(
            mesh.sjit,
            out_shardings=(sharding_rule, PS(('x', 'y')))
        )
        def sharded_fn(x):
            output_rule = {
                'a': jnp.zeros((dim, dim)),
                'b': jnp.zeros((dim, dim)),
            }
            output_ps = jnp.zeros((dim, dim))
            return output_rule, output_ps

        output_rule, output_ps = sharded_fn(1.0)
        self.assertEqual(
            output_rule['a'].sharding,
            NamedSharding(mesh.mesh, PS('x', 'y'))
        )
        self.assertEqual(
            output_rule['b'].sharding,
            NamedSharding(mesh.mesh, PS('y', 'x'))
        )
        self.assertEqual(
            output_ps.sharding,
            NamedSharding(mesh.mesh, PS(('x', 'y')))
        )

    @parameterized.parameters(32, 64, 192)
    def test_with_sharding_constraint(self, dim):
        mesh = MeshShardingHelper(
            axis_dims=(2, 4),
            axis_names=('x', 'y'),
        )

        sharding_rule = PolicyShardingRule(lambda path, value: PS('x', 'y'))

        @mesh.sjit
        def rule_constrained_fn(x):
            return MeshShardingHelper.with_sharding_constraint(
                x, PolicyShardingRule(lambda path, value: PS('x', 'y'))
            )

        @mesh.sjit
        def spec_constrained_fn(x):
            return MeshShardingHelper.with_sharding_constraint(
                x, PS('x', 'y')
            )

        @jax.jit
        def reference_fn(x):
            return jax.lax.with_sharding_constraint(
                x, NamedSharding(mesh.mesh, PS('x', 'y'))
            )

        sjit_rule_output = rule_constrained_fn(jnp.ones((dim, dim)))
        sjit_spec_output = spec_constrained_fn(jnp.ones((dim, dim)))
        reference_output = reference_fn(jnp.ones((dim, dim)))
        self.assertEqual(sjit_rule_output.sharding, reference_output.sharding)
        self.assertEqual(sjit_spec_output.sharding, reference_output.sharding)

    @parameterized.parameters(32, 64, 192)
    def test_with_sharding_annotation(self, dim):
        mesh = MeshShardingHelper(
            axis_dims=(2, 4),
            axis_names=('x', 'y'),
        )

        sharding_rule = PolicyShardingRule(lambda path, value: PS('x', 'y'))

        @partial(
            mesh.sjit,
            sharding_annotation_rules={'activation': sharding_rule}
        )
        def rule_constrained_fn(x):
            return MeshShardingHelper.with_sharding_annotation(
                x, 'activation'
            )

        @partial(
            mesh.sjit,
            sharding_annotation_rules={'activation': PS('x', 'y')}
        )
        def spec_constrained_fn(x):
            return MeshShardingHelper.with_sharding_annotation(
                x, 'activation'
            )

        @jax.jit
        def reference_fn(x):
            return jax.lax.with_sharding_constraint(
                x,
                NamedSharding(mesh.mesh, PS('x', 'y'))
            )

        sjit_rule_output = rule_constrained_fn(jnp.ones((dim, dim)))
        sjit_spec_output = spec_constrained_fn(jnp.ones((dim, dim)))
        reference_output = reference_fn(jnp.ones((dim, dim)))
        self.assertEqual(sjit_rule_output.sharding, reference_output.sharding)
        self.assertEqual(sjit_spec_output.sharding, reference_output.sharding)


if __name__ == '__main__':
  absltest.main()