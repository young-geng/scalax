from functools import partial
import re
import abc
import numpy as np

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec
from jax.sharding import Mesh, NamedSharding
from jax.experimental import mesh_utils

from scalax.utils import named_tree_map


class ShardingRule(abc.ABC):
    """ Base class for sharding rules. """

    @abc.abstractmethod
    def apply(self, pytree):
        pass


class FSDPShardingRule(ShardingRule):
    """ Create FSDP sharding PartitionSpec for a pytree. """

    def __init__(self, fsdp_axis_name='fsdp', fsdp_axis_size=None, min_fsdp_size=1048576):
        self.fsdp_axis_name = fsdp_axis_name
        self.fsdp_axis_size = fsdp_axis_size
        self.min_fsdp_size = min_fsdp_size

    def largest_power_of_two_divisor(self, n):
        k = 0
        while n % 2 == 0:
            n //= 2
            k += 1
        return 2 ** k

    def apply(self, pytree):
        def get_partition_spec(tensor):
            # We only shard the float weights
            if np.prod(tensor.shape) >= self.min_fsdp_size:
                partition_spec = [None for _ in range(len(tensor.shape))]
                if self.fsdp_axis_size is None:
                    # Guess the FSDP axis size is a power of two
                    allowed_sizes = [
                        -self.largest_power_of_two_divisor(n)
                        for n in tensor.shape
                    ]
                    for i in np.argsort(allowed_sizes):
                        if tensor.shape[i] > 1:
                            partition_spec[i] = self.fsdp_axis_name
                            return PartitionSpec(*partition_spec)
                else:
                    for i in np.argsort([-x for x in tensor.shape]):
                        if tensor.shape[i] % self.fsdp_axis_size == 0:
                            partition_spec[i] = self.fsdp_axis_name
                            return PartitionSpec(*partition_spec)
            return PartitionSpec()

        return jax.tree_util.tree_map(get_partition_spec, pytree)


class TreePathShardingRule(ShardingRule):
    """ Create PartitionSpec for a pytree according to a list of regex rules. """

    def __init__(self, *ruless, strict=True):
        self.rules = ruless
        self.strict = strict

    def apply(self, pytree):
        """ Returns a pytree of PartitionSpec according to rules. """
        def get_partition_spec(name, leaf):
            if len(leaf.shape) == 0 or np.prod(leaf.shape) == 1:
                """ Don't partition scalar values. """
                return PartitionSpec()
            for rule, ps in self.rules:
                if re.search(rule, name) is not None:
                    return ps
            if self.strict:
                raise ValueError(f'Partition rule not found for param: {name}')
            return PartitionSpec()
        return named_tree_map(get_partition_spec, pytree, sep='/')


class MeshShardingHelper(object):
    """ Helper class for creating jit sharding jax functions with sharding rules. """
    global_mesh_helper = None

    def __init__(self, axis_dims, axis_names, mesh_axis_splitting=True):
        mesh_shape = np.arange(jax.device_count()).reshape(axis_dims).shape
        if mesh_axis_splitting:
            physical_mesh = np.array(jax.devices()).reshape(mesh_shape)
        else:
            physical_mesh = mesh_utils.create_device_mesh(mesh_shape)
        self.mesh = Mesh(physical_mesh, axis_names)
        self.previous_global_meshes = []

    def __enter__(self):
        # Use current mesh as global mesh
        self.previous_global_meshes.append(MeshShardingHelper.global_mesh_helper)
        MeshShardingHelper.global_mesh_helper = self
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # Restore last global mesh
        MeshShardingHelper.global_mesh_helper = self.previous_global_meshes.pop()

    def _split_static_dynamic_args(self, static_argnums, args):
        if static_argnums is None:
            return None, args
        static_args = tuple(args[i] for i in static_argnums)
        dynamic_args = tuple(args[i] for i in range(len(args)) if i not in static_argnums)
        return static_args, dynamic_args

    def _combine_static_dynamic_args(self, static_argnums, static_args, dynamic_args):
        if static_argnums is None:
            return dynamic_args
        args = list(dynamic_args)
        for i, arg in zip(static_argnums, static_args):
            args.insert(i, arg)
        return tuple(args)

    def _match_sharding_rule(self, sharding_rules, pytree):
        def get_partition_spec(rule, pytree):
            if isinstance(rule, ShardingRule):
                return jax.tree_util.tree_map(
                    lambda x: NamedSharding(self.mesh, x),
                    rule.apply(pytree)
                )
            else:
                return jax.tree_util.tree_map(
                    lambda x: NamedSharding(self.mesh, rule),
                    pytree
                )

        def is_leaf(x):
            # Check if the node is None, a PartitionSpec or a ShardingRule
            return (
                x is None
                or isinstance(x, ShardingRule)
                or isinstance(x, PartitionSpec)
            )

        return jax.tree_util.tree_map(
            get_partition_spec, sharding_rules, pytree, is_leaf=is_leaf
        )

    def sharded_jit(self,
                    fun,
                    in_shardings=None,
                    out_shardings=None,
                    static_argnums=None,
                    args_sharding_constraint=None,
                    **kwargs):
        static_args_jitted_fn_cache = dict()

        def sharding_constrained_fun(*args):
            if args_sharding_constraint is not None:
                if isinstance(args_sharding_constraint, list):
                    _args_sharding_constraint = tuple(args_sharding_constraint)
                else:
                    _args_sharding_constraint = args_sharding_constraint
                named_shardings = self._match_sharding_rule(_args_sharding_constraint, args)
                static_args, dynamic_args = self._split_static_dynamic_args(static_argnums, args)
                dynamic_args = jax.lax.with_sharding_constraint(dynamic_args, named_shardings)
                args = self._combine_static_dynamic_args(static_argnums, static_args, dynamic_args)
            return fun(*args)

        def wrapped(*args):
            static_args = tuple(args[i] for i in static_argnums) if static_argnums is not None else ()
            if static_args in static_args_jitted_fn_cache:
                return static_args_jitted_fn_cache[static_args](*args)

            if in_shardings is None:
                matched_in_shardings = None
            else:
                if isinstance(in_shardings, list):
                    _in_shardings = tuple(in_shardings)
                else:
                    _in_shardings = in_shardings
                _, dynamic_args = self._split_static_dynamic_args(static_argnums, args)
                matched_in_shardings = self._match_sharding_rule(_in_shardings, dynamic_args)

            if out_shardings is None:
                matched_out_shardings = None
            else:
                output = jax.eval_shape(lambda: fun(*args))
                matched_out_shardings = self._match_sharding_rule(out_shardings, output)

            jitted_fn = jax.jit(
                sharding_constrained_fun,
                in_shardings=matched_in_shardings,
                out_shardings=matched_out_shardings,
                static_argnums=static_argnums,
                **kwargs
            )

            static_args_jitted_fn_cache[static_args] = jitted_fn

            with self:
                results = jitted_fn(*args)
            return results

        return wrapped

    @classmethod
    def with_sharding_constraint(cls, pytree, sharding_rule):
        # Enforce shard constraint with global mesh
        if cls.global_mesh_helper is None:
            return pytree
        named_shardings = cls.global_mesh_helper._match_sharding_rule(sharding_rule, pytree)
        return jax.lax.with_sharding_constraint(pytree, named_shardings)

    def make_shard_and_gather_fns(self, pytree, sharding_rule):
        """ Create pytree of sharding and gathering functions from sharding rule
            or a pytree of PartitionSpecs. This can be used to shard and gather
            a pytree of tensors.
        """
        named_shardings = self._match_sharding_rule(sharding_rule, pytree)
        def make_shard_fn(partition_spec):
            jax_shard_function = jax.jit(
                lambda x: x,
                in_shardings=None,
                out_shardings=partition_spec
            )
            def shard_fn(tensor):
                return jax_shard_function(tensor).block_until_ready()
            return shard_fn

        def make_gather_fn(partition_spec):
            jax_gather_fn = jax.jit(
                lambda x: x,
                in_shardings=partition_spec,
                out_shardings=None
            )
            def gather_fn(tensor):
                return jax.device_get(jax_gather_fn(tensor))
            return gather_fn

        shard_fns = jax.tree_util.tree_map(make_shard_fn, named_shardings)
        gather_fns = jax.tree_util.tree_map(make_gather_fn, named_shardings)
        return shard_fns, gather_fns

    @classmethod
    def apply_shard_and_gather_fns(cls, fns, pytree):
        return jax.tree_util.tree_map(lambda fn, x: fn(x), fns, pytree)
