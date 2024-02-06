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
        """ Create an FSDPShardingRule.

        Args:
            fsdp_axis_name: The name of the FSDP axis.
            fsdp_axis_size: The mesh axis size for FSDP. This is used to find
                a suitable axis for FSDP sharding. If None, the largest power of
                two divisor of the tensor shape will be used.
            min_fsdp_size: The minimum size of a tensor to be sharded. If the
                tensor is smaller than this size, it will be replicated.
        """
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

    def __init__(self, *rules, strict=True):
        """ Create a TreePathShardingRule according to a list of regex rules.

        Args:
            rules: A list of pairs of regex rules and PartitionSpecs. The regex
                rules are used to match the path of the pytree, which is a /
                separated string of the path from the root of the pytree to the
                leaf node. The regex must match part of the path for the
                PartitionSpec to be applied.
            strict: A boolean, whether to raise an error if no rule is found for
                a leaf node. If False, a default PartitionSpec() will be used.
        """
        self.rules = rules
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

    def __init__(self, axis_dims, axis_names, mesh_axis_splitting=False):
        """ Create a MeshShardingHelper.

        Args:
            axis_dims: A tuple of integers, the shape of the mesh.
            axis_names: A tuple of strings, the names of the mesh axes.
            mesh_axis_splitting: A boolean, whether to allow splitting one physical
                axis into multiple logical axes.
        """
        self.axis_dims = tuple(axis_dims)
        self.axis_names = tuple(axis_names)
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

    def match_sharding_rule(self, sharding_rules, pytree):
        """ Apply sharding rules to a pytree to get a pytree of PartitionSpecs.

        Args:
            sharding_rules: The sharding rules or partition specs for the pytree.
            pytree: The pytree to be sharded.

        Returns:
            A pytree of PartitionSpecs with the same structure as the input pytree.
        """
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

    def sjit(self,
             fun,
             in_shardings=None,
             out_shardings=None,
             static_argnums=None,
             args_sharding_constraint=None,
             **kwargs):
        """ JIT compile a function with sharding rules.

        Args:
            fun: The function to be JIT compiled.
            in_shardings: The sharding rule or partition specs for the input of the function.
            out_shardings: The sharding rule or partition specs for the output of the function.
            static_argnums: The indices of the static arguments.
            args_sharding_constraint: The sharding rule or partition specs to constrain
                the args after the beginning of the function.
            kwargs: Additional arguments for jax.jit.

        Returns:
            The JIT compiled function.
        """
        static_args_jitted_fn_cache = dict()

        def sharding_constrained_fun(*args):
            if args_sharding_constraint is not None:
                if isinstance(args_sharding_constraint, list):
                    _args_sharding_constraint = tuple(args_sharding_constraint)
                else:
                    _args_sharding_constraint = args_sharding_constraint
                static_args, dynamic_args = self._split_static_dynamic_args(static_argnums, args)
                named_shardings = self.match_sharding_rule(_args_sharding_constraint, dynamic_args)
                dynamic_args = jax.lax.with_sharding_constraint(dynamic_args, named_shardings)
                args = self._combine_static_dynamic_args(static_argnums, static_args, dynamic_args)
            return fun(*args)

        def wrapped(*args):
            static_args = tuple(args[i] for i in static_argnums) if static_argnums is not None else ()
            if static_args in static_args_jitted_fn_cache:
                with self:
                    results = static_args_jitted_fn_cache[static_args](*args)
                return results

            if in_shardings is None:
                matched_in_shardings = None
            else:
                if isinstance(in_shardings, list):
                    _in_shardings = tuple(in_shardings)
                else:
                    _in_shardings = in_shardings
                _, dynamic_args = self._split_static_dynamic_args(static_argnums, args)
                matched_in_shardings = self.match_sharding_rule(_in_shardings, dynamic_args)

            if out_shardings is None:
                matched_out_shardings = None
            else:
                output = jax.eval_shape(lambda: fun(*args))
                matched_out_shardings = self.match_sharding_rule(out_shardings, output)

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

    def sharded_jit(self, *args, **kwargs):
        """ Alias for sjit for backward compatibility. """
        return self.sjit(*args, **kwargs)

    @classmethod
    def with_sharding_constraint(cls, pytree, sharding_rule):
        # Enforce shard constraint with global mesh
        if cls.global_mesh_helper is None:
            return pytree
        named_shardings = cls.global_mesh_helper.match_sharding_rule(sharding_rule, pytree)
        return jax.lax.with_sharding_constraint(pytree, named_shardings)

    def make_shard_and_gather_fns(self, pytree, sharding_rule):
        """
        Create pytree of sharding and gathering functions from sharding rule
        or a pytree of PartitionSpecs. This can be used to shard and gather
        a pytree of tensors.

        Args:
            pytree: The pytree to be sharded and gathered.
            sharding_rule: The sharding rule or partition specs for the pytree.

        Returns:
            A pair of pytrees of sharding and gathering functions, each with the
            same structure as the input pytree.
        """
        named_shardings = self.match_sharding_rule(sharding_rule, pytree)
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
                out_shardings=NamedSharding(self.mesh, PartitionSpec()),
            )
            def gather_fn(tensor):
                return jax.device_get(jax_gather_fn(tensor))
            return gather_fn

        shard_fns = jax.tree_util.tree_map(make_shard_fn, named_shardings)
        gather_fns = jax.tree_util.tree_map(make_gather_fn, named_shardings)
        return shard_fns, gather_fns

    @classmethod
    def apply_shard_and_gather_fns(cls, fns, pytree):
        """ Apply pytree of sharding and gathering functions to a pytree.

        Args:
            fns: The pytree of sharding or gathering functions.
            pytree: The pytree to be sharded or gathered.

        Returns:
            The sharded or gathered pytree with the same structure as the input pytree.
        """
        return jax.tree_util.tree_map(lambda fn, x: fn(x), fns, pytree)

    def local_data_to_global_array(self, pytree, batch_axis=0, mesh_axis_subset=None):
        """ Convert local data to a global array with sharding.

        Args:
            pytree: The local data pytree.
            batch_axis: The batch axis to shard the data.
            mesh_axis_subset: The subset of mesh axes to use for sharding.

        Returns:
            The sharded pytree of global arrays with the same structure as the input pytree.
        """
        if mesh_axis_subset is None:
            mesh_axis_subset = self.axis_names
        else:
            if isinstance(mesh_axis_subset, str):
                mesh_axis_subset = (mesh_axis_subset,)
            for name in mesh_axis_subset:
                assert name in self.axis_names, f'Axis name {name} not found in mesh axis names'
            mesh_axis_subset = tuple(mesh_axis_subset)

        sharding = NamedSharding(
            self.mesh,
            PartitionSpec(*[None for _ in range(batch_axis - 1)], mesh_axis_subset)
        )

        local_devices = jax.local_devices()
        local_device_count = len(local_devices)
        process_count = jax.process_count()

        def to_global_array(array):
            if isinstance(array, jax.Array):
                # Already a device array, we can use jnp to split it
                splits = jnp.split(array, local_device_count, axis=batch_axis)
            else:
                splits = np.split(array, local_device_count, axis=batch_axis)
            local_arrays = jax.device_put(splits, local_devices)
            output_shape = list(array.shape)
            output_shape[batch_axis] = output_shape[batch_axis] * process_count
            return jax.make_array_from_single_device_arrays(
                shape=output_shape,
                sharding=sharding,
                arrays=local_arrays
            )

        return jax.tree_util.tree_map(to_global_array, pytree)

