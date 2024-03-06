# Sharding
The sharding utilities in scalax are contained in the `scalax.sharding` submodule.
`scalax.sharding` provides a set of utilities to help the users managing the accelerator
device mesh and shard the model and computations. The main object for managing
device mesh and sharding is `scalax.sharding.MeshShardingHelper` and its `sjit`
method. This can be composed with various `ShardingRule` object to specify the
shardings of computation.


## Class `MeshShardingHelper`
`MeshShardingHelper` is the main object for managing device mesh and sharding.
It mangages the devie mesh and provides a convenient interface to JAX's JIT
compiler.

### Constructor
- `axis_dims`: A list or tuple of integers representing the shape of the device
  mesh. The length of the list or tuple should be the same as the number of
  dimensions of the device mesh. Each integer represents the number of devices
  along the corresponding axis. One of the integers can be -1, which means that
  the size of mesh along that axis is inferred from the number of devices.
- `axis_names`: A list or tuple of strings representing the names of the axes of
  the device mesh. It should have the same length as `axis_dims`.
- `mesh_axis_splitting`: Default to `False`. On certain platforms like TPU,
  devices are arranged in a physical mesh with certain dimensions. This parameter
  controls whether  splitting a phyisical mesh dimension into multiple logical
  mesh dimensions is allowed. For example, on TPU pod with physical topology
  4x4x4, it would be impossible to constract 8x8 logical mesh without splitting
  the physical mesh dimensions. However, please note that splitting a physical
  mesh dimension may lead to degraded communication bandwidth between devices.

### Method `sjit`
`sjit` is a wrapper around `jax.jit`, with additional support for `ShardingRule`
and sharding annotations in addition to the usual `PartitionSpecs`.

Parameters
- `fun`: The function to be compiled.
- `in_shardings`: A tuple of `ShardingRule`, `PartitionSpecs` objects or `None`
   representing the sharding annotations for the input arguments of the function.
   `None` means that the sharding is inferred by XLA.
- `out_shardings`: A tuple of `ShardingRule`, `PartitionSpecs` objects or `None`
   representing the sharding annotations for the output of the function. `None`
   means that the sharding is inferred by XLA.
- `static_argnums`: A tuple of integers representing the argument indices that
  should be treated as static arguments.
- `args_sharding_constraint`: A tuple of `ShardingRule`, `PartitionSpecs` objects
  or `None` representing the sharding constraints for the arguments after the
  function begines. This is useful if we want to input data into the function
  in one sharding, and then reshard it inside the function. `None` means that
  the sharding is inferred by XLA.
- `annotation_shardings`: A dictionary mapping annotation names to `ShardingRule`
  or `PartitionSpecs` objects. This is used to specify the shardings for
  annotations specificied using `scalax.sharding.with_sharding_annotation`.

Returns
- A compiled function.


### Method `match_sharding_rule`
`match_sharding_rule` is a method to apply a sharding rule to a pytree under the
current mesh.

Parameters
- `sharding_rules`: A `ShardingRule` or `PartitionSpec` object to be applied to
  the pytree. It can also be a pytree of `ShardingRule` or `PartitionSpec` objects.
- `pytree`: The pytree to be sharded.

Returns
- A pytree of concrete `NamedSharding` objects, which has the same structure as
  the input pytree.


### Method `local_data_to_global_array`
`local_data_to_global_array` is a method to convert host local data to global
`jax.Array`. This is useful for data loading. We assume that each host loads a
portion of the data, and the data should be concatenated along a given axis to
form a global batch array.

Parameters
- `pytree`: The pytree of local data.
- `batch_axis`: The axis along which the data should be concatenated. Default to 0.
- `mesh_axis_subset`: A list of mesh names representing the subset of dimensions
  the data should be sharded agaist. Default to `None`, which means that the data
  is sharded against all dimensions.

Returns
- A global `jax.Array` or pytree of global `jax.Array` representing the global
  batch.


## Function `with_sharding_constraint`
Similar to `jax.lax.with_sharding_constraint`, but takes `ShardingRule` and
`PartitionSpec` objects. This functions enforces a sharding constraint inside
a sjit'ed function.

Parameters
- `pytree`: The pytree to be sharded.
- `sharding_rule`: A `ShardingRule` or `PartitionSpec` object to be applied to
  the pytree. It can also be a pytree of `ShardingRule` or `PartitionSpec` objects.

Returns
- The sharded pytree, with the same structure as the input pytree.

## Function `with_sharding_annotation`
This function provides a named sharding annotation for a pytree. The concrete
sharding for this annotated pytree can be then specified using the `annotation_shardings`
parameter of `sjit`.

Paramters
- `pytree`: The pytree to be annotated.
- `sharding_name`: The string name of the sharding annotation.

Returns
- The sharded pytree, with the same structure as the input pytree.



## Class `ShardingRule`
`ShardingRule` is a base class for sharding rules. It provides a `apply` method
to apply the sharding rule to a pytree.


## Class `FSDPShardingRule`
`FSDPShardingRule` is a sharding rule for Fully Sharded Data Parallelism (FSDP).
It analyzes the shape of the tensors in the pytree and finds a suitable axis for
FSDP sharding.

### Constructor
- `fsdp_axis_name`: The name of the FSDP axis
- `fsdp_axis_size`: The size of the FSDP axis. `FSDPShardingRule` will find an
  axis of a tensor which can be divided by `fsdp_axis_size` and use that as the
  FSDP axis. If it is None, `FSDPShardingRule` will find the axis that has the
  largest power of 2 divisor and use that as the FSDP axis.
- `min_fsdp_size`: The minimum size of tensors to be sharded. If the size of a
  tensor is smaller than `min_fsdp_size`, it will not be sharded.
### Method `apply`
Applies FSDP sharding rule to a the pytree.

Parameters
- `pytree`: The pytree to be sharded.

Returns
- A pytree of `NamedSharding` objects, which has the same structure as the input pytree.


## Class `TreePathShardingRule`
`TreePathShardingRule` is a sharding rule for sharding a pytree using the tree
path of its leaves. It is a flexible sharding rule that allows the user to specify a combination of different shardings.

### Constructor
- `*rules`: regex sharding rules to be applied to the leaves of the pytree. Each
  sharding rule should be a pair of a regex pattern and a `PartitionSpec`
  object. `TreePathShardingRule` will iterate through the rules in order and apply
  the `PartitionSpec` with the matching pattern.
- `strict`: Default to `True`. If `True`, `TreePathShardingRule` will raise an
  error if there is a leaf in the pytree that does not match any of the patterns.
  If `False`, `TreePathShardingRule` will simply provide a replicated `PartitionSpec`
  for the unmatched leaf.

### Method `apply`
Applies the sharding rule to the pytree.

Parameters
- `pytree`: The pytree to be sharded.

Returns
- A pytree of `NamedSharding` objects, which has the same structure as the input pytree.



## Class `PolicyShardingRule`
`PolicyShardingRule` is a sharding rule for sharding a pytree using a user-defined
policy function.

### Constructor
- `policy`: The policy function to be used for sharding. The policy function
  should take the pytree path and the leaf tensor and return a `PartitionSpec` object.

### Method `apply`
Applies the sharding rule to the pytree.

Parameters
- `pytree`: The pytree to be sharded.

Returns
- A pytree of `NamedSharding` objects, which has the same structure as the input pytree.
