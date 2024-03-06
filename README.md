# Scalax: scaling utilities for JAX (or scale and relax)
Scalax is a collection of utilties for helping developers to easily scale up
JAX based machine learning models. The main idea of scalax is pretty simple:
users write model and training code for a single GPU/TPU, and rely on scalax to
automatically scale it up to hundreds of GPUs/TPUs. This is made possible by
the JAX jit compiler, and scalax provides a set of utilities to help the users
obtain the sharding annotations required by the jit compiler. Because scalax
wraps around the jit compiler, existing JAX code can be easily scaled up using
scalax with minimal changes.

Scalax came out of our experience building [EasyLM](https://github.com/young-geng/EasyLM),
a scalable language model training library built on top of JAX.


## Installation
Scalax is available on PyPI and can be installed using pip:
```bash
pip install scalax
```


## Documentations
More details about scalax can be found in the [documentation page](docs/).


## Discord Server
We are running an unofficial Discord community (unaffiliated with Google) for discussion related to training large models in JAX. [Follow this link to join the Discord server](https://discord.gg/Rf4drG3Bhp). We have dedicated channel for scalax.


## Examples
We provide a set of well annotated examples in the [examples directory](/examples/). The most notable ones include:
- [MLP with Data Parallelism](/examples/mnist_data_parallel.ipynb)
- [LLaMA with fully sharded data parallelism](/examples/llama_fsdp.ipynb)
- [LLaMA with combined fully sharded data parallelism and tensor parallelism](/examples/llama_fsdp_tp.ipynb)


## Quickstart
Suppose we have a simple flax model and train step function:

```python
class Model(nn.Module):
    ...


def train_step(train_state, batch):
    ...
    return updated_train_state, metrics
```

Typically, we would use jax.jit to compile the train_step function in order to
accelerate the training:

```python
@jax.jit
def train_step(train_state, batch):
    ...
    return updated_train_state, metrics
```

This works fine for a single GPU/TPU, but if we want to scale up to multiple
GPU/TPUs, we need to partition the data or the model in order to parallelize
the training across devices. Fortunately, JAX JIT already provides a way to
handle these partitions with [sharding annotations](https://jax.readthedocs.io/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html).
For example, if we have sharding annotations for the `train_state` and `batch`
pytree, we can simply JIT compile the train_step function with these sharding
annotations:

```python
@partial(
    jax.jit,
    in_shardings=(train_state_shardings, batch_sharding),  # Shard the train_state
    out_shardings=(train_state_shardings, None),
)
def train_step(train_state, batch):
    ...
    return updated_train_state, metrics
```
The `train_state_shardings` and `batch_sharding` are pytrees having the same
structure as the `train_state` and `batch` pytrees, but with `jax.sharding.Sharding`
objects at the leaf nodes. These sharding objects are tied to the physical device
mesh and are often difficult to construct, especially for complex models and
training code. This is where scalax comes in. Scalax provides a set of utilities
to help the users automatically obtain the sharding annotations, without having
to worry about the underlying pytree structure. Scalax handles this by abstracting
away the concrete sharding objects and using a `ShardingRule` object instead. A
`ShardingRule` object can generate the sharding annotations for any given pytree
according to its internal rules.


For example, scalax provides a `FSDPShardingRule` object, which can automatically
generate sharding annotations for a pytree according to the Fully Sharded Data
Parallelism (FSDP) strategy. To apply it to our `train_step` function, we can
simply replace the `jax.jit` decorator:

```python
from functools import partial
from scalax.sharding import MeshShardingHelper, PartitionSpec, FSDPShardingRule


mesh = MeshShardingHelper([-1], ['fsdp'])  # Create a 1D mesh with fsdp axis
@partial(
    mesh.sjit,
    in_shardings=(FSDPShardingRule(), None),   # Shard the train_state using FSDP
    out_shardings=(FSDPShardingRule(), None),
    args_sharding_constraint=(FSDPShardingRule(), PartitionSpec('fsdp')),
)
def train_step(train_state, batch):
    ...
    return updated_train_state, metrics
```

In the previous example, we see that scalax provides a `MeshShardingHelper` object
using a 1D device mesh with a fsdp axis. We then use the `sjit` method to compile
the `train_step` function with the FSDP sharding rules, without having to worry
about the specific underlying pytree structure of the `train_state`. Beyond
FSDP, scalax also provides `TreePathShardingRule` and `PolicyShardingRule`, which
allows users to easily combine different sharding strategies such as replicated
data parallelism, FSDP, tensor parallelism and sequence parallelism to best fit
their model and training setup. All of these can be done with minimal changes to
the original model and training code.  This makes it easy to integrate scalax
into existing JAX codebases.


Scalax currently supports the following sharding rules:
- `FSDPShardingRule`: A sharding rule for automatically selecting an axis for
  Fully Sharded Data Parallelism (FSDP).
- `TreePathShardingRule`: A regular expression sharding rule for sharding a pytree
    according to the path of its leaves.
- `PolicyShardingRule`: A sharding rule which determins the sharding according to
    a user defined callable policy.


### Sharding Intermediate Tensors
In previous example, we see that scalax `sjit` can help us easily shard the
input and output of a jitted function. In many cases, this would be sufficient
to scale up the training, as the intermdiate tensors are automatically sharded
by XLA. However, in some cases, XLA might not be able to derive the optimal
sharding for the intermediate tensors, and we might want to manually specify
the sharding for these tensors. Similar to JAX, scalax  provides a
`with_sharding_constraint` function to manually specify the sharding.
Similar to `sjit`, `with_sharding_constraint` takes both `ShardingRule` and
`PartitionSpec` objects.

```python
from scalax.sharding import MeshShardingHelper, PartitionSpec, FSDPShardingRule
from scalax.sharding import with_sharding_constraint

mesh = MeshShardingHelper([-1], ['fsdp'])  # Create a 1D mesh with fsdp axis
@partial(
    mesh.sjit,
    in_shardings=(FSDPShardingRule(), None),   # Shard the train_state using FSDP
    out_shardings=(FSDPShardingRule(), None),
    args_sharding_constraint=(FSDPShardingRule(), PartitionSpec('fsdp')),
)
def train_step(train_state, batch):
    ...
    intermediate_pytree = ...
    intermediate_pytree = with_sharding_constraint(
        intermediate_pytree, FSDPShardingRule(),
    )
    ...
    return updated_train_state, metrics
```

In previous example, we apply the `FSDPShardingRule` to the `intermediate_pytree`.
However, this way of sharding intermediate tensors is intrusive to the original
training code. To make it easier to shard intermediate tensors, scalax provides
a `with_sharding_annotation` function, which only register a name for the sharding
within the training code without tieing it to a concate sharding rule. This allows
the same model and training code to be sharded differently without changing the
code. For example:

```python
from scalax.sharding import MeshShardingHelper, PartitionSpec, FSDPShardingRule
from scalax.sharding import with_sharding_annotation

mesh = MeshShardingHelper([-1], ['fsdp'])  # Create a 1D mesh with fsdp axis
@partial(
    mesh.sjit,
    in_shardings=(FSDPShardingRule(), None),   # Shard the train_state using FSDP
    out_shardings=(FSDPShardingRule(), None),
    args_sharding_constraint=(FSDPShardingRule(), PartitionSpec('fsdp')),
    annotation_shardings={
        'weights': FSDPShardingRule(),
        'activations': PartitionSpec('fsdp'),
    }
)
def train_step(train_state, batch):
    ...
    weights_pytree = with_sharding_annotation(
        weights_pytree, 'weights',
    )
    activations = with_sharding_annotation(
        activations, 'activations',
    )
    ...
    return updated_train_state, metrics
```
