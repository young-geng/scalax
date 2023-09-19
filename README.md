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

## Quickstart
Suppose we have a simple flax model and train step function that looks like this:
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
the training across devices. This is where scalax comes in. We can first create
a device mesh and then replace the `jax.jit` decorator with `mesh.sharded_jit`.
To use different parallelization strategies, we can provide different sharding
rules to the `sharded_jit` function. For example, to change the previous example
into a data parallel training, we can do the following:

```python
from functools import partial
from scalax.sharding import MeshShardingHelper, PartitionSpec


mesh = MeshShardingHelper([-1], ['dp'])  # Create a 1D mesh with data parallelism axis
@partial(
    mesh.sharded_jit,
    in_shardings=None,
    out_shardings=None,
    # constraint the batch argument to be sharded along the dp axis to enable data parallelism
    args_sharding_constraint=(PartitionSpec(), PartitionSpec('dp')),
)
def train_step(train_state, batch):
    ...
    return updated_train_state, metrics

```

In this example, the model weights are replicated across all devices, and the
data batch is sharded across the dp axis. This works well if the model fits into
a single device. If the model is too large to fit into a single device, we can
use fully sharded data parallelism to also partition the model across devices:

```python
from functools import partial
from scalax.sharding import MeshShardingHelper, PartitionSpec, FSDPShardingRule


mesh = MeshShardingHelper([-1], ['fsdp'])  # Create a 1D mesh with data parallelism axis
@partial(
    mesh.sharded_jit,
    in_shardings=(FSDPShardingRule(), None),   # Shard the train_state using FSDP
    out_shardings=(FSDPShardingRule(), None),
    args_sharding_constraint=(FSDPShardingRule(), PartitionSpec('fsdp')),
)
def train_step(train_state, batch):
    ...
    return updated_train_state, metrics

```

As we can see in the previous example, scalax allows user to shard the model
and training without having to change the model or training code. This makes it
easy to integrate scalax into existing JAX codebases.

