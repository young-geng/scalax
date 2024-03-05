# Scalax Documentation
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

## Module in Scalax
- [Sharding](sharding.md)
- Utils
