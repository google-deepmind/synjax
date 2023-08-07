# SynJax

[**What is SynJax?**](#what-is-synjax)
| [**Installation**](#installation)
| [**Examples**](#examples)
| [**Citing SynJax**](#citing-synjax)

## What is SynJax?<a id="what-is-synjax"></a>

SynJax is a neural network library for [JAX] structured probability
distributions. The distributions that are currently supported are:

* [Linear Chain CRF](https://github.com/deepmind/synjax/tree/master/synjax/_src/linear_chain_crf.py),
* [Semi-Markov CRF](https://github.com/deepmind/synjax/tree/master/synjax/_src/semi_markov_crf.py),
* [Constituency Tree CRF](https://github.com/deepmind/synjax/tree/master/synjax/_src/constituency_tree_crf.py),
* [Spanning Tree CRF](https://github.com/deepmind/synjax/tree/master/synjax/_src/spanning_tree_crf.py) -- including optional constraints for projectivity, (un)directionality and single-rootness,
* [Alignment CRF](https://github.com/deepmind/synjax/tree/master/synjax/_src/alignment_crf.py) -- including 1-to-many, many-to-many and other more complex monotonic alignment formsr,
* [PCFG](https://github.com/deepmind/synjax/tree/master/synjax/_src/constituency_pcfg.py),
* [Tensor-Decomposition PCFG](https://github.com/deepmind/synjax/tree/master/synjax/_src/constituency_tensor_decomposition_pcfg.py),
* [HMM](https://github.com/deepmind/synjax/tree/master/synjax/_src/hmm.py),
* [CTC](https://github.com/deepmind/synjax/tree/master/synjax/_src/ctc.py).

All these distributions support standard operations such as computing log-probability of a structure, computing marginal probability of a part of the structure, finding most likely structure, sampling, top-k, entropy, cross-entropy, kl-divergence...

All operations support standard JAX transformations `jax.vmap`, `jax.jit`, `jax.pmap` and `jax.grad`. The only exception are argmax, sample and top-k that do not support `jax.grad`.

## Installation<a id="installation"></a>

SynJax is written in pure Python, but depends on C++ code via JAX.
Because JAX installation is different depending on your CUDA version,
SynJax does not list JAX as a dependency in `requirements.txt`.

First, follow [these instructions](https://github.com/google/jax#installation)
to install JAX with the relevant accelerator support.

Then, install SynJax using pip:

```bash
$ pip install git+https://github.com/deepmind/synjax
```

## Examples<a id="examples"></a>

The [notebooks directory](https://github.com/deepmind/synjax/tree/master/notebooks) contains examples of how Synjax works:

* Introductory notebook demonstrating SynJax functionalities. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/deepmind/synjax/blob/master/notebooks/introduction_to_synjax.ipynb)

## Citing SynJax<a id="citing-synjax"></a>

To cite this repository:

```
@software{synjax2023github,
  author = {Stanojevi\'{c}, Milo\v{s} and Sartran, Laurent},
  title = "{SynJax}",
  url = {http://github.com/deepmind/synjax},
  version = {2023.8.5},
  year = {2023},
}
```

In this bibtex entry, the version number is intended to be from
[`synjax/__init__.py`](https://github.com/deepmind/synjax/blob/main/synjax/__init__.py),
and the year corresponds to the project's open-source release.

[JAX]: https://github.com/google/jax
