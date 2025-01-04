---
layout: "contents"
title: Installation
firstpage:
---

# Installation

## Prerequisites

FlyerEnv relies on Python 3.8 or higher and Rust 2021 or higher.

We recommend using a [conda environment](https://docs.anaconda.com/miniconda/miniconda-install/) to manage dependencies. You can create a new environment with the following command:

```bash
conda create --name conda-flyer python=3.8
conda activate conda-flyer
```

To install Rust we recommend you use [rustc](https://doc.rust-lang.org/book/ch01-01-installation.html) if not already installed:

```bash
curl --proto '=https' --tlsv1.2 https://sh.rustup.rs -sSf | sh
```

#### Standard Installation

```bash
pip install flyer-env
```

#### Development Installation

If you want to modify the library, clone the repository and setup a development environment.

There are 3 used to build the project:
- The core Rust flight simulator `flyer-rs`.
- The Server and interface to Python `pyflyer-rs`.
- The `flyer_env` library is written in Python.

Build the flyer-rs and pyflyer-rs libraries using `cargo build`.

To setup the Python environment, run the following commands:

```bash
git clone https://github.com./flyer-env.git
pip install -e .
```
