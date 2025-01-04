---
hide-toc: true
firstpage:
lastpage:
---

```{project-heading}
FlyerEnv is a flight simulator environment for reinforcement learning. It connects a Python Gym interface to a Rust-based physics engine, allowing AI agents to learn flight control.
```

The environments are designed to be modular and customizable, allowing users to create a wide variety of tasks and scenarios. The environments are built on top of the [Gymnasium](https://gymnasium.farama.org) framework, making them easy to integrate with existing reinforcement learning algorithms and tools.

The purpose of this documentation is to provide:

1. a {ref}`quick start guide <quickstart>` describing the environments and customization options.
2. a {ref}`detailed description <user_guide>` describing core project components and a guide to contributing.

(index-how-to-cite-this-work)=

# How to cite this work?

If you use this package, please consider citing it:

```bibtex
@misc{flyer-env,
    author={Quessy, Alexander},
    title={FlyerEnv: an Environment for Autonomous Fixed-Wing Guidance, Navigation and Control Tasks},
    year={2023},
    publisher={GitHub},
    journal={GitHub repository},
    howpublished={\url{https://github.com/AOS55/flyer-env}},
}
```

```{toctree}
:hidden:
:caption: Introduction

introduction/basic_usage
introduction/installation
introduction/architecture
```

```{toctree}
:hidden:
:caption: API

api/environment
api/observation
api/action
api/reward
api/termination
api/rendering
```

```{toctree}
:hidden:
:caption: Environments

environments/flyer
envrionments/control
environments/trajectory
environments/forced_landing
environments/runway
```


```{toctree}
:hidden:
:caption: Development

Github <https://github.com/Farama-Foundation/FlyerEnv>
Contribute to the Docs <https://github.com/Farama-Foundation/FlyerEnv/blob/main/docs/README.md>
```