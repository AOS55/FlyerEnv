(quickstart)=
# Getting Started

## Making an environment

Here is a quick example of how to create an environment:

<!-- ```{eval-rst}
.. jupyter-execute::

    import gymnasium as gym
    from matplotlib import pyplot as plt
    %matplotlib inline
    
    env = gym.make('flyer-v1', render_mode='rgb_array')
    env.reset()
    for _ in range(3):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        env.render()
        
    plt.imshow(env.render())
    plt.show()
``` -->

### All the environments

The following is a list of all the environments available and their descriptions:

```{toctree}
:maxdepth: 1

environments/flyer
environments/trajectory
environments/runway
environments/forced_landing
```

(configuration)=

## Configuring an environment

The {ref}`observations <observations>`, {ref}`actions <actions>`, {ref}`dynamics <dynamics>` and {ref}`rewards 
<rewards>` of an environment are parametrized by the configuration {py:attr}`~flyer_env.envs.common.abstract.
AbstractEnv.config` dictionary. After environment creation, the configuration can be accessed using the {py:attr}
`~flyer_env.envs.common.abstract.AbstractEnv.config` attribute.

<!-- ```{eval-rst}
.. jupyter-execute::

    import pprint
    
    env = gym.make('flyer-v1', render_mode='rgb_array')
    pprint.pprint(env.config)
``` -->

```{note}
The environment must be {py:meth}`~flyer_env.envs.common.abstract.AbstractEnv.reset` for the change in configuration 
to have effect.
```

## Training an agent

Here is an example using ... to train ... with default kinematics and an *MLP model*.

[colab-badge]: https://colab.research.google.com/assets/colab-badge.svg



