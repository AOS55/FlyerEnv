(make-your-own)=

# Make your own environment

The following are steps required to create a new environment!

## Set up files

1. Create a new `<my_env>.py` file in `flyer_env/envs/`
2. Define a class MyEnv, that *must* inherit {py:class}`~flyer_env.envs.common.abstract.AbstractEnv`

This class provides several useful functions:

- A {py:meth}`~flyer_env.envs.common.abstract.AbstractEnv.default_config` method, that provides a default 
  configuration dictionary that can be overloaded.
- A {py:meth}`~flyer_env.envs.common.abstract.AbstractEnv.define_spaces` method, that provides access to observation 
  and action types, set within the environment configuration.
- A {py:meth}`~flyer_env.envs.common.abstract.AbstractEnv.step` method, which executes the commanded action at the 
  `policy_frequency` and simulate the environment at the `simulation_frequency`. 

## Create the world

Initially create a {py:class}`flyer_env.world.world.World` which is a generic container for:
- **Terrain**: the base ground geometry
- **Vehicles**: dynamic objects, that principally navigate around the world with aircraft dynamics
- **StaticObjects**: Ground objects that can be collided into when the aircraft lands

The terrain network is described in ...
Construction of this should be achieved with `MyEnv._make_world()` called from MyEnv.reset() to populate the `self.
world` field.

## Create the aircraft

Populate the world with aircraft. This should be done with `MyEnv._make_aircraft()` called from `MyEnv.reset()` to 
set the `self.world.vehicles` list of aircraft which are either based on:

1. **FlyingVehicle**: a kinematic model for the aircraft described in ...
2. **AircraftVehicle**: a full non-linear parameterized aircraft model described in ...

The controlled ego-vehicle is defined by setting `self.vehicle`, the class depends upon `self.action_type.
vehicle_class`. Other vehicles can be added freely to the `self.world.vehicles` list.

## Make the environment configurable

To make part of the environment configurable overload the default configuration in {py:meth}`~flyer_env.envs.common.
abstract.AbstractEnv.default_config` to define new `{"config_key": value}` pairs with default values. These 
configurations can then be accessed within the environment implementation `self.config["config_key]`, and once 
created configured with `env.configure({"config_key": other_value})` followed by `env.reset()`.

## Register the environment

In `flyer_env/envs/__init__.py`, add the following line:

```python
register(
  id='my-env-v0',
  entry_point='flyer_env.envs:MyEnv'
)
```

## Test the environment

You should now be able to run the environment:

```python
import gymnasium as gym

env = gym.make('my-env-v0')
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
env.render()
```

## API

```{eval-rst}
..automodule:: flyer_env.envs.common.abstract
  :members:
  :private-members:
```
