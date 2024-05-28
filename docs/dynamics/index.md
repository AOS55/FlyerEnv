(dynamics)=

# Dynamics

The dynamics of each environment describes the motion of the vehicle. The dynamics of flying vehicles are simplified 
into 2 types of behaviors:

- Airborne dynamics, determined by either the kinematic or dynamics aircraft model
- Ground dynamics, simplified ground dynamics where the aircraft either crashes and stops or translates in a 
  straight direction over the ground. 

## World

A {py:class}`~flyer_env.world.world.World` is composed of a {py:class}`~flyer_env.terrain.terrain.TerrainMap` and a 
list of {py:class}`~flyer_env.aircraft.objects.DynamicObject` which are usually either {py:class}`~flyer_env.aircraft.
kinematics.FlyingVehicle` or {py:class}`~flyer_env.aircraft.dynamics.AircraftVehicle`.

```{toctree}
:maxdepth: 1

world/terrain
world/world

```

## Aircraft

```{toctree}
:maxdepth: 1

aircraft/kinematics
aircraft/dynamics
aircraft/controller

```
