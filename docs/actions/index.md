---
_actions:
---

<!-- ```{eval-rst}
.. py:module:: flyer_env.envs.common.action
``` -->

# Actions

<!-- Similarly to {ref}`observations <observations>`, a variety of action types can be used within each environment. 
These are defined in the {py:mod}`~flyer_env.envs.common.action` module. -->
Each environment comes with a *default* action type, that can be customized using {ref}`environment configurations 
<configuration>`. As an example:

```python
import gymnasium as gym

env = gym.make('flyer-v1')
env.configure({
    "action": {
        "type": "ContinousAction"
    }
})
env.reset()
```

## Continuous Actions

The {py:class}`~flyer_env.envs.common.action.ContinuousAction` type allows for direct control of the agents controls 
either with a {ref}`FlyingVehicle <vehicle_kinematics>` or {ref}`AircraftVehicle <vehicle_dynamics>`. These 
commands include:

| Control Surface              |    Nomenclature     |    Range    | Unit |
|------------------------------|:-------------------:|:-----------:|:----:|
| **Elevator**                 |   $\delta_{\eta}$   | [-1.0, 1.0] |  [-] |
| **Aileron**                  | $\delta_{\epsilon}$ | [-1.0, 1.0] |  [-] |
| **Thrust Lever Angle (TLA)** |   $\delta_{tla}$    |  [0.0, 1.0] |  [-] |

## Controlled Actions

The {py:class}`~flyer_env.envs.common.action.ControlledAction` type is a higher order action type to control the 
aircraft's heading, altitude, and speed with a {ref}`FlyingVehicle <vehicle_kinematics>`. These commands include:

| Controlled Parameter | Nomenclature |      Range     |   Unit   |
|----------------------|:------------:|:--------------:|:--------:|
| **Heading**          |   $\theta$   | [-$\pi$,$\pi$] | [$rads$] |
| **Altitude**         |     $H$      |  [0.0, 10,000] |   [$m$]  |
| **Airspeed**         | $V_{\infty}$ |  [0.0, 300.0]  |  [$m/s$] |

## Pursuit Actions

The {py:class}`~flyer_env.envs.common.action.PursuitAction` type is a higher order action type used to navigate a 
{ref}`FlyingVehicle <vehicle_kinematics>` to a specific 2D point at a fixed altitude and speed. These commands include:

| Controlled Parameter | Nomenclature |     Range     |   Unit  |
|----------------------|:------------:|:-------------:|:-------:|
| **GoalPos**          |   $s_{g}$    |      [-]      |  [$m$]  |
| **Altitude**         |     $H$      | [0.0, 10,000] |  [$m$]  |
| **Airspeed**         | $V_{\infty}$ |  [0.0, 300.0] | [$m/s$] |
