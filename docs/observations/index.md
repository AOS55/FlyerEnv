(observations)=

% py:currentmodule::flyer_env.envs.common.observation

# Observations

For all environments, a variety of observation types can be used. These are defined with the {py:mod}`~flyer_env.
envs.common.observation` module. Each environment has a *default* observation, which can be changed or customized 
using {ref}`environment configurations <configuration>`. For example: 

```python
import gymnasium as gym
import flyer_env

env = gym.make('flyer-v1')
env.configure({
    'observation': {
        
    }
})
env.reset()
```

<!-- # Kinematics

The {py:class}`~flyer_env.envs.common.observation.KinematicObservation` is a $V \times F$ array containing a list of 
$V$ nearby vehicles by a set of features of size $F$, listed in the `"features"` configuration field. For example:

| Name      |     Param    |             Description             |   Unit  |
|-----------|:------------:|:-----------------------------------:|:-------:|
|    `x`    |      $x$     | Aircraft's position in the $x$-axis |  [$m$]  |
|    `y`    |      $y$     | Aircraft's position in the $y$-axis |  [$m$]  |
|    `z`    |      $z$     | Aircraft's position in the $z$-axis |  [$m$]  |
| `heading` |    $\psi$    |       Aircraft's heading angle      | [$rad$] |
|  `pitch`  |   $\theta$   |        Aircraft's pitch angle       | [$rad$] |
|   `bank`  |    $\phi$    |        Aircraft's bank angle        | [$rad$] |
|  `speed`  | $V_{\infty}$ |         Aircraft's airspeed         | [$m/s$] |

# Dynamics

| Name    |   Param  |                   Description                  |    Unit   |
|---------|:--------:|:----------------------------------------------:|:---------:|
|   `x`   |    $x$   |       Aircraft's position in the $x$-axis      |   [$m$]   |
|   `y`   |    $y$   |       Aircraft's position in the $y$-axis      |   [$m$]   |
|   `z`   |    $z$   |       Aircraft's position in the $z$-axis      |   [$m$]   |
| `pitch` | $\theta$ |             Aircraft's pitch angle             |  [$rad$]  |
|  `roll` |  $\phi$  |              Aircraft's roll angle             |  [$rad$]  |
|  `yaw`  |  $\psi$  |              Aircraft's yaw angle              |  [$rad$]  |
|   `u`   |    $u$   |   Aircraft's linear velocity in the $x$-axis   |  [$m/s$]  |
|   `v`   |    $v$   |   Aircraft's linear velocity in the $y$-axis   |  [$m/s$]  |
|   `w`   |    $w$   |   Aircraft's linear velocity in the $z$-axis   |  [$m/s$]  |
|   `p`   |    $p$   | Aircraft's rotational velocity in the $x$-axis | [$rad/s$] |
|   `q`   |    $q$   | Aircraft's rotational velocity in the $y$-axis | [$rad/s$] |
|   `r`   |    $r$   | Aircraft's rotational velocity in the $z$-axis | [$rad/s$] | -->


