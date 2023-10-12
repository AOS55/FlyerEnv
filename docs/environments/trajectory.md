(environments-trajectory)=

# Trajectory

In this task the aircraft needs to follow a configuration specified by the track controller. Reward is provided based on the $L_{2}$ Euclidian distance from the ideal position at time $\mathbf{s}_{t}$. The task is intended to investigate non-linear control problems. 

## Usage
```python
env = gym.make("trajectory-v1")
```

## Default Configuration

```python
{
    "observation": {
        "type": "Dynamics"
    },
    "action": {
        "type": "ContinuousAction"
    },
    "area": (1024, 1024),  # terrain map area [tiles]
    "vehicle_type": "Dynamic",  # vehicle type, only dynamic available
    "duration": 10.0,  # simulation duration [s]
    "collision_reward": -200.0,  # max -ve reward for crashing
    "traj_reward": 10.0,  # max +ve reward for reaching for following trajectory
    "normalize_reward": True,  # whether to normalize the reward [-1, +1]
    "trajectory_config": {
        "name": "climb",  
        "final_height": 200.0,
        "climb_angle": 10.0 * np.pi / 180.0,
        "length": 15.0,
    }  # trajectory configuration details
}
```

Specifically this is defined in:

```{eval-rst}
.. automethod:: flyer_env.envs.trajectory_env.TrajectoryEnv.default_config
```

## API

```{eval-rst}
.. autoclass:: flyer_env.envs.trajectory_env.TrajectoryEnv
    :members:
```
