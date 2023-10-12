(environments-flyer)=

# Flyer

In this task, the aircraft needs to be controlled and navigated towards a goal-state. Reward is provided based on the $L_{2}$ Euclidian distance from the goal-state up to a maximum reward. The task is intended to test goal-oriented control problems. 

## Usage

```python
env = gym.make("flyer-v1")
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
    "point_reward": 100.0,  # max +ve reward for hitting the goal
    "normalize_reward": True, # whether to normalize the reward [-1, +1]
    "goal_generation": {
        "heading_limits": [-np.pi, np.pi],
        "pitch_limits": [-10.0 * np.pi/180.0, 10.0 * np.pi/180.0],
        "dist_limits": [1000.0, 10000.0],
        "dist_terminal": 20.0
    }  # goal generation details
}
```

Specifically this is defined in:

```{eval-rst}
.. automethod:: flyer_env.envs.flyer_env.FlyerEnv.default_config
```

## API

```{eval-rst}
.. autoclass:: flyer_env.envs.flyer_env.FlyerEnv
    :members:
```
