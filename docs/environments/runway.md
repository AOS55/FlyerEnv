(environments-runway)=

# Runway

In this task the aircraft needs to land on a runway, specified within a configuration. Reward is sparse and provided based on whether the aircraft successfully lands on the airfield.

## Usage
```python
env = gym.make("runway-v1")
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
    "landing_reward": 100.0,  # max +ve reward for landing successfully
    "normalize_reward": True,  # whether to normalize the reward
    "runway_configuration": {
        "runway_position": [0.0, 0.0],
        "runway_width": 20.0,
        "runway_length": 1500.0,
        "runway_heading": 0.0
    }  # trajectory configuration details
}
```
Specifically this is defined in:

```{eval-rst}
.. automethod:: flyer_env.envs.runway_env.RunwayEnv.default_config
```

## API

```{eval-rst}
.. autoclass:: flyer_env.envs.runway_env.RunwayEnv
    :members:
```