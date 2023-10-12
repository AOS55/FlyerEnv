(environments-forced_landing)=

# Forced Landing

In this taskt the aircraft must navigate to a safe ground landing location, by default the aircraft starts from altitude with a failed engine. Reward is sparse and provided when the aircraft finds a safe space to land. 

## Usage
```python
env = gym.make("forced_landing-v1")
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
    "duration": 500.0,  # simulation duration [s]
    "collision_reward": -200.0,  # max -ve reward for crashing
    "landing_reward": 100.0,  # max +ve reward for landing successfully
    "normalize_reward": True 
}
```

Specifically this is defined in:

```{eval-rst}
.. automethod:: flyer_env.envs.forced_landing_env.ForcedLandingEnv.default_config
```

## API

```{eval-rst}
.. autoclass:: flyer_env.envs.forced_landing_env.ForcedLandingEnv
    :members:
```
