# 🏗️FlyerEnv🏗️ (Under Construction)

[![build](https://github.com/AOS55/FlyerEnv/actions/workflows/build.yml/badge.svg?branch=master)](https://github.com/AOS55/FlyerEnv/actions/workflows/build.yml)

## The environments

### Point Navigation

```python
env = gymnasium.make("flyer-v1")
```

In this task the aircraft must navigate to a specified 3D goal point in space in the shortest possible time.

### Trajectory Following

```python
env = gymnasium.make("trajectory-v1")
```

In this task the aircraft must follow a trajectory created by a moving target, maintaining on target maximizes the reward. The possible trajectory primitives are as follows:
- `sl`, maintain straight and level flight.
- `climb`, climb to a specified level at a given climb angle.
- `descend`, descend to a specified level at a given descent angle.
- `lt`, turn left to a specified heading at a given rate.
- `rt`, turn right to a specified heading at a given rate.


### Runway Landing

```python
env = gymnasium.make("runway_landing-v1")
```

### Forced Landing

```python
env = gymnasium.make("forced_landing-v1")
```

## Installation

```pip install flyer-env```

## Usage

```python
import gymnasium as gym

env = gym.make("flyer-v1", render_mode="human")

done = truncated = False
while not (done or truncated):
    action = ... # Your agent code goes here
    obs, reward, done, truncated, info = env.step(action)
```

## Documentation

TODO: add the documentation

## Citing

If you use the project in your work, please consider citing it with:

```text
@misc{flyer-env,
    author={Quessy, Alexander},
    title={An Environment for Autonomous Fixed-Wing Guidance, Navigation and Control Tasks},
    year={2023},
    publisher={GitHub},
    journal={GitHub repository},
    howpublished={\url{https://github.com/AOS55/flyer-env}},
}
```
