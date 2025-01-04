---
layout: "contents"
title: Basic Usage
firstpage:
---

# Basic Usage

The creation and interaction with `flyer_env` is:

```python
import gymnasium as gym
import flyer_env

env = gym.make("flyer-v1", render_mode="human")
observation, info = env.reset(seed=42)
for _ in range(1000):
   action = policy(observation)  # User-defined policy function
   observation, reward, terminated, truncated, info = env.step(action)

   if terminated or truncated:
      observation, info = env.reset()
env.close()
```