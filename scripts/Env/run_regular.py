import gymnasium as gym
import flyer_env

flyer_env.register_flyer_envs()

def main():
    env = gym.make("flyer_control-v1",
        render_mode="rgb_array",
        seed=5,
        control_type="altitude",
        target_value=500.0,
        tolerance=10.0,
        use_full_aircraft=False,
        max_episode_steps=True
    )


    env.reset(seed=5)
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = truncated


if __name__ == "__main__":
    main()
