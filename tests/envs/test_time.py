import timeit
import gymnasium as gym

def wrapper(func, *args, **kwargs):
    def wrapped():
        return func(*args, **kwargs)
    
    return wrapped


def time_env(env_name, steps=20):
    env = gym.make(env_name)
    env.reset()
    for _ in range(steps):
        _, _, done, truncated, _ = env.step(env.action_space.sample())
        env.reset() if done or truncated else _
    env.close()


def test_running_time(repeat=1):
    for env_name, steps in [
        ("flyer-v1", 100),
        ("trajectory-v1", 100),
        ("runway-v1", 100),
        ("forced_landing-v1", 100),
        ("control-v1", 100)
    ]:
        env_time = wrapper(time_env, env_name, steps)
        time_spent = timeit.timeit(env_time, number=repeat)
        env = gym.make(env_name)
        time_simulated = steps / env.unwrapped.config["policy_frequency"]
        real_time_ratio = time_simulated / time_spent
        print("Real time ratio for {}: {}".format(env_name, real_time_ratio))
        assert real_time_ratio > 0.5  # let's not be too ambitious for now


if __name__ == "__main__":
    test_running_time()
