import gymnasium as gym


def main():
    env = gym.make("flyer-v1", render_mode="rgb_array")
    env.config['action'] = {'type': 'HeadingAction'}
    v_env = gym.wrappers.RecordVideo(env, 'videos', step_trigger = lambda x: x % 100 == 0)
    # v_env = env
    v_env.reset()
    idr = 0
    for _ in range(100000):
        action = 0.5
        obs, reward, terminated, truncated, info = v_env.step(action)
        # if idr == 100:
        #     v_env.render()
        #     idr = 0
        # else:
        #     idr += 1


if __name__ == "__main__":
    main()
