import gymnasium as gym


def main():
    env = gym.make("flyer-v1", render_mode="human")
    env.config['action'] = {'type': 'HeadingAction'}
    print(f'config:{env.config}')
    env.reset()
    idr = 0
    for _ in range(100000):
        action = env.action_space.sample()
        # action = [0.5, 0.5, 1.0]
        action = 0.5
        # print(f'action: {action}')
        obs, reward, terminated, truncated, info = env.step(action)
        # print(f'obs: {obs}')
        if idr == 100:
            env.render()
            idr = 0
        else:
            idr += 1


if __name__ == "__main__":
    main()
