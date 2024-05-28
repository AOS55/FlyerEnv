import gymnasium as gym


def main():
    env = gym.make("flyer-v1", render_mode="rgb_array")
    # env.config['action'] = {'type': 'HeadingAction'}
    v_env = gym.wrappers.RecordVideo(env, 'videos')
    v_env.reset()
    idr = 0
    done = False
    while not done:
        action = v_env.action_space.sample()
        # print(f'action sample: {action}')
        # action = 0.5
        obs, reward, terminated, truncated, info = v_env.step(action)
        done = terminated or truncated
        # print(f'terminate: {terminated, env.vehicle.crashed}')
        # env.render()
        # if idr == 1:
        #     env.render()
        #     idr = 0
        # else:
        #     idr += 1
    v_env.close()

if __name__ == "__main__":
    main()
