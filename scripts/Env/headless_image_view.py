import matplotlib.pyplot as plt
import flyer_env
import gymnasium as gym
flyer_env.register_flyer_envs()

def display_image(image):
    """Display an image using matplotlib."""
    plt.imshow(image)
    plt.axis('off')
    plt.show()

def main():
    env = gym.make("flyer_control-v1",
        render_mode="rgb_array",
        control_type="altitude",
        target_value=500.0,
        tolerance=10.0
    )
    env.reset()
    image = env.render()
    display_image(image)


if __name__=="__main__":
    main()
