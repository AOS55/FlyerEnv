import gymnasium as gym
import flyer_env
import time
import numpy as np

flyer_env.register_flyer_envs()

def main():
    env = gym.make("flyer_control-v1",
        render_mode="rgb_array",
        control_type="altitude",
        target_value=500.0,
        tolerance=10.0
    )

    obs, info = env.reset()

    # FPS tracking variables
    frame_count = 0
    start_time = time.time()
    last_time = start_time
    fps_buffer = []

    # Print FPS every N frames
    fps_print_interval = 100

    for _ in range(10000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            obs, info = env.reset()

        # FPS calculation
        frame_count += 1
        if frame_count % fps_print_interval == 0:
            current_time = time.time()
            # Instantaneous FPS (last N frames)
            inst_fps = fps_print_interval / (current_time - last_time)
            fps_buffer.append(inst_fps)
            # Average FPS
            avg_fps = frame_count / (current_time - start_time)

            print(f"Frame {frame_count:5d} | Instantaneous FPS: {inst_fps:3.1f} | Average FPS: {avg_fps:3.1f}")
            last_time = current_time

    # Final statistics
    total_time = time.time() - start_time
    final_avg_fps = frame_count / total_time
    if fps_buffer:
        print("\nFinal Statistics:")
        print(f"Total Frames: {frame_count}")
        print(f"Total Time: {total_time:.2f} seconds")
        print(f"Average FPS: {final_avg_fps:.1f}")
        print(f"Min FPS: {min(fps_buffer):.1f}")
        print(f"Max FPS: {max(fps_buffer):.1f}")

if __name__ == "__main__":
    main()
