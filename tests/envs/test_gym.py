# import pytest
# import numpy as np
# import subprocess
# import time
# from flyer_env.envs.common.abstract import AbstractEnv

# @pytest.fixture(scope="module")
# def running_server():
#     print("\n=== Starting server fixture ===")
    
#     # Start the actual Rust server
#     print("Launching Bevy server process...")
#     process = subprocess.Popen(
#         ["pyflyer-rs/target/release/bevy_server"],
#         stdout=subprocess.PIPE,
#         stderr=subprocess.PIPE,
#         text=True,
#         bufsize=1,
#         universal_newlines=True
#     )
#     print(f"Process started with PID: {process.pid}")
    
#     # Wait for server to start with timeout
#     start_time = time.time()
#     timeout = 30  # 30 seconds timeout
#     port = None
    
#     print("\nWaiting for server initialization...")
#     while time.time() - start_time < timeout:
#         if process.poll() is not None:
#             # Process ended prematurely
#             stdout, stderr = process.communicate()
#             raise RuntimeError(
#                 f"Server process ended unexpectedly (return code: {process.returncode}).\n"
#                 f"Stdout:\n{stdout}\nStderr:\n{stderr}"
#             )
            
#         line = process.stdout.readline()
#         if not line:
#             continue
            
#         print(f"Server output: {line.strip()}")
        
#         # Check for port number
#         if line.startswith("PORT="):
#             port = int(line.strip().split("=")[1])
#             print(f"Found server port: {port}")
        
#         # Check for server ready
#         if "Starting Bevy app..." in line:
#             if port is not None:
#                 print("Server initialization complete!")
#                 break
#             else:
#                 print("Warning: Got 'Starting Bevy app...' but no port yet")
#     else:
#         print("Timeout while waiting for server to start")
#         process.terminate()
#         stdout, stderr = process.communicate()
#         raise TimeoutError(
#             f"Server failed to start in {timeout} seconds.\n"
#             f"Stdout:\n{stdout}\nStderr:\n{stderr}"
#         )
    
#     print("\n=== Server ready, yielding to test ===\n")
#     yield process
    
#     # Cleanup
#     print("\n=== Starting server cleanup ===")
#     print("Terminating server process...")
#     process.terminate()
    
#     try:
#         print("Waiting for process to end...")
#         stdout, stderr = process.communicate(timeout=5)
#         print("Server final output:")
#         print(f"Stdout:\n{stdout}")
#         print(f"Stderr:\n{stderr}")
#     except subprocess.TimeoutExpired:
#         print("Server didn't terminate, killing process...")
#         process.kill()
#         stdout, stderr = process.communicate()
#         print("Final output after kill:")
#         print(f"Stdout:\n{stdout}")
#         print(f"Stderr:\n{stderr}")
    
#     print("=== Server cleanup complete ===\n")

# class TestIntegration:
#     def test_basic_reset(running_server):
#         """
#         Minimal test that just tries to create an environment and reset it.
#         """
#         print("\n=== Starting basic reset test ===")
        
#         print("Creating environment...")
#         env = AbstractEnv()
        
#         print("Environment created, checking initialization...")
#         assert env._connected, "Environment failed to connect to server"
#         assert env._sock is not None, "Socket not initialized"
        
#         print("Environment initialized successfully. Calling reset...")
#         try:
#             obs, info = env.reset()
#             print(f"Reset successful!")
#             print(f"Observation shape: {obs.shape if obs is not None else None}")
#             print(f"Info: {info}")
#         except Exception as e:
#             print(f"Reset failed with error: {str(e)}")
#             print("Attempting to close environment...")
#             env.close()
#             raise
        
#         print("Closing environment...")
#         env.close()
        
#         print("=== Test complete ===\n")
    
#     # def test_full_episode(self, running_server):
#     #     env = AbstractEnv()
        
#     #     # Test reset
#     #     obs, info = env.reset()
#     #     assert obs is not None
#     #     assert isinstance(obs, np.ndarray)
        
#     #     # Run a full episode
#     #     total_reward = 0
#     #     max_steps = env.config["max_episode_steps"]
        
#     #     for _ in range(max_steps):
#     #         # Generate random action
#     #         action = np.random.uniform(-1, 1, size=(len(env.controlled_vehicles), 2))
            
#     #         obs, reward, terminated, truncated, info = env.step(action)
#     #         total_reward += reward
            
#     #         assert obs is not None
#     #         assert isinstance(reward, float)
#     #         assert isinstance(terminated, bool)
#     #         assert isinstance(truncated, bool)
            
#     #         if terminated or truncated:
#     #             break
        
#     #     env.close()

#     # def test_multiple_episodes(self, running_server):
#     #     env = AbstractEnv()
#     #     n_episodes = 3
        
#     #     for episode in range(n_episodes):
#     #         obs, info = env.reset()
#     #         episode_reward = 0
            
#     #         while True:
#     #             action = np.random.uniform(-1, 1, size=(len(env.controlled_vehicles), 2))
#     #             obs, reward, terminated, truncated, info = env.step(action)
#     #             episode_reward += reward
                
#     #             if terminated or truncated:
#     #                 break
            
#     #         # Verify episode completed successfully
#     #         assert episode_reward != 0  # Basic sanity check
        
#     #     env.close()

#     # def test_concurrent_environments(self, running_server):
#     #     # Test multiple environments running simultaneously
#     #     env1 = AbstractEnv()
#     #     env2 = AbstractEnv()
        
#     #     obs1, _ = env1.reset()
#     #     obs2, _ = env2.reset()
        
#     #     # Run both environments for a few steps
#     #     for _ in range(10):
#     #         action1 = np.random.uniform(-1, 1, size=(len(env1.controlled_vehicles), 2))
#     #         action2 = np.random.uniform(-1, 1, size=(len(env2.controlled_vehicles), 2))
            
#     #         obs1, r1, term1, trunc1, _ = env1.step(action1)
#     #         obs2, r2, term2, trunc2, _ = env2.step(action2)
            
#     #         # Verify both environments are functioning independently
#     #         assert not np.array_equal(obs1, obs2)  # Observations should differ
        
#     #     env1.close()
#     #     env2.close()

#     # def test_performance(self, running_server):
#     #     env = AbstractEnv()
#     #     obs, _ = env.reset()
        
#     #     # Test step time
#     #     start_time = time.time()
#     #     n_steps = 100
        
#     #     for _ in range(n_steps):
#     #         action = np.random.uniform(-1, 1, size=(len(env.controlled_vehicles), 2))
#     #         obs, reward, terminated, truncated, info = env.step(action)
            
#     #         if terminated or truncated:
#     #             obs, _ = env.reset()
        
#     #     total_time = time.time() - start_time
#     #     avg_step_time = total_time / n_steps
        
#     #     # Verify performance meets requirements
#     #     assert avg_step_time < 0.1  # Maximum 100ms per step
        
#     #     env.close()