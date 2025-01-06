import pytest
import numpy as np
from unittest.mock import patch, MagicMock
import socket
import json
import subprocess
import threading
import time

from flyer_env.envs.common.abstract import ConnectionConfig, AbstractEnv

# Hide for now no need for mock process testing, better to use the live server.
# class MockProcess:
#     """
#     Create a mock process to simulate subprocess behavior.
#     Allows testing scenarios such as successful process creation,
#     process termination, and timeout errors.
#     """
#     def __init__(self, port_response="PORT=12345\n", raise_error=False):
#         self.stdout = MagicMock()
#         self.stdout.readline.return_value = port_response
#         self.raise_error = raise_error

#     def terminate(self):
#         """Simulates process termination."""
#         pass

#     def wait(self, timeout=None):
#         """Simulates waiting for the process to finish, raising an error if required."""
#         if self.raise_error:
#             raise subprocess.TimeoutExpired([], timeout)

#     def kill(self):
#         """Simulates forcibly killing the process."""
#         pass

# @pytest.fixture
# def mock_socket_factory():
#     """
#     Fixture to create a mock socket.
#     Allows simulating socket responses or errors for different scenarios.
#     """
#     def _create_mock_socket(responses=None):
#         mock_socket = MagicMock()
#         if responses:
#             response_iter = iter(responses)
#             def side_effect(*args, **kwargs):
#                 try:
#                     response = next(response_iter)
#                     if isinstance(response, Exception):
#                         raise response
#                     return response.encode() if isinstance(response, str) else response
#                 except StopIteration:
#                     return b''
#             mock_socket.recv.side_effect = side_effect
#         return mock_socket
#     return _create_mock_socket

# class TestAbstractEnv:
#     @pytest.mark.parametrize("config,expected", [
#         (None, {"max_episode_steps": 1000}),
#         ({"max_episode_steps": 2000}, {"max_episode_steps": 2000}),
#         ({"new_param": "value"}, {"max_episode_steps": 1000, "new_param": "value"})
#     ])
#     def test_configuration(self, config, expected):
#         """
#         Test if the environment correctly handles its configuration.
#         - Verifies default configuration values.
#         - Checks if custom configurations override defaults.
#         - Ensures new parameters are properly added.
#         """
#         with patch("subprocess.Popen") as mock_popen, \
#              patch("socket.socket") as mock_socket:
#             mock_popen.return_value = MockProcess()
#             mock_socket.return_value.recv.return_value = '{"status": "ready"}\n'.encode()

#             env = AbstractEnv(config=config)
#             for key, value in expected.items():
#                 assert env.config[key] == value

#     def test_connection_retry_mechanism(self, mock_socket_factory):
#         """
#         Test the environment's connection retry mechanism.
#         - Simulates initial connection failures followed by success.
#         - Verifies the environment retries until the connection is established.
#         """
#         with patch("subprocess.Popen") as mock_popen:
#             mock_popen.return_value = MockProcess()
#             mock_socket = mock_socket_factory([
#                 socket.timeout(),
#                 '{"status": "ready"}\n',
#                 '{"status": "ready"}\n'  # Add extra response for initialization
#             ])

#             with patch("socket.socket", return_value=mock_socket):
#                 env = AbstractEnv(connection_config=ConnectionConfig(retry_interval=0.1))
#                 assert env._connected
#                 assert mock_socket.recv.call_count >= 2

#     def test_malformed_server_responses(self, mock_socket_factory):
#         """
#         Test how the environment handles malformed server responses.
#         - Simulates invalid JSON followed by a valid response.
#         - Ensures a RuntimeError is raised when invalid responses are received.
#         """
#         with patch("subprocess.Popen") as mock_popen:
#             mock_popen.return_value = MockProcess()
#             mock_socket = mock_socket_factory([
#                 'invalid json\n',
#                 '{"status": "ready"}\n'
#             ])

#             with patch("socket.socket", return_value=mock_socket):
#                 with pytest.raises(RuntimeError, match="Command failed"):
#                     AbstractEnv()

#     def test_step_with_multiple_vehicles(self, mock_socket_factory):
#         """
#         Test the `step` method when controlling multiple vehicles.
#         - Simulates a response with combined observations and rewards.
#         - Verifies that observations and info for each vehicle are handled correctly.
#         """
#         with patch("subprocess.Popen") as mock_popen:
#             mock_popen.return_value = MockProcess()
#             response = {
#                 "obs": [0.1, 0.2, 0.3, 0.4],
#                 "reward": 1.0,
#                 "terminated": False,
#                 "truncated": False,
#                 "info": {"vehicle_states": {"aircraft_0": "active", "aircraft_1": "active"}}
#             }
#             mock_socket = mock_socket_factory([
#                 '{"status": "ready"}\n',
#                 json.dumps(response) + '\n'
#             ])

#             with patch("socket.socket", return_value=mock_socket):
#                 env = AbstractEnv()
#                 env.controlled_vehicles = ["aircraft_0", "aircraft_1"]
#                 action = [np.array([0.1, 0.2]), np.array([0.3, 0.4])]

#                 obs, reward, terminated, truncated, info = env.step(action)
#                 assert len(obs) == 4  # Combined observations from both vehicles
#                 assert info["vehicle_states"]["aircraft_0"] == "active"

#     def test_graceful_shutdown(self, mock_socket_factory):
#         """
#         Test the environment's ability to shut down cleanly.
#         - Ensures resources (sockets, processes) are released properly.
#         - Verifies shutdown behavior even if a process timeout occurs.
#         """
#         with patch("subprocess.Popen") as mock_popen:
#             mock_process = MockProcess(raise_error=True)
#             mock_popen.return_value = mock_process
#             mock_socket = mock_socket_factory(['{"status": "ready"}\n'])

#             with patch("socket.socket", return_value=mock_socket):
#                 env = AbstractEnv()
#                 env.close()
#                 assert not env._connected
#                 assert env._sock is None
#                 assert env._process is None

#     def test_connection_timeouts(self):
#         """
#         Test connection timeout behavior.
#         - Simulates a scenario where the server does not provide a port.
#         - Ensures a TimeoutError is raised.
#         """
#         with patch("subprocess.Popen") as mock_popen:
#             mock_popen.return_value = MockProcess(port_response="")

#             with pytest.raises(TimeoutError, match="Timeout waiting for server port"):
#                 AbstractEnv(connection_config=ConnectionConfig(
#                     connection_timeout=0.1,
#                     response_timeout=0.1
#                 ))

#     @pytest.mark.parametrize("invalid_action,expected_error,controlled_vehicles", [
#         (None, TypeError, ["aircraft_0"]),
#         ("invalid", TypeError, ["aircraft_0"]),
#         ([1, 2, 3], IndexError, ["aircraft_0"]),
#         (np.array([1, 2, 3, 4]), IndexError, ["aircraft_0"])
#     ])
#     def test_invalid_actions(self, invalid_action, expected_error, controlled_vehicles, mock_socket_factory):
#         """
#         Test how invalid actions are handled during the `step` method.
#         - Verifies the environment raises appropriate errors for invalid inputs.
#         """
#         with patch("subprocess.Popen") as mock_popen:
#             mock_popen.return_value = MockProcess()
#             # Provide responses for both initialization and step
#             mock_socket = mock_socket_factory([
#                 '{"status": "ready"}\n',  # Initialization response
#                 '{"obs": [], "reward": 0, "terminated": false, "truncated": false, "info": {}}\n'  # Step response
#             ])

#             with patch("socket.socket", return_value=mock_socket):
#                 env = AbstractEnv()
#                 env.controlled_vehicles = controlled_vehicles
#                 with pytest.raises(expected_error):
#                     env.step(invalid_action)

#     def test_step_valid_action(self, mock_socket_factory):
#         """
#         Test the `step` method for a valid single-vehicle action.
#         - Simulates a valid step and verifies correct output values.
#         """
#         with patch("subprocess.Popen") as mock_popen:
#             mock_popen.return_value = MockProcess()
#             mock_socket = mock_socket_factory([
#                 '{"status": "ready"}\n',
#                 '{"obs": [0.1, 0.2], "reward": 1.0, "terminated": false, "truncated": false, "info": {}}\n'
#             ])

#             with patch("socket.socket", return_value=mock_socket):
#                 env = AbstractEnv()
#                 env.controlled_vehicles = ["aircraft_0"]
#                 action = [np.array([0.1, 0.2])]  # Wrap in list for one vehicle
#                 obs, reward, terminated, truncated, info = env.step(action)
#                 assert not terminated
#                 assert not truncated
#                 assert reward == 1.0

#     def test_reset_with_custom_options(self, mock_socket_factory):
#         """
#         Test the `reset` method with custom options.
#         - Simulates a reset and verifies the environment applies custom configurations.
#         """
#         custom_config = {
#             "max_episode_steps": 2000,
#             "custom_param": "value"
#         }

#         with patch("subprocess.Popen") as mock_popen:
#             mock_popen.return_value = MockProcess()
#             response = {
#                 "obs": [0.1, 0.2],
#                 "info": {"reset_count": 1}
#             }
#             mock_socket = mock_socket_factory([
#                 '{"status": "ready"}\n',
#                 json.dumps(response) + '\n'
#             ])

#             with patch("socket.socket", return_value=mock_socket):
#                 env = AbstractEnv()
#                 obs, info = env.reset(options={"config": custom_config})

#                 assert env.config["max_episode_steps"] == 2000
#                 assert env.config["custom_param"] == "value"
#                 assert len(obs) == 2
#                 assert info["reset_count"] == 1


# def test_live_connection():
#     """
#     Test the environment with a live Rust Bevy server.
#     Assumes the server is already running and accessible.
#     """
#     connection_config = ConnectionConfig(host="127.0.0.1", connection_timeout=10)

#     try:
#         env = AbstractEnv(connection_config=connection_config)
#         print(f"Controlled Vehicles: {env.controlled_vehicles}")
#         # Send initialization command
#         # command = {"Initialize": {"config": env.config}}
#         # response = env._send_command(command)
#         # assert response.get("status") == "ready"

#         # Send a step command
#         # command = {"Step": {"actions": {"aircraft_0": np.array([0.1, 0.2, 0.5])}}}
#         print(f"Action space: {env.controlled_vehicles[0].action.space()}")
#         print(f"Features: {env.controlled_vehicles[0].action.features}")

#         env.reset()

#         # for _ in range(100):
#         #     action_samples = {}
#         #     for aircraft in env.controlled_vehicles:
#         #         action_samples[aircraft.id] = aircraft.action.space().sample()
#         #     # print(f"action: {action}")
#         #     print(f"Step: {env.step(action_samples)}")
#         # try:
#         #     response = env._send_command(command)
#         #     print(f"Response: {response}")
#         # except Exception as e:
#         #     print(f"Error: {e}")

#         # env.step([np.array([0.1, 0.2])])
#         # assert "obs" in response
#         # assert "reward" in response

#         # Send a reset command
#         # command = {"Reset": {"seed": 42}}
#         # response = env._send_command(command)
#         # # env.reset(seed=42)
#         # print(response)

#         # Test Reset with different seed formats
#         # commands = [
#         #     {"Reset": {"seed": 42}},  # integer seed
#         #     {"Reset": {"seed": "Hats"}},  # null/None seed
#         #     {"reset": {"seed": 42}},  # lowercase to test case sensitivity
#         #     {"Reset": {"Seed": 42}},  # uppercase field name
#         # ]

#         # for cmd in commands:
#         #     print(f"\nTesting command format: {json.dumps(cmd)}")
#         #     try:
#         #         response = env._send_command(cmd)
#         #         print(f"Response: {response}")
#         #     except Exception as e:
#         #         print(f"Error: {e}")

#     finally:
#         env.close()

    #     reset_command = {"Reset": {"seed": 42}}
    #     command_str = json.dumps(reset_command) + "\n"
    #     print(f"Raw command being sent: {repr(command_str)}")
    #     print(f"Command bytes being sent: {command_str.encode()}")

    #     response = env._send_command(reset_command)
    #     print(f"Raw response received: {repr(response)}")



    # finally:
    #     env.close()


# if __name__=="__main__":
#     test_live_connection()
