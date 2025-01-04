import pytest
import numpy as np
from unittest.mock import patch, MagicMock
import socket
import json
import subprocess

from flyer_env.envs.common.abstract import ConnectionConfig, AbstractEnv

class MockProcess:
    """
    Create a mock process to test simple happy path scenarios
    """
    def __init__(self, port_response="PORT=12345\n", raise_error=False):
        self.stdout = MagicMock()
        self.stdout.readline.return_value = port_response
        self.raise_error = raise_error

    def terminate(self):
        pass

    def wait(self, timeout=None):
        if self.raise_error:
            raise subprocess.TimeoutExpired([], timeout)

    def kill(self):
        pass

@pytest.fixture
def mock_socket_factory():
    def _create_mock_socket(responses=None):
        mock_socket = MagicMock()
        if responses:
            response_iter = iter(responses)
            def side_effect(*args, **kwargs):
                try:
                    response = next(response_iter)
                    if isinstance(response, Exception):
                        raise response
                    return response.encode() if isinstance(response, str) else response
                except StopIteration:
                    return b''
            mock_socket.recv.side_effect = side_effect
        return mock_socket
    return _create_mock_socket

class TestAbstractEnv:
    @pytest.mark.parametrize("config,expected", [
        (None, {"max_episode_steps": 1000}),
        ({"max_episode_steps": 2000}, {"max_episode_steps": 2000}),
        ({"new_param": "value"}, {"max_episode_steps": 1000, "new_param": "value"})
    ])
    def test_configuration(self, config, expected):
        with patch("subprocess.Popen") as mock_popen, \
             patch("socket.socket") as mock_socket:
            mock_popen.return_value = MockProcess()
            mock_socket.return_value.recv.return_value = '{"status": "ready"}\n'.encode()
            
            env = AbstractEnv(config=config)
            for key, value in expected.items():
                assert env.config[key] == value

    def test_connection_retry_mechanism(self, mock_socket_factory):
        with patch("subprocess.Popen") as mock_popen:
            mock_popen.return_value = MockProcess()
            mock_socket = mock_socket_factory([
                socket.timeout(),
                '{"status": "ready"}\n',
                '{"status": "ready"}\n'  # Add extra response for initialization
            ])
            
            with patch("socket.socket", return_value=mock_socket):
                env = AbstractEnv(connection_config=ConnectionConfig(retry_interval=0.1))
                assert env._connected
                assert mock_socket.recv.call_count >= 2

    def test_malformed_server_responses(self, mock_socket_factory):
        with patch("subprocess.Popen") as mock_popen:
            mock_popen.return_value = MockProcess()
            mock_socket = mock_socket_factory([
                'invalid json\n',
                '{"status": "ready"}\n'
            ])
            
            with patch("socket.socket", return_value=mock_socket):
                with pytest.raises(RuntimeError, match="Command failed"):
                    env = AbstractEnv()

    def test_step_with_multiple_vehicles(self, mock_socket_factory):
        with patch("subprocess.Popen") as mock_popen:
            mock_popen.return_value = MockProcess()
            response = {
                "obs": [0.1, 0.2, 0.3, 0.4],
                "reward": 1.0,
                "terminated": False,
                "truncated": False,
                "info": {"vehicle_states": {"aircraft_0": "active", "aircraft_1": "active"}}
            }
            mock_socket = mock_socket_factory([
                '{"status": "ready"}\n',
                json.dumps(response) + '\n'
            ])
            
            with patch("socket.socket", return_value=mock_socket):
                env = AbstractEnv()
                env.controlled_vehicles = ["aircraft_0", "aircraft_1"]
                action = [np.array([0.1, 0.2]), np.array([0.3, 0.4])]
                
                obs, reward, terminated, truncated, info = env.step(action)
                assert len(obs) == 4  # Combined observations from both vehicles
                assert info["vehicle_states"]["aircraft_0"] == "active"

    def test_graceful_shutdown(self, mock_socket_factory):
        with patch("subprocess.Popen") as mock_popen:
            mock_process = MockProcess(raise_error=True)
            mock_popen.return_value = mock_process
            mock_socket = mock_socket_factory(['{"status": "ready"}\n'])
            
            with patch("socket.socket", return_value=mock_socket):
                env = AbstractEnv()
                env.close()
                assert not env._connected
                assert env._sock is None
                assert env._process is None

    def test_connection_timeouts(self):
        with patch("subprocess.Popen") as mock_popen:
            mock_popen.return_value = MockProcess(port_response="")
            
            with pytest.raises(TimeoutError, match="Timeout waiting for server port"):
                env = AbstractEnv(connection_config=ConnectionConfig(
                    connection_timeout=0.1,
                    response_timeout=0.1
                ))

    @pytest.mark.parametrize("invalid_action,expected_error,controlled_vehicles", [
        (None, TypeError, ["aircraft_0"]),
        ("invalid", TypeError, ["aircraft_0"]),
        ([1, 2, 3], IndexError, ["aircraft_0"]),
        (np.array([1, 2, 3, 4]), IndexError, ["aircraft_0"])
    ])
    def test_invalid_actions(self, invalid_action, expected_error, controlled_vehicles, mock_socket_factory):
        with patch("subprocess.Popen") as mock_popen:
            mock_popen.return_value = MockProcess()
            # Provide responses for both initialization and step
            mock_socket = mock_socket_factory([
                '{"status": "ready"}\n',  # Initialization response
                '{"obs": [], "reward": 0, "terminated": false, "truncated": false, "info": {}}\n'  # Step response
            ])
            
            with patch("socket.socket", return_value=mock_socket):
                env = AbstractEnv()
                env.controlled_vehicles = controlled_vehicles
                with pytest.raises(expected_error):
                    env.step(invalid_action)

    def test_step_valid_action(self, mock_socket_factory):
        """Test single vehicle action"""
        with patch("subprocess.Popen") as mock_popen:
            mock_popen.return_value = MockProcess()
            mock_socket = mock_socket_factory([
                '{"status": "ready"}\n',
                '{"obs": [0.1, 0.2], "reward": 1.0, "terminated": false, "truncated": false, "info": {}}\n'
            ])
            
            with patch("socket.socket", return_value=mock_socket):
                env = AbstractEnv()
                env.controlled_vehicles = ["aircraft_0"]
                action = [np.array([0.1, 0.2])]  # Wrap in list for one vehicle
                obs, reward, terminated, truncated, info = env.step(action)
                assert not terminated
                assert not truncated
                assert reward == 1.0

    def test_step_with_multiple_vehicles(self, mock_socket_factory):
        """Test multiple vehicle actions"""
        with patch("subprocess.Popen") as mock_popen:
            mock_popen.return_value = MockProcess()
            response = {
                "obs": [0.1, 0.2, 0.3, 0.4],
                "reward": 1.0,
                "terminated": False,
                "truncated": False,
                "info": {"vehicle_states": {"aircraft_0": "active", "aircraft_1": "active"}}
            }
            mock_socket = mock_socket_factory([
                '{"status": "ready"}\n',
                json.dumps(response) + '\n'
            ])
            
            with patch("socket.socket", return_value=mock_socket):
                env = AbstractEnv()
                env.controlled_vehicles = ["aircraft_0", "aircraft_1"]
                action = [np.array([0.1, 0.2]), np.array([0.3, 0.4])]
                
                obs, reward, terminated, truncated, info = env.step(action)
                assert len(obs) == 4
                assert info["vehicle_states"]["aircraft_0"] == "active"
    
    def test_reset_with_custom_options(self, mock_socket_factory):
        custom_config = {
            "max_episode_steps": 2000,
            "custom_param": "value"
        }
        
        with patch("subprocess.Popen") as mock_popen:
            mock_popen.return_value = MockProcess()
            response = {
                "obs": [0.1, 0.2],
                "info": {"reset_count": 1}
            }
            mock_socket = mock_socket_factory([
                '{"status": "ready"}\n',
                json.dumps(response) + '\n'
            ])
            
            with patch("socket.socket", return_value=mock_socket):
                env = AbstractEnv()
                obs, info = env.reset(options={"config": custom_config})
                
                assert env.config["max_episode_steps"] == 2000
                assert env.config["custom_param"] == "value"
                assert len(obs) == 2
                assert info["reset_count"] == 1