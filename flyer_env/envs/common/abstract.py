import os
from typing import Dict, List, Optional, Text, Tuple, TypeVar
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import json
import socket
import sys
import threading
import time
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass
from contextlib import contextmanager

from flyer_env.envs.common.observation import observation_factory, ObservationType
from flyer_env.envs.common.action import action_factory, ActionType

Observation = TypeVar("Observation")
Action = TypeVar("Action")

@dataclass
class Aircraft:
    "Configuration for aircraft objects"
    id: str  # Unique identifier
    type: str  # Aircraft type (e.g. "dubins")
    observation: ObservationType  # Observation space object
    action: ActionType  # Action space object

@dataclass
class ConnectionConfig:
    """Configuration for TCP connection to Rust server"""
    host: str = "127.0.0.1"
    connection_timeout: float = 120.0
    response_timeout: float = 120.0
    retry_interval: float = 0.1
    max_retries: int = 3
    backoff_factor: float = 2.0

class AbstractEnv(ABC):
    """
    A generic environment that serves as the basis for the FlyerEnv
    This environment creates a server to connect to the Bevy Running app.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        config: dict = None,
        render_mode: Optional[str] = None,
        connection_config: Optional[ConnectionConfig] = None,
        debug_level: str = "info"
    ) -> None:

        super().__init__()

        # Connection management
        self._connection_config = connection_config or ConnectionConfig()
        self._debug_level = debug_level
        self._process = None
        self._sock = None
        self._connected = False

        # Initialize base attributes
        self.controlled_vehicles: List[Aircraft] = []
        self.config = self.default_config()
        if config:
            self.configure(config)
        self.dt = self.config["time_step"]

        # Rendering setup
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        if self.render_mode:
            if self.render_mode == "rgb_array":
                self.config['agent_config']['mode'] = "RGBArray"
            else:
                self.config['agent_config']['mode'] = "human"

        # Initialize connection to Rust server
        try:
            self._start_game()
        except Exception as e:
            print(f"Failed to start game: {e}")
            self.close()
            raise

    # @staticmethod
    # def stream_logs(process):
    #     """Stream logs from process stderr to console"""
    #     for line in iter(process.stderr.readline, ''):
    #         print(f"[SERVER] {line.strip()}", file=sys.stderr, flush=True)

    @staticmethod
    def stream_logs(process):
        """Stream logs from process stderr to console"""
        print("Log thread started - waiting for server logs...")
        try:
            for line in process.stderr:
                line = line.strip()
                if line:
                    print(f"[SERVER] {line}", file=sys.stderr, flush=True)
        except Exception as e:
            print(f"Error in log streaming: {e}", file=sys.stderr)

    def _start_game(self) -> None:
        """Initialize connection to Rust server"""
        print("Starting Flyer initialization...")

        server_command = "flyer_serve"  # Globally installed exexutable
        env = os.environ.copy()
        env["RUST_LOG"] = self._debug_level

        # Start Bevy process
        try:
            self._process = subprocess.Popen(
                [server_command],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True,
                env=env
            )

            # Start log streaming in a separate thread
            self._log_thread = threading.Thread(
                target=self.stream_logs,
                args=(self._process,),
                daemon=True
            )
            self._log_thread.start()

            print(f"self._log_thread: {self._log_thread}")

        except FileNotFoundError as e:
            raise RuntimeError(
                f"Could not find 'flyer_serve' executable. Ensure it is installed and accessible in PATH. Original error: {e}"
            )

        # Get port from process output with timeout
        start_time = time.time()
        self.port = None

        while time.time() - start_time < self._connection_config.connection_timeout:
            line = self._process.stdout.readline()
            if line.startswith("PORT="):
                self.port = int(line.strip().split("=")[1])
                break

        if not self.port:
            self._cleanup_socket()
            self._cleanup_process()
            raise TimeoutError("Timeout waiting for server port")

        # Connect socket
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        try:
            self._sock.connect((self._connection_config.host, self.port))
            self._connected = True
            self._initialize_env()
        except Exception as e:
            self._cleanup_socket()
            self._cleanup_process()
            raise RuntimeError(f"Failed to connect to server: {e}")

    @contextmanager
    def _managed_connection(self, timeout: float) -> socket.socket:
        """Context manager for handling socket operations with timeout"""
        if not self._sock:
            raise RuntimeError("Socket not initialized")

        original_timeout = self._sock.gettimeout()
        try:
            self._sock.settimeout(timeout)
            yield self._sock
        finally:
            if self._sock:
                self._sock.settimeout(original_timeout)

    def _send_command(self, command: dict, timeout: float = None) -> dict:
        """Send command to Rust server with retry mechanism"""
        if not self._connected:
            raise RuntimeError("Not connected to server")

        timeout = timeout or self._connection_config.response_timeout
        command_str = json.dumps(command) + "\n"

        retries = 0
        last_exception = None

        while retries < self._connection_config.max_retries:
            try:
                with self._managed_connection(timeout) as sock:
                    sock.sendall(command_str.encode())
                    if list(command.keys())[0] == "Render":
                        return self._read_render_response(timeout)
                    else:
                        return self._read_response(timeout)

            except (socket.timeout, json.JSONDecodeError) as e:
                last_exception = e
                retries += 1

                if retries < self._connection_config.max_retries:
                    # Calculate backoff time
                    backoff_time = self._connection_config.retry_interval * \
                                 (self._connection_config.backoff_factor ** retries)
                    print(f"Command failed, retrying in {backoff_time:.2f}s (attempt {retries + 1}/{self._connection_config.max_retries})")
                    time.sleep(backoff_time)

                    # Try to reconnect if needed
                    if not self._connected:
                        try:
                            self._reconnect()
                        except Exception as conn_err:
                            print(f"Reconnection failed: {conn_err}")
                            continue

        # If we get here, all retries failed
        raise RuntimeError(f"Command failed after {retries} retries. Last error: {last_exception}")

    def _read_response(self, timeout: float) -> dict:
        """Read and parse response from socket with improved timeout handling"""
        buffer = ""
        start_time = time.time()

        while True:
            if time.time() - start_time > timeout:
                raise socket.timeout("Timeout waiting for complete response")

            try:
                chunk = self._sock.recv(4096).decode()
                if not chunk:
                    # Connection closed by server
                    raise RuntimeError("Server closed connection")

                buffer += chunk
                if '\n' in buffer:
                    message, buffer = buffer.split('\n', 1)
                    try:
                        return json.loads(message)
                    except json.JSONDecodeError as e:
                        raise RuntimeError(f"Invalid JSON response: {e}")

            except socket.timeout:
                # Short timeout, continue reading
                continue

    def _reconnect(self):
        """Attempt to reconnect to the server"""
        self._cleanup_socket()

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

        try:
            self._sock.connect((self._connection_config.host, self.port))
            self._connected = True
            print("Successfully reconnected to server")
        except Exception as e:
            self._connected = False
            raise RuntimeError(f"Reconnection failed: {e}")

    def _read_render_response(self, timeout: float) -> dict:
        """Read length-prefixed response from socket"""
        start_time = time.time()

        # Read 4-byte length prefix
        length_bytes = self._recv_exactly(4, start_time, timeout)
        if not length_bytes:
            raise TimeoutError("Connection closed")

        message_length = int.from_bytes(length_bytes, 'big')
        message = self._recv_exactly(message_length, start_time, timeout).decode()
        return json.loads(message)

    def _recv_exactly(self, n: int, start_time: float, timeout: float) -> bytes:
        """Read exactly n bytes with timeout"""
        buffer = bytearray()
        while len(buffer) < n:
            if time.time() - start_time > timeout:
                raise TimeoutError("Timeout waiting for response")
            chunk = self._sock.recv(min(4096, n - len(buffer)))
            if not chunk:
                return None
            buffer.extend(chunk)
        return bytes(buffer)

    def _cleanup_socket(self) -> None:
        """Clean up socket connection"""
        if self._sock:
            try:
                self._sock.close()
            except Exception:
                pass
            finally:
                self._sock = None
                self._connected = False

    def _cleanup_process(self) -> None:
        """Clean up Bevy server process"""
        if self._process:
            try:
                self._process.terminate()
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
            finally:
                if hasattr(self, '_log_thread'):
                    # Give the log thread time to finish
                    self._log_thread.join(timeout=1)
                self._process = None

    def _initialize_env(self) -> None:
        """Initialize environment on server and create Aircraft instances"""
        print(f"Sent: {self.config}")
        init_msg = {
            "Initialize": {
                "config": self.config
            }
        }

        response = self._send_command(init_msg)
        if response.get("status") != "ready":
            raise RuntimeError(f"Failed to initialize: {response}")

        normalize_observations = self.config.get("normalize_observations", True)
        normalize_actions = self.config.get("normalize_actions", True)

        # Setup controlled vehicles
        for aircraft_info in response["aircraft"]:
            print(f"aircraft_info: {aircraft_info}")
            aircraft_type = list(aircraft_info["config"].keys())[0]
            aircraft_config = aircraft_info["config"][aircraft_type]  # limits from aircraft config

            observation = observation_factory(
                aircraft_type,
                list(aircraft_info["observation_space"].keys())[0],
                config = aircraft_config,
                normalize = normalize_observations
            )
            action = action_factory(
                aircraft_type,
                list(aircraft_info["action_space"].keys())[0],
                config = aircraft_config,
                normalize = normalize_actions
            )

            aircraft = Aircraft(
                id=aircraft_info["name"],
                type=aircraft_type,
                observation=observation,
                action=action
            )
            self.controlled_vehicles.append(aircraft)

    @abstractmethod
    def step(self, action: Action) -> Tuple[Observation, float, bool, bool, dict]:
        """Step envrionment with action"""
        pass

    @abstractmethod
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Observation, dict]:
        """Reset Env with seed and options"""
        pass

    @abstractmethod
    def render(self):
        """Get RGB array render from the environment"""
        pass

    @classmethod
    def default_config(cls) -> dict:
        """
        Default environment configuration

        Can be overloaded within environment config or with configure()
        :return: a configuration dict
        """
        return {
            "max_episode_steps": 1000,
            "steps_per_action": 4,
            "time_step": 1.0/120.0,
            "agent_config": {
                "render_width": 800.0,
                "render_height": 600.0,
                "mode": "human"
            }
        }

    def configure(self, config: dict) -> None:
        """Update configuration with new values"""
        if config:
            if 'agent_config' not in config:
                config['agent_config'] = self.default_config()['agent_config']
            self.config.update(config)

    def close(self) -> None:
        """Clean up resources"""
        if self._connected:
            try:
                self._send_command({"Close"})
            except Exception:
                pass

        self._cleanup_socket()
        self._cleanup_process()

        self.done = True

    def __del__(self) -> None:
        """Ensure cleanup on deletion"""
        self.close()
