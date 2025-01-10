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
    connection_timeout: float = 30.0
    response_timeout: float = 5.0
    retry_interval: float = 0.1

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
        connection_config: Optional[ConnectionConfig] = None
    ) -> None:

        super().__init__()

        # Connection management
        self._connection_config = connection_config or ConnectionConfig()
        self._process = None
        self._sock = None
        self._connected = False

        # Initialize base attributes
        self.controlled_vehicles: List[Aircraft] = []
        self.config = self.default_config()
        if config:
            self.configure(config)

        # Rendering setup
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        if self.render_mode:
            self.config.agent_config.mode = self.render_mode

        # Initialize connection to Rust server
        try:
            self._start_game()
        except Exception as e:
            print(f"Failed to start game: {e}")
            self.close()
            raise

    @staticmethod
    def stream_logs(process):
        """Stream logs from process stderr to console"""
        for line in iter(process.stderr.readline, ''):
            print(f"[SERVER] {line.strip()}", file=sys.stderr, flush=True)

    def _start_game(self) -> None:
        """Initialize connection to Rust server"""
        print("Starting Flyer initialization...")

        # Start Bevy process
        try:
            self._process = subprocess.Popen(
                ["flyer-rs/target/release/serve"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )

            # Start log streaming in a separate thread
            self._log_thread = threading.Thread(
                target=self.stream_logs,
                args=(self._process,),
                daemon=True
            )
            self._log_thread.start()

        except FileNotFoundError as e:
            raise RuntimeError(f"Check 'pyflyer-rs/target/release/bevy_server' exists and is executable, {e} found.")

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
        """Send command to Rust server and receive response"""

        if not self._connected:
            raise RuntimeError("Not connected to server")

        timeout = timeout or self._connection_config.response_timeout
        command_str = json.dumps(command) + "\n"

        with self._managed_connection(timeout) as sock:
            try:
                sock.sendall(command_str.encode())
                return self._read_response(timeout)
            except (socket.timeout, json.JSONDecodeError) as e:
                raise RuntimeError(f"Command failed: {e}")

    def _read_response(self, timeout: float) -> dict:
        """Read and parse response from socket with timeout"""
        buffer = ""
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                chunk = self._sock.recv(4096).decode()
                if not chunk:
                    time.sleep(self._connection_config.retry_interval)
                    continue

                buffer += chunk
                if '\n' in buffer:
                    message, buffer = buffer.split('\n', 1)
                    return json.loads(message)
            except socket.timeout:
                continue

        raise TimeoutError("Timeout waiting for response")

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
        init_msg = {
            "Initialize": {
                "config": self.config
            }
        }

        response = self._send_command(init_msg)
        if response.get("status") != "ready":
            raise RuntimeError(f"Failed to initialize: {response}")

        normalize_observations = self.config.get("normalize_observations", False)
        normalize_actions = self.config.get("normalize_actions", False)

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
        """Must be implemented by child classes"""
        pass

    @abstractmethod
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Observation, dict]:
        """Must be implemented by child classes"""
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
