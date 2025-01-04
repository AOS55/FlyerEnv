from typing import Dict, List, Optional, Text, Tuple, TypeVar
import gymnasium as gym
import numpy as np
import json
import socket
import time
import subprocess
from dataclasses import dataclass
from contextlib import contextmanager

Observation = TypeVar("Observation")

@dataclass
class ConnectionConfig:
    """Configuration for TCP connection to Rust server"""
    host: str = "127.0.0.1"
    connection_timeout: float = 30.0
    response_timeout: float = 5.0
    retry_interval: float = 0.1

class AbstractEnv(gym.Env):
    """
    A generic environment that serves as the basis for the FlyerEnv
    This environment creates a server to connect to the Bevy Running app.
    """

    # observation_type: ObservationType
    # action_type: ActionType
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
    
        # Configuration
        self.config = self.default_config()
        if config:
            self.configure(config)

        # Scene and vehicle management
        self.controlled_vehicles = []
        # self._setup_spaces()  # Initialize action and observation spaces

        # State tracking
        self.time = 0.0
        self.steps = 0
        self.done = False

        # Rendering
        self.viewer = None  # TODO: This would have been a pygame object, should delete? 
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # Initialize connection to Rust server
        try:
            self._start_game()
        except Exception as e:
            self.close()
            raise
        
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
    
    def _start_game(self) -> None:
        """Initialize connection to Rust server"""
        print("Starting Flyer initialization...")
        
        # Start Bevy process
        try:
            self._process = subprocess.Popen(
                ["pyflyer-rs/target/release/bevy_server"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
        except FileNotFoundError as e:
            raise RuntimeError("Check 'pyflyer-rs/target/release/bevy_server' exists and is executable.")
        
        # Get port from process output with timeout
        start_time = time.time()
        self.port = None
        
        while time.time() - start_time < self._connection_config.connection_timeout:
            line = self._process.stdout.readline()
            if line.startswith("PORT="):
                self.port = int(line.strip().split("=")[1])
                break
                
        if not self.port:
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
            raise

    def _initialize_env(self) -> None:
        """Send initial configuration to server"""
        init_msg = {
            "Initialize": {
                "config": self.config
            }
        }
        
        response = self._send_command(init_msg)
        if response.get("status") != "ready":
            raise RuntimeError(f"Failed to initialize: {response}")
    
    # @property
    # def vehicle(self) -> Aircraft:
    #     """First (default) controlled vehicle"""
    #     return self.controlled_vehicles[0] if self.controlled_vehicles else None

    # @vehicle.setter
    # def vehicle(self, vehicle: Aircraft) -> None:
    #     """Set a unique controlled vehicle"""
    #     self.controlled_vehicles = [vehicle]

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
            "aircraft_config": [{
                "type": "dubins",
                "action_type": "Continuous",
                "observation_type": "Continuous"
            }],
            "agent_config": {
                "render_width": 800.0,
                "render_height": 600.0
            }
        }
    
    def configure(self, config: dict) -> None:
        if config:
            self.config.update(config)     
            
    def step(self, action: np.ndarray) -> Tuple[Observation, float, bool, bool, dict]:
        """Execute action and get new state"""
        if not self._connected:
            raise RuntimeError("Not connected to server")

        # Basic input validation
        if not isinstance(action, (list, np.ndarray)):
            raise TypeError(f"Action must be list or ndarray, got {type(action)}")

        if len(self.controlled_vehicles) == 0:
            raise RuntimeError("No controlled vehicles available")

        # Validate number of actions matches number of vehicles
        if isinstance(action, np.ndarray):
            # For numpy arrays, first dimension should match number of vehicles
            if len(action.shape) < 1 or action.shape[0] != len(self.controlled_vehicles):
                raise IndexError(f"Action shape {action.shape} does not match number of vehicles {len(self.controlled_vehicles)}")
        else:
            # For lists, length should match number of vehicles
            if len(action) != len(self.controlled_vehicles):
                raise IndexError(f"Action length {len(action)} does not match number of vehicles {len(self.controlled_vehicles)}")

        # Create action dictionary with native Python types
        action_dict = {}
        for i in range(len(self.controlled_vehicles)):
            if isinstance(action[i], np.ndarray):
                action_dict[f"aircraft_{i}"] = action[i].tolist()
            elif isinstance(action[i], list):
                action_dict[f"aircraft_{i}"] = action[i]
            else:
                # Single value
                action_dict[f"aircraft_{i}"] = float(action[i])

        command = {
            "Step": {
                "actions": action_dict
            }
        }

        try:
            response = self._send_command(command)
            return (
                np.array(response["obs"]),
                response["reward"],
                response["terminated"],
                response["truncated"],
                response["info"]
            )
        except Exception as e:
            self.close()
            raise RuntimeError(f"Step failed: {e}")

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None
    ) -> Tuple[Observation, dict]:
        """Reset environment state"""
        # super().reset(seed=seed, options=options)
        
        # if options and "config" in options:
        #     self.configure(options["config"])
        
        command = {
            "Reset": {
                "seed": seed
            }
        }
        
        try:
            response = self._send_command(command)
            print(f"Response: {response}")
            return np.array(response["obs"]), response["info"]
        except Exception as e:
            self.close()
            raise RuntimeError(f"Reset failed: {e}")
    
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
                self._process = None

    def close(self) -> None:
        """Clean up resources"""
        if self._connected:
            try:
                self._send_command({"Close": {}})
            except Exception:
                pass
                
        self._cleanup_socket()
        self._cleanup_process()
        
        if self.viewer:
            pygame.quit()
            self.viewer = None
            
        self.done = True

    def __del__(self) -> None:
        """Ensure cleanup on deletion"""
        self.close()