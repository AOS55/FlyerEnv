import gymnasium as gym
import subprocess
import socket
import json
import numpy as np
import time
from typing import Optional, Dict, Any

class FlyerServeEnv(gym.Env):
    def __init__(self, config: Optional[Dict] = None, render_mode: str = "human"):
        
        print("Starting FlyerServeEnv initialization...")
        
        self.config = config or {
            "max_episode_steps": 1000,
            "steps_per_action": 4,
            "time_step": 1.0/60.0,
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
        
        print(f"Config initialized: {json.dumps(self.config, indent=2)}")
        
        # Add render mode to config
        self.config["agent_config"]["render_mode"] = render_mode
        
        # Start Bevy process
        try:
            print("Starting Bevy process...")
            self.process = subprocess.Popen(
                ["pyflyer-rs/target/release/bevy_server"],
                stdout = subprocess.PIPE,
                stderr = subprocess.PIPE,
                text = True,
                bufsize=1,
                universal_newlines=True
            )
        except FileNotFoundError as e:
            print(f"Failed to start Bevy process: {e}")
            raise RuntimeError("Check that 'target/debug/bevy_runner' exists and is executable.")
        
        print("Bevy process started")
        
        # Get port from process output
        for line in self.process.stdout:
            if line.startswith("PORT="):
                self.port = int(line.strip().split("=")[1])
                break
            
        try:
            # Connect socket
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.connect(("127.0.0.1", self.port))
            
            # Send initial config
            self._initialize_env()
            
        except Exception as e:
            self.close()
            raise RuntimeError(f"Failed to initialize environment: {e}")
        
        # Setup spaces based on config
        self._setup_spaces()
    
    def _get_port(self, timeout: float) -> int:
        """Get port number from Bevy process output with timeout"""
        start_time = time.time()
        while True:
            if self.process.poll() is not None:
                # Process died
                raise RuntimeError(
                    f"Bevy process died before sending port. "
                    f"Exit code: {self.process.returncode}"
                )
                
            line = self.process.stdout.readline()
            if line.startswith("PORT="):
                return int(line.strip().split("=")[1])
                
            if time.time() - start_time > timeout:
                raise TimeoutError("Timeout waiting for port number")
            
            time.sleep(0.1)
            
    def _initialize_env(self):
        """Send initial config and wait for ready signal"""
        # Send config in the format Rust expects
        init_msg = {
            "Initialize": {
                "config": self.config
            }
        }
        msg_str = json.dumps(init_msg)
        print(f"Sending initialization message: {msg_str}")
        self.sock.sendall((msg_str + "\n").encode())
        
        print("Waiting for ready signal...")
        response = self._read_response(timeout=5.0)
        print(f"Received response: {response}")
        
        if response.get("status") != "ready":
            print(f"Unexpected response status: {response}")
            raise RuntimeError(f"Failed to initialize: {response}")
            
    def _read_response(self, timeout: float) -> Dict:
        """Read response from socket with timeout"""
        self.sock.settimeout(timeout)
        buffer = ""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                print(f"Attempting to receive data... (Time remaining: {timeout - (time.time() - start_time):.2f}s)")  # Debug
                chunk = self.sock.recv(4096).decode()
                if not chunk:
                    print("No data received, waiting...")  # Debug
                    time.sleep(0.1)
                    continue
                
                print(f"Received chunk: '{chunk}'")  # Debug
                buffer += chunk
                
                if '\n' in buffer:
                    message, buffer = buffer.split('\n', 1)
                    print(f"Complete message received: '{message}'")  # Debug
                    try:
                        return json.loads(message)
                    except json.JSONDecodeError as e:
                        print(f"Failed to decode message: '{message}'")
                        print(f"JSON decode error: {e}")
                        raise
            except socket.timeout:
                print("Socket timeout, retrying...")  # Debug
                continue
                
        print("Timeout reached while waiting for response")  # Debug
        raise TimeoutError("Timeout waiting for response")
            
    def _setup_spaces(self):
        """Setup gym spaces based on config"""
        # Get dimensions from aircraft config
        aircraft = self.config["aircraft_config"][0]
        if aircraft["action_type"] == "Continuous":
            self.action_space = gym.spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(3,),
                dtype=np.float32
            )
        
        if aircraft["observation_type"] == "Continuous":
            self.observation_space = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(6,),
                dtype=np.float32
            )
    
    def step(self, action):
        
        # Send step command to Bevy process
        cmd = {"Step": {"count": 4}}
        self.sock.sendall(json.dumps(cmd).encode() + b"\n")
        
        obs = np.zeros(6)
        reward = 0.0
        terminated = False
        truncated = False
        info = {}
        
        return obs, reward, terminated, truncated, info
    
    def reset(self, seed=None, options=None):
        
        # Send reset command
        cmd = {"Reset": {}}
        self.sock.sendall(json.dumps(cmd).encode() + b"\n")
        
        obs = np.zeros(6)
        info = {}
        return obs, info
    
    def close(self):
        
        # Send exit command
        cmd = {"Exit": {}}
        self.sock.sendall(json.dumps(cmd).encode() + b"\n")
        
        self.sock.close()
        self.process.wait()
        
    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
        

def main():
    env = FlyerServeEnv()
    print(f"The created serve object: {env}")
    action = np.array([0.1, 0.2, 0.3]) 
    for ida in range(100):
        env.step(action)
        print(f"Took Action {ida}")

if __name__=="__main__":
    main()
        