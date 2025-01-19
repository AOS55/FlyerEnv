import json
from pathlib import Path

class EnvironmentConfigs:
    @staticmethod
    def get_dubins_config(seed=None):
        config = {
            "max_episode_steps": 1000,
            "normalize_actions": True,
            "steps_per_action": 4,
            "time_step": 1.0/120.0,
            "aircraft_config": [{
                "type": "dubins",
                "action_type": "Continuous",
                "observation_type": "Continuous"
            }],
            "agent_config": {
                "render_width": 800.0,
                "render_height": 600.0,
                "mode": "human"
            }
        }
        if seed is not None:
            config["seed"] = seed
        return config

    @staticmethod
    def get_full_config():
        return {
            "max_episode_steps": 1000,
            "normalize_actions": True,
            "steps_per_action": 4,
            "time_step": 1.0/120.0,
            "aircraft_config": [{
                "type": "full",
                "action_type": "Continuous",
                "observation_type": "Continuous"
            }],
            "agent_config": {
                "render_width": 800.0,
                "render_height": 600.0,
                "mode": "human"
            },
        }
        pass

    @staticmethod
    def get_multi_agent_config():
        # Configuration for multiple aircraft
        pass

    @staticmethod
    def load_config(config_path: Path) -> dict:
        with open(config_path, 'r') as f:
            return json.load(f)

    @staticmethod
    def modify_config(base_config: dict, updates: dict) -> dict:
        """Deep update of configuration dictionary."""
        config = base_config.copy()

        def update_dict(d, u):
            for k, v in u.items():
                if isinstance(v, dict):
                    d[k] = update_dict(d.get(k, {}), v)
                else:
                    d[k] = v
            return d

        return update_dict(config, updates)
