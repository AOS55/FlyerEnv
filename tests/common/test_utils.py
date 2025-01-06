from typing import Optional
from flyer_env.envs.common.abstract import AbstractEnv

class EpisodeRunner:
    @staticmethod
    def run(env: AbstractEnv, max_steps: int = 100, 
            seed: Optional[int] = None) -> dict:
        """Run a test episode and collect metrics."""
        metrics = {
            'total_reward': 0,
            'steps': 0,
            'terminated': False,
            'truncated': False
        }
        
        obs, info = env.reset(seed=seed)
        
        for _ in range(max_steps):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            metrics['total_reward'] += reward
            metrics['steps'] += 1
            
            if terminated or truncated:
                metrics['terminated'] = terminated
                metrics['truncated'] = truncated
                break
                
        return metrics