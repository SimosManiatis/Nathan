import gymnasium as gym
import numpy as np

class MetricLoggingWrapper(gym.Wrapper):
    """
    Wrapper to track and log custom metrics for Gridlock RL.
    Logs:
    - keys_collected: Count at end of episode.
    - first_key_step: Step number when first key was collected (or max_steps if none).
    - time_after_last_key_to_goal: steps from last key to goal (if success).
    - shaping_reward_sum: Cumulative shaping reward.
    - extrinsic_reward_sum: Cumulative extrinsic reward.
    """
    def __init__(self, env):
        super().__init__(env)
        self.reset_metrics()
        
    def reset_metrics(self):
        self.episode_keys_collected = 0
        self.first_key_step = None
        self.shaping_sum = 0.0
        self.extrinsic_sum = 0.0
        self.episode_steps = 0
        self.last_key_step = None
        
    def reset(self, **kwargs):
        self.reset_metrics()
        return self.env.reset(**kwargs)
        
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.episode_steps += 1
        
        # Track Rewards
        self.shaping_sum += info.get("shaping_reward", 0.0)
        self.extrinsic_sum += info.get("extrinsic_reward", 0.0)
        
        # Track Keys
        current_keys = info.get("keys_collected", 0)
        # Note: info['keys_collected'] from GridEnv is the count.
        # We want to detect change.
        if current_keys > self.episode_keys_collected:
            if self.first_key_step is None:
                self.first_key_step = self.episode_steps
            self.last_key_step = self.episode_steps
            self.episode_keys_collected = current_keys
            
        # On Termination, inject metrics
        if terminated or truncated:
            # Time from last key to goal (only if success and collected keys)
            time_to_goal = 0
            if info.get("event") == "success" and self.last_key_step is not None:
                time_to_goal = self.episode_steps - self.last_key_step
            
            # Prepare Metrics
            metrics = {
                "keys_collected": self.episode_keys_collected,
                "shaping_reward_sum": self.shaping_sum,
                "extrinsic_reward_sum": self.extrinsic_sum,
                "episode_steps": self.episode_steps,
            }
            
            if self.first_key_step is not None:
                metrics["first_key_step"] = self.first_key_step
                
            if info.get("event") == "success":
                metrics["time_after_last_key_to_goal"] = time_to_goal
                metrics["is_success"] = 1.0
            else:
                metrics["is_success"] = 0.0
                
            # Add to info
            info["metrics"] = metrics
            
            # Also for SB3 to pickup automatically locally if we use Monitor?
            # SB3 Monitor usually looks for 'episode' key.
            # We can pack these into 'episode' dict if we want them logged by Monitor.
            # But Monitor overwrites 'episode' key. 
            # We will rely on a Callback to extract info['metrics'].
            
        return obs, reward, terminated, truncated, info
