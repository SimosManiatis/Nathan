from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

class MetricsCallback(BaseCallback):
    """
    Custom callback for logging detailed metrics from MetricLoggingWrapper.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.metrics_buffer = {
            "keys_collected": [],
            "total_keys": [],
            "first_key_step": [],
            "time_to_goal": [],
            "shaping_sum": [],
            "extrinsic_sum": [],
            "is_success": []
        }

    def _on_step(self) -> bool:
        # Check for done episodes in infos
        infos = self.locals.get("infos", [])
        for info in infos:
            if "metrics" in info:
                m = info["metrics"]
                self.metrics_buffer["keys_collected"].append(m["keys_collected"])
                # Extract total_keys from the info dict itself (GridEnv puts it there)
                # Default to 3 if missing (shouldn't happen with updated env)
                self.metrics_buffer["total_keys"].append(info.get("total_keys", 3))
                
                self.metrics_buffer["shaping_sum"].append(m["shaping_reward_sum"])
                self.metrics_buffer["extrinsic_sum"].append(m["extrinsic_reward_sum"])
                self.metrics_buffer["is_success"].append(m.get("is_success", 0.0))
                
                if "first_key_step" in m:
                    self.metrics_buffer["first_key_step"].append(m["first_key_step"])
                
                if "time_after_last_key_to_goal" in m:
                    self.metrics_buffer["time_to_goal"].append(m["time_after_last_key_to_goal"])
        return True

    def _on_rollout_end(self) -> None:
        # Compute and Log Aggregates
        n_episodes = len(self.metrics_buffer["keys_collected"])
        
        if n_episodes > 0:
            keys = np.array(self.metrics_buffer["keys_collected"])
            totals = np.array(self.metrics_buffer["total_keys"])
            
            # Keys Distribution
            frac_ge_1 = np.mean(keys >= 1)
            frac_ge_2 = np.mean(keys >= 2)
            frac_all = np.mean(keys >= totals)
            
            self.logger.record("env/keys_mean", np.mean(keys))
            self.logger.record("env/keys_ge_1_prob", frac_ge_1)
            self.logger.record("env/keys_ge_2_prob", frac_ge_2)
            self.logger.record("env/keys_all_prob", frac_all)
            
            # Rewards
            self.logger.record("env/shaping_reward_mean", np.mean(self.metrics_buffer["shaping_sum"]))
            self.logger.record("env/extrinsic_reward_mean", np.mean(self.metrics_buffer["extrinsic_sum"]))
            
            # Timing
            if self.metrics_buffer["first_key_step"]:
                self.logger.record("env/first_key_step_mean", np.mean(self.metrics_buffer["first_key_step"]))
            
            if self.metrics_buffer["time_to_goal"]:
                self.logger.record("env/time_to_goal_mean", np.mean(self.metrics_buffer["time_to_goal"]))
                
            # Success
            self.logger.record("env/success_rate_custom", np.mean(self.metrics_buffer["is_success"]))

        # Clear buffers
        for k in self.metrics_buffer:
            self.metrics_buffer[k] = []
            
        return True
