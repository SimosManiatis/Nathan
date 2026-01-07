import gymnasium as gym
from stable_baselines3 import PPO
from gridlock_rl.envs.grid_env import GridEnv
import argparse
import time

def debug_policy(model_path, env_config):
    print(f"Loading model from {model_path}")
    # Load model
    env = GridEnv(**env_config, render_mode="human")
    model = PPO.load(model_path)
    
    for i in range(5):
        print(f"\n--- Episode {i+1} ---")
        obs, info = env.reset()
        env.render()
        terminated = False
        truncated = False
        total_reward = 0
        
        while not (terminated or truncated):
            # Slow down for visualization
            time.sleep(0.5) 
            
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            env.render()
            
        print(f"Result: {info['event']}, Reward: {total_reward:.2f}, Keys: {info['keys_collected']}")

if __name__ == "__main__":
    # Stage 0B Config
    config = {
        "width": 6,
        "height": 6,
        "trap_density": 0.05,
        "num_keys": 1,
        "dense_reward": True # Model expects dense input? Dict obs doesn't change, but behavior does.
    }
    
    # Path to latest model
    model_path = "runs/stage0b_traps/models/final_model.zip"
    
    debug_policy(model_path, config)
