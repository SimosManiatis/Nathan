import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from gridlock_rl.envs.grid_env import GridEnv
import argparse
import time

def evaluate_model(model_path, env_config, n_episodes=500):
    print(f"Loading model from {model_path}")
    print(f"Environment Config: {env_config}")
    
    # Create Env
    env = GridEnv(**env_config)
    model = PPO.load(model_path)
    
    outcomes = {
        "success": 0,
        "trap": 0,
        "timeout": 0
    }
    
    metrics = {
        "keys_collected_counts": [], # To compute >=1 prob
        "first_key_steps": [],
        "success_steps": []
    }
    
    print(f"Running {n_episodes} episodes...")
    start_time = time.time()
    
    for i in range(n_episodes):
        obs, info = env.reset()
        terminated = False
        truncated = False
        steps = 0
        first_key_step = None
        current_keys = 0
        
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1
            
            # Track first key
            if info["keys_collected"] > current_keys:
                current_keys = info["keys_collected"]
                if first_key_step is None:
                    first_key_step = steps
        
        # Episode Done
        event = info["event"]
        outcomes[event] = outcomes.get(event, 0) + 1
        
        metrics["keys_collected_counts"].append(info["keys_collected"])
        
        if first_key_step is not None:
            metrics["first_key_steps"].append(first_key_step)
            
        if event == "success":
            metrics["success_steps"].append(steps)
            
    elapsed = time.time() - start_time
    
    # Report
    print("\n" + "="*30)
    print("EVALUATION RESULTS")
    print("="*30)
    print(f"Episodes: {n_episodes}")
    print(f"Time: {elapsed:.2f}s")
    
    # Breakdown
    print("\nTermination Breakdown:")
    for evt, count in outcomes.items():
        pct = (count / n_episodes) * 100
        print(f"  {evt}: {count} ({pct:.1f}%)")
        
    # Metrics
    keys_arr = np.array(metrics["keys_collected_counts"])
    keys_ge_1 = np.mean(keys_arr >= 1)
    
    print("\nMetrics:")
    print(f"  Keys >= 1 Prob: {keys_ge_1:.3f}")
    if metrics["first_key_steps"]:
        print(f"  Mean First Key Step: {np.mean(metrics['first_key_steps']):.1f}")
    else:
        print("  Mean First Key Step: N/A")
        
    if metrics["success_steps"]:
        print(f"  Mean Steps (Success): {np.mean(metrics['success_steps']):.1f}")
    else:
        print("  Mean Steps (Success): N/A")
        
    print("="*30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    args = parser.parse_args()
    
    # Stage 0B Config (Fixed)
    # Stage 2 Config
    config = {
        "width": 8,
        "height": 8,
        "trap_density": 0.03,
        "num_keys": 3,
        "dense_reward": True,
        "step_cost": 0.02, # Make sure eval environment matches training dynamics
        "timeout_penalty": 10.0,
        "success_reward": 20.0,
        "min_traps": 1
    }
    
    evaluate_model(args.model, config)
