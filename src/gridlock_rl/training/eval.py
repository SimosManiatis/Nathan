import yaml
import numpy as np
import argparse
import os
from stable_baselines3 import PPO
from gridlock_rl.envs.grid_env import GridEnv
from stable_baselines3.common.evaluation import evaluate_policy

def evaluate(model_path, config_path, benchmark_path=None, n_episodes=100):
    # Load config for env settings
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    env_cfg = config["env"]
    
    # Load seeds
    bench_seeds = []
    if benchmark_path and os.path.exists(benchmark_path):
        with open(benchmark_path, "r") as f:
            bench_seeds = yaml.safe_load(f)["seeds"]
    
    # Override n_episodes if we have a benchmark set
    if bench_seeds:
        n_episodes = len(bench_seeds)
        print(f"Evaluating on {n_episodes} benchmark seeds...")
    else:
        print(f"Evaluating on {n_episodes} random seeds...")

    # Load Model
    model = PPO.load(model_path)
    
    # Create Env
    env = GridEnv(**env_cfg)
    
    results = {
        "success": 0,
        "trap": 0,
        "timeout": 0,
        "steps": [],
        "keys": []
    }
    
    for i in range(n_episodes):
        seed = bench_seeds[i] if bench_seeds else i
        obs, info = env.reset(seed=seed)
        terminated, truncated = False, False
        steps = 0
        
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1
            
        event = info["event"]
        results["steps"].append(steps)
        results["keys"].append(info["keys_collected"])
        
        if event == "success":
            results["success"] += 1
        elif event == "trap":
            results["trap"] += 1
        elif event == "timeout":
            results["timeout"] += 1
            
    # Metrics
    n = n_episodes
    print("\n--- Evaluation Report ---")
    print(f"Success Rate: {results['success']/n:.2%}")
    print(f"Trap Rate: {results['trap']/n:.2%}")
    print(f"Timeout Rate: {results['timeout']/n:.2%}")
    print(f"Mean Steps: {np.mean(results['steps']):.1f}")
    print(f"Mean Keys: {np.mean(results['keys']):.2f}")
    
    # Key Distribution
    keys = np.array(results["keys"])
    print(f"Episodes >= 1 Key: {np.mean(keys >= 1):.2%}")
    print(f"Episodes >= 2 Keys: {np.mean(keys >= 2):.2%}")
    print(f"Episodes == 3 Keys: {np.mean(keys == 3):.2%}")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/train/ppo.yaml")
    parser.add_argument("--benchmark", type=str, default="configs/maps/benchmark_seeds.yaml")
    args = parser.parse_args()
    
    evaluate(args.model, args.config, args.benchmark)
