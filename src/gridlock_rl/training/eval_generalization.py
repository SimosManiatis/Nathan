import yaml
import numpy as np
import argparse
import os
import pandas as pd
from stable_baselines3 import PPO
from gridlock_rl.envs.grid_env import GridEnv

def run_eval_batch(model, seeds, config, label="Default"):
    print(f"\nRunning {label} Evaluation ({len(seeds)} episodes)...")
    
    env = GridEnv(**config)
    
    results = {
        "success": 0,
        "trap": 0,
        "timeout": 0,
        "steps": [],
        "keys": [],
        "success_steps": []
    }
    
    for seed in seeds:
        # PPO default deterministic=True
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
            results["success_steps"].append(steps)
        elif event == "trap":
            results["trap"] += 1
        elif event == "timeout":
            results["timeout"] += 1
            
    n = len(seeds)
    metrics = {
        "Set": label,
        "Success Rate": results["success"]/n,
        "Trap Rate": results["trap"]/n,
        "Timeout Rate": results["timeout"]/n,
        "Mean Steps": np.mean(results["steps"]),
        "Mean Keys": np.mean(results["keys"]),
        ">=1 Key": np.mean(np.array(results["keys"]) >= 1),
        ">=2 Keys": np.mean(np.array(results["keys"]) >= 2),
        "3 Keys": np.mean(np.array(results["keys"]) == 3),
        "Success Steps (Mean)": np.mean(results["success_steps"]) if results["success_steps"] else float('nan')
    }
    return metrics

def eval_generalization(model_path, id_config_path, id_bench_path, ood_bench_path):
    # Load Model
    print(f"Loading model: {model_path}")
    model = PPO.load(model_path)
    
    # 1. ID Evaluation
    with open(id_config_path, "r") as f:
        id_cfg = yaml.safe_load(f)["env"]
    with open(id_bench_path, "r") as f:
        id_seeds = yaml.safe_load(f)["seeds"]
        
    m_id = run_eval_batch(model, id_seeds, id_cfg, label="ID (Train-Like)")
    
    # 2. OOD Evaluation
    with open(ood_bench_path, "r") as f:
        ood_data = yaml.safe_load(f)
        ood_seeds = ood_data["seeds"]
        ood_cfg = ood_data["config"]
        
    # Inject ID config parameters (e.g. padding, max_steps) into OOD config to ensure compatibility
    # The Model expects a specific observation shape (max_width, max_height)
    if "max_width" in id_cfg:
        ood_cfg["max_width"] = id_cfg["max_width"]

    if "max_height" in id_cfg:
        ood_cfg["max_height"] = id_cfg["max_height"]
        
    # NOTE: If the model works on 8x8 but OOD is 10x10, the dictionary observation 
    # will have different shapes. 
    # SB3 PPO MultiInputPolicy (CNN/MLP) usually fixes input size at creation.
    # An MLP will CRASH if observation size changes (flattened input size differs).
    # A CNN might survive if it's fully convolutional, but SB3's default NatureCNN flattens at the end.
    
    # If this crashes, it proves the architecture is not generalizable by default.
    try:
        m_ood = run_eval_batch(model, ood_seeds, ood_cfg, label="OOD (Generalized)")
    except ValueError as e:
        print(f"\n[!] OOD Evaluation Failed: {e}")
        print("Reason: Model input shape mismatch. Standard SB3 PPO cannot handle variable grid sizes.")
        m_ood = {"Set": "OOD", "Success Rate": 0.0, "Trap Rate": 1.0, "Timeout Rate": 0.0, "Mean Keys": 0.0}
        
    # Report
    df = pd.DataFrame([m_id, m_ood])
    
    # Format percentages for specific columns
    for col in ["Success Rate", "Trap Rate", "Timeout Rate", ">=1 Key", ">=2 Keys", "3 Keys"]:
        df[col] = df[col].apply(lambda x: f"{x:.2%}")
        
    # Custom formatting for non-percentage columns
    print("\n" + "="*50)
    print("GENERALIZATION REPORT")
    print("="*50)
    print(df.to_string(index=False))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--id-config", type=str, default="configs/train/curr_stage2.yaml")
    parser.add_argument("--id-bench", type=str, default="configs/maps/benchmark_seeds.yaml")
    parser.add_argument("--ood-bench", type=str, default="configs/maps/benchmark_ood_seeds.yaml")
    args = parser.parse_args()
    
    eval_generalization(args.model, args.id_config, args.id_bench, args.ood_bench)
