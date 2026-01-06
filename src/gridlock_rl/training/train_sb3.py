import yaml
import os
import argparse
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor

from gridlock_rl.envs.grid_env import GridEnv

def make_env(**kwargs):
    def _init():
        env = GridEnv(**kwargs)
        return env
    return _init

def train(config_path, run_name="default", load_model_path=None):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    env_cfg = config["env"]
    train_cfg = config["training"]
    
    print("\n" + "="*50)
    print("RESOLVED TRAINING CONFIGURATION:")
    for k, v in train_cfg.items():
        print(f"  {k}: {v}")
    print("="*50 + "\n")
    
    # Paths (Hardening: Create all dirs)
    base_dir = f"runs/{run_name}"
    model_dir = os.path.join(base_dir, "models")
    log_dir = os.path.join(base_dir, "logs")
    
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # 1. Create Vectorized Environment
    n_envs = 8 # Increased parallel envs for better exploration signal
    
    # Use SubprocVecEnv for speed
    env = SubprocVecEnv([
        make_env(**env_cfg) 
        for _ in range(n_envs)
    ])
    
    # Hardening: Use VecMonitor for correct parallel logging
    env = VecMonitor(env, filename=os.path.join(log_dir, "monitor.csv"))
    
    # 2. Evaluation Environment
    # Use Monitor here for eval stats
    eval_env = DummyVecEnv([
        make_env(**env_cfg)
    ])
    
    # 3. Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=train_cfg["checkpoint_freq"] // n_envs,
        save_path=model_dir,
        name_prefix="ppo_gridlock"
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(base_dir, "best_model"),
        log_path=log_dir,
        eval_freq=config["evaluation"]["eval_freq"] // n_envs,
        deterministic=True,
        render=False
    )
    
    # 4. Initialize or Load Model
    if load_model_path and os.path.exists(load_model_path):
        print(f"Loading pretrained model from: {load_model_path}")
        model = PPO.load(load_model_path, env=env, tensorboard_log=log_dir)
        
        # FORCE UPDATE HYPERPARAMETERS from new config
        # PPO.load preserves the old ones by default.
        model.ent_coef = train_cfg["ent_coef"]
        model.learning_rate = train_cfg["learning_rate"]
        model.learning_rate = train_cfg["learning_rate"]
        
        # Buffer Resizing Logic
        if model.n_steps != train_cfg["n_steps"]:
            print(f"Resizing rollout buffer from {model.n_steps} to {train_cfg['n_steps']}")
            model.n_steps = train_cfg["n_steps"]
            model.rollout_buffer.buffer_size = train_cfg["n_steps"]
            model.rollout_buffer.n_steps = train_cfg["n_steps"]
            model.rollout_buffer.reset() 
        
        model.batch_size = train_cfg["batch_size"]
        model.gamma = train_cfg["gamma"]
        model.gae_lambda = train_cfg["gae_lambda"]
        # SB3 requires clip_range to be a function (schedule)
        model.clip_range = lambda _: train_cfg["clip_range"]
        model.vf_coef = train_cfg["vf_coef"]
        model.max_grad_norm = train_cfg["max_grad_norm"]
        
        print(f"Updated loaded model hyperparameters to: ent_coef={model.ent_coef}, lr={model.learning_rate}")
    else:
        print("Initializing new PPO model")
        model = PPO(
            train_cfg["policy"],
            env,
            learning_rate=train_cfg["learning_rate"],
            n_steps=train_cfg["n_steps"],
            batch_size=train_cfg["batch_size"],
            n_epochs=train_cfg["n_epochs"],
            gamma=train_cfg["gamma"],
            gae_lambda=train_cfg["gae_lambda"],
            clip_range=train_cfg["clip_range"],
            ent_coef=train_cfg["ent_coef"],
            vf_coef=train_cfg["vf_coef"],
            max_grad_norm=train_cfg["max_grad_norm"],
            verbose=1,
            tensorboard_log=log_dir,
            device="auto"
        )
    
    print(f"Starting training: {run_name}")
    model.learn(
        total_timesteps=train_cfg["total_timesteps"],
        callback=[checkpoint_callback, eval_callback],
        reset_num_timesteps=False if load_model_path else True
    )
    
    # Save final
    model.save(os.path.join(model_dir, "final_model"))
    print("Training complete.")
    
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train/ppo.yaml")
    parser.add_argument("--run-name", type=str, default="ppo_baseline")
    parser.add_argument("--load-model", type=str, default=None, help="Path to pretrained model.zip")
    args = parser.parse_args()
    
    train(args.config, args.run_name, args.load_model)
