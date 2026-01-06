import gymnasium as gym
import numpy as np
from gridlock_rl.envs.grid_env import GridEnv
import time

def run_verification(n_episodes=100):
    print(f"Running {n_episodes} episodes with random policy...")
    env = GridEnv(width=8, height=8, trap_density=0.1)
    
    successes = 0
    traps = 0
    timeouts = 0
    total_steps = 0
    
    start_time = time.time()
    
    for episode in range(n_episodes):
        obs, info = env.reset(seed=episode) # Use episode index as seed for coverage
        terminated = False
        truncated = False
        steps = 0
        
        while not (terminated or truncated):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1
            
        total_steps += steps
        event = info["event"]
        
        if event == "success":
            successes += 1
        elif event == "trap":
            traps += 1
        elif event == "timeout":
            timeouts += 1
            
    elapsed = time.time() - start_time
    
    print("\nVerification Results:")
    print(f"Episodes: {n_episodes}")
    print(f"Time: {elapsed:.2f}s")
    print(f"Success Rate: {successes}/{n_episodes} ({successes/n_episodes*100:.1f}%)")
    print(f"Trap Rate: {traps}/{n_episodes} ({traps/n_episodes*100:.1f}%)")
    print(f"Timeout Rate: {timeouts}/{n_episodes} ({timeouts/n_episodes*100:.1f}%)")
    print(f"Avg Steps: {total_steps/n_episodes:.1f}")
    
    # Sanity check ASCII render of last episode final state
    print("\nLast Episode Final State:")
    env.render()
    
    return True

if __name__ == "__main__":
    run_verification()
