# Gridlock RL: Training "Nathan"

**Gridlock RL** is a research initiative focused on the development of **[Nathan](NATHAN_PROFILE.md)**, an autonomous AI agent designed to evolve into an intelligent Non-Player Character (NPC).

Currently undergoing the "Gridlock Protocol", Nathan is learning to navigate procedurally generated hazard zones, demonstrating advanced spatial reasoning and survival instincts. This repository documents his training architecture, curriculum progress, and the underlying PPO implementation.

> ** Meet the Agent**: Read Nathan's full biography and roadmap in **[NATHAN_PROFILE.md](NATHAN_PROFILE.md)**.

The project demonstrates advanced RL techniques including **Curriculum Learning**, **Dense Reward Shaping** (with potential-based guarantees), and **Vectorized Training** using Stable Baselines 3 (PPO).

![Gridlock Environment Concept](https://via.placeholder.com/800x400?text=Gridlock+RL+Environment+Visualization)
*(Replace with actual screenshot if available)*

## 🚀 Features

*   **Procedural Environments**: Infinite variation of maps generated on the fly with guaranteed solvability (verified via BFS).
*   **Curriculum Learning**: Structured training pipeline advancing from simple navigation (Stage 0) to complex, hazardous 8x8 forced-trap puzzles (Stage 2C).
*   **Robust Reward Shaping**: Implements potential-based shaping with "Target Tracking" to handle multi-objective sequences without reward hacking.
*   **High Performance**: Vectorized training (`SubprocVecEnv`) supports thousands of steps per second.
*   **Comprehensive Tooling**: Includes scripts for Oracle verification, Policy Visualization, and Batch Evaluation.

## 🛠️ Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/gridlock-rl.git
    cd gridlock-rl
    ```

2.  **Set up the environment**:
    It is recommended to use a virtual environment (Conda or venv).
    ```bash
    # Install dependencies
    pip install stable-baselines3 gymnasium numpy pygame pyyaml tensorboard
    ```

3.  **Set PYTHONPATH**:
    Ensure the `src` directory is in your Python path.
    ```powershell
    # Windows PowerShell
    $env:PYTHONPATH="src"
    ```
    ```bash
    # Linux/Mac
    export PYTHONPATH=src
    ```

## 🏃 Usage

### Training
The core training script uses PPO. Configuration is managed via YAML files in `configs/train/`.

**Run standard training (Stage 2C):**
```bash
python src/gridlock_rl/training/train_sb3.py --config configs/train/ppo.yaml --run-name my_experiment
```

**Resume from a checkpoint:**
```bash
python src/gridlock_rl/training/train_sb3.py --config configs/train/ppo.yaml --load-model runs/stage2a/models/final_model.zip
```

### Evaluation
Evaluate a trained model's performance metrics (Success Rate, Termination Breakdown).

```bash
python scripts/eval_model.py --model runs/my_experiment/models/final_model.zip
```
*Note: The script automatically loads the appropriate configuration derived from the training settings.*

### Visualization (Debug)
Watch the agent play in real-time.

```bash
python scripts/debug_policy.py --model runs/my_experiment/models/final_model.zip
```

## 📚 Curriculum Stages

The agent is trained through a rigorous curriculum:

| Stage | Description | Key Challenge | Status |
| :--- | :--- | :--- | :--- |
| **Stage 0** | 6x6, 1 Key, Safe | Basic Navigation | ✅ Solved (93%) |
| **Stage 0B** | 6x6, 1 Key, Traps | Hazard Avoidance | ✅ Solved (91%) |
| **Stage 1** | 6x6, 3 Keys, Safe | Multi-Objective TSP | ✅ Solved (93%) |
| **Stage 1B** | 6x6, 3 Keys, Traps | Combined Complexity | ✅ Solved (88%) |
| **Stage 2** | 8x8, 3 Keys, Traps | Scalability | 🔄 In Progress |

## 📂 Project Structure

```text
d:\Nathan
├── configs/              # Hyperparameter configurations (YAML)
│   └── train/            # PPO config (learning rate, steps, etc.)
├── scripts/              # Executable tools
│   ├── eval_model.py     # Metrics & Evaluation
│   ├── debug_policy.py   # Visualizer
│   └── oracle_rollout.py # Solvability verification
├── src/
│   └── gridlock_rl/
│       ├── envs/         # Gymnasium Environment logic
│       ├── maps/         # Procedural Map Generation
│       └── training/     # SB3 Training Loop
└── runs/                 # Training artifacts (Models & Logs)
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
