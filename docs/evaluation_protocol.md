# Evaluation Protocol

## Protocol
To ensure "various grid configurations" are measurable:

1. **Benchmark Map Bank**: 
    - A set of maps generated once with fixed seeds (e.g., N=100) and validated for solvability.
    - This set is kept stable across training iterations.

2. **Generalization Test**:
    - Evaluate on completely new, unseen random maps with new seeds to measure generalization.

## Metrics to Track
- **Success Rate**: % of episodes where agent reaches goal with all 3 keys.
- **Trap Rate**: % of episodes terminated by stepping on a trap.
- **Timeout Rate**: % of episodes reaching the step limit.
- **Efficiency**:
    - Mean steps to collect all keys.
    - Mean steps to finish after collecting 3rd key.

## Reproducibility Rules
- **Global Seed**: Set for the entire run.
- **Env Seed**: Set per episode.
- **Map Seed**: Optional, to regenerate identical grids.
- **Artifacts**: Store configs, git commit hash, and evaluation results on the fixed benchmark set with each run.
