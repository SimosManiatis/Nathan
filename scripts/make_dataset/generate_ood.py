import yaml
import numpy as np
import os
from gridlock_rl.maps.generator import MapGenerator

def generate_ood(n_maps=50, output_path="configs/maps/benchmark_ood_seeds.yaml"):
    print(f"Generating {n_maps} OOD benchmark seeds (10x10, 0.15 Traps)...")
    
    # OOD Settings: Harder than anything seen in training
    width = 10
    height = 10
    trap_density = 0.15 
    
    gen = MapGenerator(width=width, height=height, trap_density=trap_density)
    
    valid_seeds = []
    candidate = 10000 # Start far from ID seeds
    count = 0
    
    while count < n_maps:
        try:
            gen.generate(seed=candidate)
            valid_seeds.append(candidate)
            count += 1
        except Exception:
            pass
        candidate += 1
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w") as f:
        yaml.dump({
            "seeds": valid_seeds,
            "config": {
                "width": width,
                "height": height,
                "trap_density": trap_density
            }
        }, f)
        
    print(f"Saved {len(valid_seeds)} OOD seeds to {output_path}")

if __name__ == "__main__":
    generate_ood()
