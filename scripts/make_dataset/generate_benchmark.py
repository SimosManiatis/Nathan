import yaml
import numpy as np
import os
from gridlock_rl.maps.generator import MapGenerator

def generate_benchmark(n_maps=100, output_path="configs/maps/benchmark_seeds.yaml"):
    print(f"Generating {n_maps} benchmark seeds...")
    
    # We store valid seeds that produce solvable maps.
    # Since MapGenerator guarantees solvability or raises error, 
    # we just need to confirm we can generate N distinct seeds.
    
    valid_seeds = []
    # We use a deterministic sequence of seeds to check, 
    # but only keep ones that don't error (though our generator retries internally).
    
    # Actually, MapGenerator.generate(seed=X) resets the RNG with X.
    # So we just need a list of integers.
    # We will verify them just in case.
    
    gen = MapGenerator(width=8, height=8, trap_density=0.1)
    
    count = 0
    candidate = 0
    while count < n_maps:
        try:
            # Check if this seed works (it should, generator retries)
            gen.generate(seed=candidate)
            valid_seeds.append(candidate)
            count += 1
        except Exception as e:
            print(f"Seed {candidate} failed: {e}")
        candidate += 1
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w") as f:
        yaml.dump({"seeds": valid_seeds}, f)
        
    print(f"Saved {len(valid_seeds)} seeds to {output_path}")

if __name__ == "__main__":
    generate_benchmark()
