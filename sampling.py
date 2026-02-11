import numpy as np
import pandas as pd
from scipy.stats import qmc

def sample_list(options:list, n_samples:int):
    rng = np.random.default_rng()
    return rng.choice(options, size=n_samples, replace=True)
   
    
def sample_transformed_log(n_samples:int, min_val:float=2, max_val:float=30)->list[float]:
    """
    Samples uniformly in the space of ln(1+D) and converts back to D.
    This results in more frequent sampling of smaller D values.
    """
    if min_val<0:
        raise ValueError(f"{min_val} cannot be negative")
    elif min_val > max_val:
        raise ValueError(f"min_val cannot be greater than max_val")
    # Define the bounds in the log-transformed space
    log_low = np.log(1 + min_val)
    log_high = np.log(1 + max_val)
    
    # Sample uniformly in the log space
    y = np.random.uniform(log_low, log_high, n_samples)
    
    # Transform back to the D space
    d_values = np.exp(y) - 1
    
    return d_values


def latin_hypercube(n_samples=60, seed=42):
    """
    Generates a Latin Hypercube Sample for pipe flow validation.
    
    Parameters:
    - n_samples: Total number of experimental runs (will be split between radii).
    - seed: For reproducibility.
    
    Returns:
    - pd.DataFrame: DOE matrix with Radius, Orientation, and Separation.
    """
    # 1. Configuration
    radii = [2, 3]
    samples_per_r = n_samples // len(radii)
    
    # Natural log boundaries for S/D (2 to 30)
    ln_min, ln_max = np.log(2), np.log(30)
    
    # 2. Initialize LHS Sampler (2D: Orientation and Separation)
    sampler = qmc.LatinHypercube(d=2, seed=seed)
    
    results = []
    
    for r in radii:
        # Generate raw samples in [0, 1]
        raw = sampler.random(n=samples_per_r)
        
        # Transform: Orientation (Linear 0-180)
        orient = raw[:, 0] * 180
        
        # Transform: Separation (Natural Log 2-30)
        # Formula: exp( ln_min + (normalized_val * (ln_max - ln_min)) )
        sep = np.exp(ln_min + raw[:, 1] * (ln_max - ln_min))
        
        df_r = pd.DataFrame({
            'Bend_Radius_D': r,
            'Orientation_deg': orient,
            'Separation_D': sep
        })
        results.append(df_r)
    
    # 3. Combine and Shuffle
    # Shuffling ensures that you don't run all 'Radius 2' cases then all 'Radius 3'
    df_final = pd.concat(results).sample(frac=1, random_state=seed).reset_index(drop=True)
    
    return df_final

samples = sample_list([(0.01,90),(0.01,270), (180,90), (180, 270)], 20)
samples = sample_list([20,30], 20)
#samples = sample_transformed_log(10, 2, 30)
for sample in samples:
    print(sample)

