import numpy as np
import pandas as pd
from scipy.stats import qmc
import meshes

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


def latin_hypercube_triplets(n_samples=60, seed=43):
    """
    Generates a DOE where each row is a 'triplet' data point:
    - 3 Random Radii (Selected from [2, 3] or a range)
    - 2 Random Separations (Natural Log 2-30)
    - 2 Random Orientations (Linear 0-180)
    """
    # 1. Configuration
    # We need 7 dimensions: 3 for R, 2 for S, 2 for O
    sampler = qmc.LatinHypercube(d=7, seed=seed)
    raw = sampler.random(n=n_samples)
    
    # Natural log boundaries for S/D (2 to 30)
    ln_min, ln_max = np.log(2), np.log(30)
    
    # 2. Extract and Transform Dimensions
    # Radii: Assuming you want them picked from the [2, 3] set randomly
    # We use the raw [0,1] value to pick 2 or 3
    r1 = np.where(raw[:, 0] < 0.5, 20, 30)
    r2 = np.where(raw[:, 1] < 0.5, 20, 30)
    r3 = np.where(raw[:, 2] < 0.5, 20, 30)
    
    # Separations: New Formula (Natural Log)
    s1 = 10*np.exp(ln_min + raw[:, 3] * (ln_max - ln_min))
    s2 = 10*np.exp(ln_min + raw[:, 4] * (ln_max - ln_min))
    
    # Orientations: Linear 0-180
    orientations = []
    o1 = raw[:, 5] * 180
    o2 = raw[:, 6] * 180
    for or1,or2 in zip(o1,o2):
        orientations.append(meshes.twist_to_cad(or1, or2))
    # 3. Build the DataFrame
    df = pd.DataFrame({
        'R1': r1, 'R2': r2, 'R3': r3,
        'Sep1': s1, 'Sep2': s2,
        'Orientations': orientations,
        'Twist angles': zip(o1,o2)
    })
    
    return df

print(latin_hypercube_triplets(20))
#points = sample_transformed_log(6)
#for point in points:
#    print(point)
