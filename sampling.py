import numpy as np

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


print(sample_list([2,3], 5))
"""
samples = sample_transformed_log(5)
for sample in samples:
    print(sample)
"""