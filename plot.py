import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Small-x dataset
x = np.array([20, 10, 5, 3, 2, 1], dtype=float)
y = np.array([0.95, 0.89, 0.8, 0.718, 0.658, 0.595], dtype=float)

# Exponential model
def exp_model(x, k1, k2, lam):
    return k1 + k2 * np.exp(-lam * x)

# Use the earlier good initial_guess
p0 = [min(y)*0.9, (max(y)-min(y))*1.1, 0.1]

# Fit
params, cov = curve_fit(exp_model, x, y, p0=p0, bounds=([-np.inf, -np.inf, 0], [np.inf, np.inf, np.inf]))
k1, k2, lam = params

# Smooth curve
xs = np.linspace(min(x), max(x), 400)
ys = exp_model(xs, k1, k2, lam)

# Plot
plt.figure(figsize=(6,4))
plt.scatter(x, y, label="data")
plt.plot(xs, ys, label=f"fit: y = {k1:.6f} + {k2:.6f} exp(-{lam:.6f} x)", linewidth=1.5)
plt.title("Small-x dataset — Restored Exponential Fit")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True, linestyle=":")
plt.legend()
plt.tight_layout()
plt.show()

# Print parameters
print("Restored exponential fit parameters (small-x dataset):")
print(f"  k1 = {k1}")
print(f"  k2 = {k2}")
print(f"  lambda = {lam}")