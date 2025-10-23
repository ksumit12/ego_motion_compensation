import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import ransac, CircleModel
from tqdm import trange

def ransac_center(points, residual_threshold=2.0, max_trials=1000):
    model, inliers = ransac(points, CircleModel, min_samples=3, residual_threshold=residual_threshold, max_trials=max_trials)
    if model is None:
        raise RuntimeError("RANSAC failed")
    return np.array(model.params), inliers

def naive_circle_fit(points):
    model = CircleModel()
    model.estimate(points)
    return np.array(model.params)

# Load tracker data
tracker_path = "spin-dot_track.csv"
data = np.loadtxt(tracker_path, delimiter=",", skiprows=1)
points = data[:, 1:3]

# Baseline RANSAC full fit
base_params, inliers = ransac_center(points)
x0_base, y0_base, r_base = base_params
print(f"Base RANSAC Center: ({x0_base:.3f}, {y0_base:.3f}), Radius: {r_base:.3f}")

# Repeat RANSAC multiple times
num_trials = 50
centers = []
radii = []

for i in trange(num_trials, desc="Testing RANSAC Stability"):
    # Optional: subsample points to simulate variation
    subset = points[np.random.choice(len(points), size=int(0.8 * len(points)), replace=False)]
    params, _ = ransac_center(subset)
    centers.append(params[:2])
    radii.append(params[2])

centers = np.array(centers)
radii = np.array(radii)

mean_center = centers.mean(axis=0)
std_center = centers.std(axis=0)
mean_radius = radii.mean()
std_radius = radii.std()

print("\n--- RANSAC Stability Check ---")
print(f"Mean Center:  ({mean_center[0]:.3f}, {mean_center[1]:.3f})")
print(f"Std Dev:      ({std_center[0]:.4f}, {std_center[1]:.4f}) pixels")
print(f"Mean Radius:  {mean_radius:.3f}, Std Dev: {std_radius:.4f} pixels")

# Compare with naive circle fit (no outlier handling)
naive_params = naive_circle_fit(points)
x_naive, y_naive, r_naive = naive_params
print("\n--- Naive Fit (No RANSAC) ---")
print(f"Naive Center: ({x_naive:.3f}, {y_naive:.3f}), Radius: {r_naive:.3f}")

# Plot distribution of centers
plt.figure(figsize=(6, 6))
plt.scatter(centers[:, 0], centers[:, 1], c='teal', label='RANSAC Centers', alpha=0.6)
plt.scatter(*base_params[:2], c='blue', marker='x', s=100, label='Base RANSAC')
plt.scatter(x_naive, y_naive, c='red', marker='*', s=150, label='Naive Fit')
plt.title("Center Estimate Distribution (50 RANSAC Runs)")
plt.legend()
plt.axis('equal')
plt.gca().invert_yaxis()
plt.grid(True)
plt.show()
