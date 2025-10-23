# test_rotation_model.py

import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import ransac, CircleModel
from scipy.spatial.distance import cdist

# Constants
OMEGA = np.array([0, 0, 2.5 * np.pi])  # Angular velocity as a 3D vector
DT = 0.7                              # Predict 100 ms into the future

def load_event_data(path):
    """Load event data and convert time from microseconds to seconds, skip bad lines."""
    clean_lines = []
    with open(path, 'r') as f:
        for i, line in enumerate(f):
            parts = line.strip().split(',')
            if len(parts) != 4:
                print(f"Skipping line {i}: Incorrect number of columns")
                continue
            try:
                x, y, p, t = map(float, parts)
                clean_lines.append([x, y, p, t / 1e6])  # convert time
            except ValueError:
                print(f"Skipping line {i}: Could not convert to float → {parts}")
                continue
    return np.array(clean_lines)

def load_center_from_tracker(path):
    """Load center coordinates using RANSAC circle fitting"""
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    points = data[:, 1:3]
    model, _ = ransac(points, CircleModel, min_samples=3, residual_threshold=2.0, max_trials=1000)
    x0, y0, _ = model.params
    return np.array([x0, y0, 0])

def apply_rotation_3d(pos, center, omega, dt):
    """Apply 3D rotation to a point `pos` using angular velocity `omega` and time `dt`."""
    r = pos - center
    theta = np.linalg.norm(omega) * dt
    if theta == 0:
        return pos
    k = omega / np.linalg.norm(omega)

    # Rodrigues' rotation formula
    r_rot = (r * np.cos(theta) +
             np.cross(k, r) * np.sin(theta) +
             k * (np.dot(k, r)) * (1 - np.cos(theta)))
    
    return center + r_rot

def apply_rotation_2d_simple(x, y, center, omega_z, dt):
    """Apply 2D rotation transformation to predict future event position"""
    r = np.array([x, y, 0]) - center
    theta = omega_z * dt
    rot = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])
    rotated = rot @ r[:2] + center[:2]
    return rotated

def main():
    # Test over a range of DT values
    dt_values = np.linspace(0.05, 1.0, 10)
    for DT in dt_values:
        print(f"\n{'='*30}\nTesting for DT = {DT:.3f} s\n{'='*30}")
        # All code inside main() should be indented and run for each DT
        # Load real data
        events = load_event_data("sample.csv")
        center = load_center_from_tracker("spin-dot_track.csv")
        sample_events = events[:1000]
        original_points = sample_events[:, :3]
        predicted_points = np.array([apply_rotation_3d(p, center, OMEGA, DT) for p in original_points])

        # Quantitative Evaluation
        euclidean_distances = np.linalg.norm(predicted_points[:, :2] - original_points[:, :2], axis=1)
        avg_distance = np.mean(euclidean_distances)
        std_distance = np.std(euclidean_distances)
        print(f"Average Euclidean distance (pixels): {avg_distance:.2f} ± {std_distance:.2f}")

        def angle_from_center(pt, center):
            return np.arctan2(pt[1] - center[1], pt[0] - center[0])
        original_angles = np.array([angle_from_center(p, center) for p in original_points])
        predicted_angles = np.array([angle_from_center(p, center) for p in predicted_points])
        angle_diff = (predicted_angles - original_angles + np.pi) % (2 * np.pi) - np.pi
        avg_angle = np.mean(angle_diff)
        std_angle = np.std(angle_diff)
        print(f"Average angular displacement (radians): {avg_angle:.4f} ± {std_angle:.4f}")
        print(f"Expected angular displacement: {np.linalg.norm(OMEGA) * DT:.4f} radians")

        original_radii = np.linalg.norm(original_points[:, :2] - center[:2], axis=1)
        predicted_radii = np.linalg.norm(predicted_points[:, :2] - center[:2], axis=1)
        radii_diff = predicted_radii - original_radii
        avg_radii_diff = np.mean(radii_diff)
        std_radii_diff = np.std(radii_diff)
        print(f"Average change in radius: {avg_radii_diff:.4f} ± {std_radii_diff:.4f} pixels (should be ~0)")

        
        print("\n--- Additional Rigorous Checks ---")
        # 1. Inverse Consistency Test
        errors = []
        for p in original_points:
            p_rot = apply_rotation_3d(p, center, OMEGA, DT)
            p_inv = apply_rotation_3d(p_rot, center, -OMEGA, DT)
            errors.append(np.linalg.norm(p - p_inv))
        errors = np.array(errors)
        print(f"Inverse consistency error (should be ~0): mean={errors.mean():.2e}, max={errors.max():.2e}")

        # 2. Compare 3D rotation to 2D rotation for z-axis
        def apply_rotation_2d(pos, center, omega_z, dt):
            r = pos[:2] - center[:2]
            theta = omega_z * dt
            rot = np.array([
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta),  np.cos(theta)]
            ])
            r_rot = rot @ r
            return np.array([center[0] + r_rot[0], center[1] + r_rot[1], 0])

        diffs_2d_3d = []
        for p in original_points:
            p_3d = apply_rotation_3d(p, center, OMEGA, DT)
            p_2d = apply_rotation_2d(p, center, OMEGA[2], DT)
            diffs_2d_3d.append(np.linalg.norm(p_3d - p_2d))
        diffs_2d_3d = np.array(diffs_2d_3d)
        print(f"3D vs 2D rotation error (should be ~0): mean={diffs_2d_3d.mean():.2e}, max={diffs_2d_3d.max():.2e}")

        # 3. Rotation matrix orthogonality and determinant
        theta = np.linalg.norm(OMEGA) * DT
        k = OMEGA / np.linalg.norm(OMEGA)
        K = np.array([
            [0, -k[2], k[1]],
            [k[2], 0, -k[0]],
            [-k[1], k[0], 0]
        ])
        R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
        ortho_error = np.linalg.norm(R.T @ R - np.eye(3))
        det_R = np.linalg.det(R)
        print(f"Rotation matrix orthogonality error (should be ~0): {ortho_error:.2e}")
        print(f"Rotation matrix determinant (should be 1): {det_R:.6f}")

        # Compare 2D and 3D models for all points
        diffs_2d_simple_3d = []
        for p in original_points:
            p_3d = apply_rotation_3d(p, center, OMEGA, DT)
            p_2d_simple = apply_rotation_2d_simple(p[0], p[1], center, OMEGA[2], DT)
            # Compare only x, y
            diffs_2d_simple_3d.append(np.linalg.norm(p_3d[:2] - p_2d_simple))
        diffs_2d_simple_3d = np.array(diffs_2d_simple_3d)
        print(f"2D (simple) vs 3D rotation error (should be ~0): mean={diffs_2d_simple_3d.mean():.2e}, max={diffs_2d_simple_3d.max():.2e}")

        # Optionally, plot for a few DTs (e.g., first, middle, last)
        if DT in [dt_values[0], dt_values[len(dt_values)//2], dt_values[-1]]:
            fig, axs = plt.subplots(1, 3, figsize=(18, 5))
            axs[0].hist(euclidean_distances, bins=30, color='purple', alpha=0.7)
            axs[0].set_title('Euclidean Distance (Original vs Predicted)')
            axs[0].set_xlabel('Distance (pixels)')
            axs[0].set_ylabel('Count')

            axs[1].hist(angle_diff, bins=30, color='orange', alpha=0.7)
            axs[1].set_title('Angular Displacement')
            axs[1].set_xlabel('Angle Difference (radians)')
            axs[1].set_ylabel('Count')

            axs[2].hist(radii_diff, bins=30, color='green', alpha=0.7)
            axs[2].set_title('Change in Radius')
            axs[2].set_xlabel('Radius Difference (pixels)')
            axs[2].set_ylabel('Count')
            plt.tight_layout()
            plt.show()

            # Quiver plot for a subset of points
            subset = 50 if len(original_points) > 50 else len(original_points)
            idx = np.random.choice(len(original_points), subset, replace=False)
            plt.figure(figsize=(10,10))
            plt.quiver(original_points[idx,0], original_points[idx,1],
                       predicted_points[idx,0] - original_points[idx,0],
                       predicted_points[idx,1] - original_points[idx,1],
                       angles='xy', scale_units='xy', scale=1, color='teal', width=0.003, alpha=0.7)
            plt.scatter(original_points[idx,0], original_points[idx,1], color='red', label='Original', alpha=0.7)
            plt.scatter(predicted_points[idx,0], predicted_points[idx,1], color='green', label='Predicted', alpha=0.7)
            plt.scatter(*center[:2], c='blue', label='Center', s=100)
            plt.legend()
            plt.gca().invert_yaxis()
            plt.title(f'Quiver Plot: Original to Predicted (subset)\nDT={DT:.3f}s')
            plt.axis('equal')
            plt.grid(True)
            plt.show()

if __name__ == "__main__":
    main()
