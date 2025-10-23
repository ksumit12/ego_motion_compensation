import numpy as np
from motion_model import load_event_data, load_center_from_tracker, apply_rotation
import cv2

# Parameters
OMEGA_Z = np.array([0, 0, 2.5 * np.pi])  # Angular velocity (z-axis)
DT = 0.1                                 # Prediction time (seconds)
EVENTS_PATH = "sample.csv"
CENTER_PATH = "spin-dot_track.csv"
HEIGHT, WIDTH = 720, 1280
FPS = 100
dt_display = 1 / FPS

if __name__ == "__main__":
    # Load and sort data
    events = load_event_data(EVENTS_PATH)
    events = events[np.argsort(events[:, 3])]  # Sort by timestamp
    center = load_center_from_tracker(CENTER_PATH)
    sample_events = events[:1000]
    original_points = sample_events[:, :3]  # x, y, z (z will be 0)

    # Predict their future positions using the 2D model
    predicted_points = np.array([
        apply_rotation(p[0], p[1], center, OMEGA_Z[2], DT) for p in original_points
    ])

    # Quantitative Evaluation
    # 1. Euclidean distance between original and predicted points
    euclidean_distances = np.linalg.norm(predicted_points - original_points[:, :2], axis=1)
    avg_distance = np.mean(euclidean_distances)
    std_distance = np.std(euclidean_distances)
    print(f"Average Euclidean distance (pixels): {avg_distance:.2f} ± {std_distance:.2f}")

    # 2. Angular displacement for each event around the center
    def angle_from_center(pt, center):
        return np.arctan2(pt[1] - center[1], pt[0] - center[0])
    original_angles = np.array([angle_from_center(p, center) for p in original_points])
    predicted_angles = np.array([angle_from_center(np.append(p, 0), center) for p in predicted_points])
    angle_diff = (predicted_angles - original_angles + np.pi) % (2 * np.pi) - np.pi
    avg_angle = np.mean(angle_diff)
    std_angle = np.std(angle_diff)
    print(f"Average angular displacement (radians): {avg_angle:.4f} ± {std_angle:.4f}")
    print(f"Expected angular displacement: {np.linalg.norm(OMEGA_Z) * DT:.4f} radians")

    # 3. Histogram of distances from center (should be preserved)
    original_radii = np.linalg.norm(original_points[:, :2] - center[:2], axis=1)
    predicted_radii = np.linalg.norm(predicted_points - center[:2], axis=1)
    radii_diff = predicted_radii - original_radii
    avg_radii_diff = np.mean(radii_diff)
    std_radii_diff = np.std(radii_diff)
    print(f"Average change in radius: {avg_radii_diff:.4f} ± {std_radii_diff:.4f} pixels (should be ~0)")

    # 4. Inverse Consistency Test
    errors = []
    for p in original_points:
        p_pred = apply_rotation(p[0], p[1], center, OMEGA_Z[2], DT)
        p_inv = apply_rotation(p_pred[0], p_pred[1], center, -OMEGA_Z[2], DT)
        errors.append(np.linalg.norm(p[:2] - p_inv))
    errors = np.array(errors)
    print(f"Inverse consistency error (should be ~0): mean={errors.mean():.2e}, max={errors.max():.2e}")

    # 5. 2D/3D Equivalence Test (if you have apply_rotation_3d)
    try:
        from motion_model import apply_rotation_3d
        diffs_2d_3d = []
        for p in original_points:
            p_3d = apply_rotation_3d(p, center, OMEGA_Z, DT)
            p_2d = apply_rotation(p[0], p[1], center, OMEGA_Z[2], DT)
            diffs_2d_3d.append(np.linalg.norm(p_3d[:2] - p_2d))
        diffs_2d_3d = np.array(diffs_2d_3d)
        print(f"2D vs 3D rotation error (should be ~0): mean={diffs_2d_3d.mean():.2e}, max={diffs_2d_3d.max():.2e}")
    except ImportError:
        pass

    # 6. Rotation Matrix Orthogonality and Determinant
    theta = np.linalg.norm(OMEGA_Z) * DT
    k = OMEGA_Z / np.linalg.norm(OMEGA_Z)
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

    # 7. Improved OpenCV visualization: accumulate and display both original and predicted events
    image = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    window_size = 100  # Number of events per frame

    for i in range(0, len(original_points), window_size):
        image[:] = 0
        for p in original_points[i:i+window_size]:
            x, y = int(p[0]), int(p[1])
            if 0 <= x < WIDTH and 0 <= y < HEIGHT:
                image[y, x, 2] = 255
        for p in predicted_points[i:i+window_size]:
            x, y = int(p[0]), int(p[1])
            if 0 <= x < WIDTH and 0 <= y < HEIGHT:
                image[y, x, 1] = 255
        cv2.imshow('Test Events Animation', image)
        key = cv2.waitKey(50)
        if key == 27:
            break

    cv2.destroyAllWindows()

    # Print mean distance for this subset
    print("Mean distance between original and predicted (sample_events):", np.mean(np.linalg.norm(predicted_points - original_points[:, :2], axis=1))) 