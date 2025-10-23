import numpy as np
import cv2
from motion_model import load_event_data, load_center_from_tracker, apply_rotation

# Parameters
OMEGA_Z = np.array([0, 0, 2.5 * np.pi])
DT = 0.6
EVENTS_PATH = "sample.csv"
CENTER_PATH = "spin-dot_track.csv"
HEIGHT, WIDTH = 720, 1280
FPS = 100
dt_display = 1 / FPS

# Load data
events = load_event_data(EVENTS_PATH)
center = load_center_from_tracker(CENTER_PATH)

# Predict future positions
predicted_events = np.array([
    apply_rotation(x, y, center, OMEGA_Z[2], DT) for x, y, _, _ in events
])

# Prepare for visualization
image = np.zeros((HEIGHT, WIDTH, 3), dtype=np.float32)
ts_last_display = events[0, 3]
for idx, (orig, pred, event) in enumerate(zip(events, predicted_events, events)):
    x, y, p, t = orig
    x_pred, y_pred = pred
    x, y = int(x), int(y)
    x_pred, y_pred = int(x_pred), int(y_pred)

    # Draw original event in red
    if 0 <= x < WIDTH and 0 <= y < HEIGHT:
        image[y, x, 2] += 1  # Red channel

    # Draw predicted event in green
    if 0 <= x_pred < WIDTH and 0 <= y_pred < HEIGHT:
        image[y_pred, x_pred, 1] += 1  # Green channel

    # Display condition based on time difference
    if t - ts_last_display >= dt_display:
        ts_last_display = t

        # Normalize for display
        norm_img = image.copy()
        max_val = norm_img.max()
        if max_val > 0:
            norm_img /= max_val
        norm_img = (255 * norm_img).astype(np.uint8)

        cv2.imshow('Events: Red=Original, Green=Predicted', norm_img)
        key = cv2.waitKey(1)
        if key == 27:
            break

        image *= 0  # Reset for next frame

cv2.destroyAllWindows()