import numpy as np
import matplotlib.pyplot as plt

SPATIAL_TOLERANCE = 5.0
TEMPORAL_TOLERANCE = 0.005
TIME_WINDOW = 0.02

def test_prediction_hypothesis():
    combined_events = np.load("combined_events.npy")
    real_events = combined_events[combined_events[:, 4] == 0]
    predicted_events = combined_events[combined_events[:, 4] == 1]

    first_10_real = real_events[:100]
    all_matches = []

    for i, real_event in enumerate(first_10_real):
        x_real, y_real, p_real, t_real, _ = real_event
        matches = []
        time_min = t_real - TIME_WINDOW
        time_max = t_real + TIME_WINDOW
        time_mask = (predicted_events[:, 3] >= time_min) & (predicted_events[:, 3] <= time_max)
        candidate_pred_events = predicted_events[time_mask]

        for j, pred_event in enumerate(candidate_pred_events):
            x_pred, y_pred, p_pred, t_pred, _ = pred_event
            spatial_dist = np.sqrt((x_real - x_pred)**2 + (y_real - y_pred)**2)
            temporal_dist = abs(t_real - t_pred)
            if spatial_dist <= SPATIAL_TOLERANCE and temporal_dist <= TEMPORAL_TOLERANCE and p_real != p_pred:
                matches.append({'pred_index': j, 'spatial_dist': spatial_dist,
                                'temporal_dist': temporal_dist, 'pred_event': pred_event})
        
        matches.sort(key=lambda x: x['spatial_dist'])
        all_matches.append({'real_event': real_event, 'matches': matches})

    total_matches = 0
    for real_event in first_10_real:
        x_real, y_real, p_real, t_real, _ = real_event
        matches = 0
        time_min = t_real - TIME_WINDOW
        time_max = t_real + TIME_WINDOW
        time_mask = (predicted_events[:, 3] >= time_min) & (predicted_events[:, 3] <= time_max)
        candidate_pred_events = predicted_events[time_mask]
        for pred_event in candidate_pred_events:
            x_pred, y_pred, p_pred, t_pred, _ = pred_event
            spatial_dist = np.sqrt((x_real - x_pred)**2 + (y_real - y_pred)**2)
            temporal_dist = abs(t_real - t_pred)
            if spatial_dist <= SPATIAL_TOLERANCE and temporal_dist <= TEMPORAL_TOLERANCE and p_real != p_pred:
                matches += 1
        total_matches += matches

    fig, ax = plt.subplots(1, 1, figsize=(16, 9))
    real_x = [event[0] for event in first_10_real]
    real_y = [event[1] for event in first_10_real]
    real_polarity = [event[2] for event in first_10_real]

    for i, (x, y, p) in enumerate(zip(real_x, real_y, real_polarity)):
        color = 'red' if p == 0 else 'blue'
        ax.scatter(x, y, c=color, s=100, alpha=0.8, edgecolors='black', linewidth=1)
        ax.annotate(f'R{i+1}', (x, y), xytext=(5, 5), textcoords='offset points', fontsize=8, fontweight='bold')

    matched_pred_x = []
    matched_pred_y = []
    matched_pred_polarity = []

    for match_data in all_matches:
        for match in match_data['matches']:
            pred_event = match['pred_event']
            matched_pred_x.append(pred_event[0])
            matched_pred_y.append(pred_event[1])
            matched_pred_polarity.append(pred_event[2])

    for i, (x, y, p) in enumerate(zip(matched_pred_x, matched_pred_y, matched_pred_polarity)):
        color = 'orange' if p == 0 else 'cyan'
        ax.scatter(x, y, c=color, s=80, alpha=0.6, marker='s')
        ax.annotate(f'P{i+1}', (x, y), xytext=(5, -15), textcoords='offset points', fontsize=7, color='darkgreen')

    for match_data in all_matches:
        real_event = match_data['real_event']
        real_x, real_y = real_event[0], real_event[1]
        for match in match_data['matches']:
            pred_event = match['pred_event']
            pred_x, pred_y = pred_event[0], pred_event[1]
            distance = match['spatial_dist']
            alpha = max(0.1, 1.0 - distance / SPATIAL_TOLERANCE)
            ax.plot([real_x, pred_x], [real_y, pred_y], 'g-', alpha=alpha, linewidth=1)

    ax.set_xlim(0, 1280)
    ax.set_ylim(720, 0)
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    ax.grid(True, alpha=0.3)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', alpha=0.8, label='Real Events (OFF)'),
        Patch(facecolor='blue', alpha=0.8, label='Real Events (ON)'),
        Patch(facecolor='orange', alpha=0.6, label='Matched Predicted (OFF)'),
        Patch(facecolor='cyan', alpha=0.6, label='Matched Predicted (ON)')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    plt.tight_layout()
    plt.savefig('event_matching_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    test_prediction_hypothesis()
