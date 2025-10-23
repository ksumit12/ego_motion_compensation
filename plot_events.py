import matplotlib.pyplot as plt
import numpy as np

# ================== CONFIG ==================
TRACK_PATH = r"/home/sumit/anu_research/ego_motion/AEB_tracker/perlin_1280hz_hand_outframe_2.csv"
SAMPLE_SIZE = 20  # Number of samples to show

def load_tracking_data():
    try:
        data = np.loadtxt(TRACK_PATH, delimiter=",", skiprows=1)
        print(f"Loaded with loadtxt: {data.shape}")
        return data
    except:
        try:
            data = np.loadtxt(TRACK_PATH, delimiter=",")
            print(f"Loaded without skiprows: {data.shape}")
            return data
        except:
            print("Trying manual parsing...")
            with open(TRACK_PATH, 'r') as f:
                lines = f.readlines()
            
            parsed = []
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(',')
                vals = []
                for p in parts:
                    for s in p.split():
                        try:
                            vals.append(float(s))
                        except:
                            pass
                if len(vals) >= 3:
                    parsed.append(vals[:3])
            
            if parsed:
                data = np.array(parsed)
                print(f"Manual parse: {data.shape}")
                return data
            else:
                raise ValueError("Could not parse file")

def analyze_tracking_data():
    """Analyze the tracking data to understand the format and compute real-time angular velocity"""
    
    print("=== ANALYZING TRACKING DATA ===")
    
    try:
        tracker_data = load_tracking_data()
        print(f"Data shape: {tracker_data.shape}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    if len(tracker_data.shape) == 1:
        tracker_data = tracker_data.reshape(1, -1)
    
    if tracker_data.shape[1] < 3:
        print(f"Need at least 3 columns, got {tracker_data.shape[1]}")
        return
    
    print(f"First {min(SAMPLE_SIZE, len(tracker_data))} samples:")
    print("Time(s)    X(px)    Y(px)")
    print("-" * 35)
    
    for i in range(min(SAMPLE_SIZE, len(tracker_data))):
        row = tracker_data[i]
        if len(row) >= 3:
            print(f"{row[0]:8.3f}  {row[1]:8.1f}  {row[2]:8.1f}")
    
    ts = tracker_data[:, 0]
    px = tracker_data[:, 1] 
    py = tracker_data[:, 2]
    
    print(f"\nTime range: {ts[0]:.3f}s to {ts[-1]:.3f}s")
    print(f"Duration: {ts[-1] - ts[0]:.2f}s")
    print(f"X range: {px.min():.1f} to {px.max():.1f}")
    print(f"Y range: {py.min():.1f} to {py.max():.1f}")
    
    if len(ts) > 1:
        dt_avg = np.mean(np.diff(ts))
        print(f"Avg time step: {dt_avg:.4f}s ({1/dt_avg:.1f} Hz)")

if __name__ == "__main__":
    analyze_tracking_data() 