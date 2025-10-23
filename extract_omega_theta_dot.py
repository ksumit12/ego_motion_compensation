#!/usr/bin/env python3
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import os
import glob

# folders
TRACKER_FOLDER = "AEB_tracker"
OUTPUT_CSV_FOLDER = "results_csv"
OUTPUT_PLOT_FOLDER = "results_plots"
OUTPUT_PX_FOLDER = "results_px_analysis"

# circle-fit params (simple)
WINDOW0_S   = 0.30      # start window (s)
STEP_S      = 0.01      # slide step (s)
W_MAX_S     = 0.80      # cap (s)
GROW        = 1.20      # grow factor
MIN_POINTS  = 300
R_MIN, R_MAX = 50.0, 500.0
INLIER_TOL  = 5.0       # px
MIN_INLIERS = 200
MAX_RMS     = 2.5       # px
MIN_ARC_DEG = 90.0      # want about a quarter turn
rng = np.random.default_rng(0)

#  basic IO
def _fix_t(t):
    t = np.asarray(t, float)
    m = t.max()
    if m > 1e12: t *= 1e-9
    elif m > 1e6: t *= 1e-6
    elif m > 6e4 and np.median(np.diff(np.sort(t))) > 1: t *= 1e-3
    return t

def read_positions(path):
    rows=[]
    for line in open(path):
        p=line.strip().split(","); 
        if len(p)<2: continue
        try: ts=float(p[0]); 
        except: continue
        s=p[1].split()
        if len(s)<2: continue
        try: rows.append((ts,float(s[0]),float(s[1])))
        except: pass
    a=np.asarray(rows,float); t=_fix_t(a[:,0]); x=a[:,1]; y=a[:,2]
    o=np.argsort(t); return t[o],x[o],y[o]

def read_theta_dot(path, assume_deg=False, warmup_s=0.15):
    times = []
    values = []

    # Read file line by line
    with open(path) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 2:
                continue

            # First column = timestamp
            try:
                ts = float(parts[0])
            except ValueError:
                continue

            # Second column has several space-separated numbers
            cols = parts[1].split()
            if len(cols) < 6:
                continue

            # Take the 6th number (index 5) = theta_dot
            try:
                val = float(cols[5])
            except ValueError:
                continue

            times.append(ts)
            values.append(val)

    # Convert to arrays
    t = np.asarray(times, float)
    w = np.asarray(values, float)

    # Fix time units (ns → s, ms → s, etc.)
    t = _fix_t(t)

    # Convert to radians if values were in degrees
    if assume_deg:
        w = np.deg2rad(w)

    # Sort by time (just in case file is unsorted)
    order = np.argsort(t)
    t = t[order]
    w = w[order]

    # Skip first "warm-up" period
    if warmup_s > 0:
        mask = t >= (t[0] + warmup_s)
        if mask.sum() >= 10:   # keep only if enough points left
            t = t[mask]
            w = w[mask]

    return t, w


# --- simple circle fit + RANSAC ---
def fit_circle(x0,y0,x1,y1,x2,y2):
    A=np.array([[2*(x1-x0),2*(y1-y0)],
                [2*(x2-x0),2*(y2-y0)]],float)
    b=np.array([x1*x1+y1*y1-x0*x0-y0*y0,
                x2*x2+y2*y2-x0*x0-y0*y0],float)
    if abs(np.linalg.det(A))<1e-12: return None
    cx,cy=np.linalg.solve(A,b); r=np.hypot(x0-cx,y0-cy); return cx,cy,r

def ransac_circle(x,y):
    n=len(x); 
    if n<3: return None
    best=None; best_in=-1
    for _ in range(200):
        i0,i1,i2=rng.choice(n,3,replace=False)
        res=fit_circle(x[i0],y[i0],x[i1],y[i1],x[i2],y[i2])
        if res is None: continue
        cx,cy,r=res
        if not (R_MIN<=r<=R_MAX): continue
        d=np.hypot(x-cx,y-cy); resid=np.abs(d-r)
        inliers=resid<=INLIER_TOL; nin=int(inliers.sum())
        if nin>best_in:
            rms=np.sqrt(np.mean(resid[inliers]**2)) if nin>=3 else np.inf
            best_in=nin; best=(cx,cy,r,nin,rms,inliers)
    return best

# θ = atan2(y-cy,x-cx), ω = dθ/dt (slope of line through [t,θ])
def omega_from_inliers(tw,xw,yw,cx,cy,mask):
    tw=tw[mask]; xw=xw[mask]; yw=yw[mask]
    if len(tw)<5: return None,0.0
    theta=np.unwrap(np.arctan2(yw-cy,xw-cx))
    arc=np.degrees(theta.max()-theta.min())
    if arc<MIN_ARC_DEG: return None,arc
    t0=tw[0]; slope=np.polyfit(tw-t0,theta,1)[0]
    return float(slope), arc

# --- adaptive extractor (grow window until arc+quality met) ---
def extract_series(t,x,y):
    out=[]
    left=t[0]; tend=t[-1]
    while left <= tend - WINDOW0_S:
        w=max(WINDOW0_S,0.10); wmax=W_MAX_S; row=None
        while left+w <= tend and w <= wmax:
            right=left+w
            i0=np.searchsorted(t,left,'left'); i1=np.searchsorted(t,right,'right')
            if i1-i0 < MIN_POINTS: w*=GROW; continue
            tw, xw, yw = t[i0:i1], x[i0:i1], y[i0:i1]
            rans=ransac_circle(xw,yw)
            if rans is None: w*=GROW; continue
            cx,cy,r,nin,rms,mask=rans
            if nin<MIN_INLIERS or rms>MAX_RMS: w*=GROW; continue
            om,arc=omega_from_inliers(tw,xw,yw,cx,cy,mask)
            if om is None or not np.isfinite(om): w*=GROW; continue
            row=dict(timestamp=0.5*(left+right),
                     window_left=left, window_right=right,
                     center_x=cx, center_y=cy, radius=r,
                     omega_circlefit_rad_s=om,
                     num_points=i1-i0, inliers=nin,
                     inlier_rms=rms, arc_deg=arc)
            break
        if row: out.append(row)
        left += STEP_S
    return pd.DataFrame(out)

# # --- px(t) frequency: f ≈ argmax |FFT{x(t)}|; compare to ω/(2π) ---
# def px_freq(t,x,fmax=30.0):
#     dt=np.median(np.diff(t)); 
#     fs=min(1.0/dt,2000.0); du=1.0/fs
#     tu=np.arange(t[0],t[-1],du); xu=np.interp(tu,t,x)
#     z=(xu-xu.mean())*np.hanning(len(xu))
#     X=np.fft.rfft(z); f=np.fft.rfftfreq(len(z),du); mag=np.abs(X)
#     band=(f>0)&(f<=fmax); 
#     return (f[band][np.argmax(mag[band])] if band.any() else np.nan), f, mag

# --- plotting ---
def plot_omegas(df, filename, output_folder):
    fig,ax=plt.subplots(2,2,figsize=(14,10))
    ax[0,0].plot(df["timestamp"],df["omega_circlefit_rad_s"],label="omega_circlefit")
    ax[0,0].plot(df["timestamp"],df["theta_dot_rad_s"],label="theta_dot (interp)",alpha=.9)
    ax[0,0].set(xlabel="time (s)",ylabel="rad/s",title="Angular velocity"); ax[0,0].grid(True); ax[0,0].legend()
    ax[0,1].hist(df["omega_circlefit_rad_s"],bins=24,edgecolor="k"); 
    m=np.median(df["omega_circlefit_rad_s"]); ax[0,1].axvline(m,ls="--",c="r")
    ax[0,1].set_title(f"omega_circlefit | median={m:.3f}")
    d=df["theta_dot_rad_s"]-df["omega_circlefit_rad_s"]
    ax[1,0].plot(df["timestamp"],d); ax[1,0].axhline(0,color="k",lw=.8); ax[1,0].grid(True)
    ax[1,0].set(xlabel="time (s)",title="theta_dot − omega_circlefit (rad/s)")
    ax[1,1].hist(d,bins=24,edgecolor="k"); ax[1,1].axvline(np.median(d),ls="--",c="r")
    ax[1,1].set_title("difference distribution")
    
    # Create filename based on input file
    base_name = os.path.splitext(os.path.basename(filename))[0]
    plot_path = os.path.join(output_folder, f"{base_name}_omega_analysis.png")
    plt.tight_layout(); plt.savefig(plot_path,dpi=140,bbox_inches="tight"); plt.close(fig)
    return plot_path

# def plot_px(t,x,df):
#     step=max(len(t)//8000,1); tp=t[::step]; xp=x[::step]
#     fpx, f, mag = px_freq(t,x,30.0)
#     f_cf = np.median(df["omega_circlefit_rad_s"])/(2*np.pi)
#     f_td = np.median(df["theta_dot_rad_s"])/(2*np.pi)
#     fig,axs=plt.subplots(2,1,figsize=(12,8),constrained_layout=True)
#     axs[0].plot(tp,xp,lw=.8); axs[0].set(title="x(t)",xlabel="time (s)",ylabel="x (px)"); axs[0].grid(True)
#     axs[1].plot(f,mag,lw=.9); axs[1].set(xlim=(0,30),xlabel="frequency (Hz)",ylabel="|FFT|"); axs[1].grid(True)
#     axs[1].axvline(fpx,ls="--",c="k",label=f"peak x(t): {fpx:.3f} Hz")
#     axs[1].axvline(f_cf,ls="--",c="tab:blue",label=f"median ω_cf/(2π): {f_cf:.3f} Hz")
#     axs[1].axvline(f_td,ls="--",c="tab:orange",label=f"median θ̇/(2π): {f_td:.3f} Hz")
#     axs[1].legend(); plt.savefig(OUTPUT_PXFIG,dpi=140,bbox_inches="tight"); plt.close(fig)

# --- process single file ---
def process_single_file(tracker_file):
    """Process a single tracker file and return results"""
    print(f"Processing {tracker_file}...")
    
    # Load data
    t, x, y = read_positions(tracker_file)
    if len(t) < 100:
        print(f"  Skipping {tracker_file} - insufficient data ({len(t)} points)")
        return None
    
    # Extract circle fit data
    df_cf = extract_series(t, x, y)
    if df_cf.empty:
        print(f"  Skipping {tracker_file} - no valid windows found")
        return None
    
    df_cf = df_cf.sort_values("timestamp").reset_index(drop=True)
    
    # Load theta_dot data
    tt, w = read_theta_dot(tracker_file, assume_deg=False, warmup_s=0.15)
    if len(tt) < 10:
        print(f"  Skipping {tracker_file} - insufficient theta_dot data ({len(tt)} points)")
        return None
    
    w_on = np.interp(df_cf["timestamp"].to_numpy(float), tt, w, left=w[0], right=w[-1])
    
    # Combine data
    df = df_cf.copy()
    df["theta_dot_rad_s"] = w_on
    df["source_file"] = os.path.basename(tracker_file)
    
    return df

# --- main ---
def main():
    # Create output folders
    os.makedirs(OUTPUT_CSV_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_PLOT_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_PX_FOLDER, exist_ok=True)
    
    # Find all CSV files in tracker folder
    tracker_files = glob.glob(os.path.join(TRACKER_FOLDER, "*.csv"))
    if not tracker_files:
        print(f"No CSV files found in {TRACKER_FOLDER}")
        return
    
    print(f"Found {len(tracker_files)} tracker files")
    
    all_results = []
    successful_files = 0
    
    for tracker_file in tracker_files:
        try:
            # Process single file
            df = process_single_file(tracker_file)
            if df is None:
                continue
            
            # Save CSV
            base_name = os.path.splitext(os.path.basename(tracker_file))[0]
            csv_path = os.path.join(OUTPUT_CSV_FOLDER, f"{base_name}_results.csv")
            df.to_csv(csv_path, index=False)
            print(f"  Saved CSV: {csv_path} ({len(df)} rows)")
            
            # Create plot
            plot_path = plot_omegas(df, tracker_file, OUTPUT_PLOT_FOLDER)
            print(f"  Saved plot: {plot_path}")
            
            # Store for summary
            all_results.append(df)
            successful_files += 1
            
        except Exception as e:
            print(f"  Error processing {tracker_file}: {e}")
            continue
    
    # Create summary
    if all_results:
        print(f"\nProcessed {successful_files}/{len(tracker_files)} files successfully")
        
        # Combine all results for summary
        combined_df = pd.concat(all_results, ignore_index=True)
        summary_path = os.path.join(OUTPUT_CSV_FOLDER, "all_trackers_summary.csv")
        combined_df.to_csv(summary_path, index=False)
        print(f"Saved combined summary: {summary_path}")
        
        # Print statistics
        print(f"\nSummary statistics:")
        print(f"  Total windows: {len(combined_df)}")
        print(f"  Median omega (circle fit): {np.median(combined_df['omega_circlefit_rad_s']):.3f} rad/s")
        print(f"  Median omega (theta_dot): {np.median(combined_df['theta_dot_rad_s']):.3f} rad/s")
        print(f"  Files processed: {successful_files}")
        
        # Create overall summary plot
        summary_plot_path = os.path.join(OUTPUT_PLOT_FOLDER, "all_trackers_summary.png")
        plot_omegas(combined_df, "all_trackers", OUTPUT_PLOT_FOLDER)
        print(f"Saved summary plot: {summary_plot_path}")
    
    print("Done.")

if __name__ == "__main__":
    main()
