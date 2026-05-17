import os
import re
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.interpolate import Rbf

def get_peak_activation(df: pd.DataFrame, sensor_id: int, muscle_name: str) -> float:
    """
    Extracts the peak (max) of the mean activation curve.
    """
    sensor_row = df[df['sensor_id'] == sensor_id]
    if sensor_row.empty:
        return np.nan
    
    mean_cols = [f"mean_{p:02d}%" for p in range(100)]
    try:
        mean_data = sensor_row[mean_cols].to_numpy().ravel()
        if mean_data.size != 100:
            print(f"Warning: Data for sensor {sensor_id} is not 100 points long.")
            return np.nan
        
        phase_data = np.array([])
        
        if muscle_name == "Rectus Femoris":
            # Phase: 60-100%
            phase_data = mean_data[60:100]
        elif muscle_name == "Bicep Femoris":
            # Phase: 85-99% and 0-10% (wrap-around)
            phase_data = np.concatenate((mean_data[85:100], mean_data[0:11]))
        elif muscle_name == "Tibialis Anterior":
            # Phase: 80-99% and 0-10% (wrap-around)
            phase_data = np.concatenate((mean_data[80:100], mean_data[0:11]))
        elif muscle_name == "Gastrocnemius":
            # Phase: 20-60%
            phase_data = mean_data[20:61]  # indices 20 to 60
        else:
            print(f"Warning: Muscle '{muscle_name}' not in defined phases. Using full gait cycle.")
            phase_data = mean_data

        if phase_data.size == 0:
            return np.nan
            
        return np.nanmax(phase_data)

    except KeyError:
        return np.nan

def load_all_peak_data(processed_dir: str, subject_name: str, walking_condition: str, 
                       muscles: dict, magnitudes: list, delays: list) -> dict:
    """
    Loads all peak activation data for all trials into a nested dictionary.
    """
    peak_data = {muscle: {} for muscle in muscles}
    condition_base_dir = os.path.join(processed_dir, walking_condition)
    
    for muscle_name, sensor_id in muscles.items():
        # --- 1. Load NoExo Peak (Baseline) ---
        no_exo_filename = f"{subject_name}_{walking_condition}_NoExo_1.csv"
        no_exo_path = os.path.join(condition_base_dir, no_exo_filename)
        if os.path.exists(no_exo_path):
            df_no_exo = pd.read_csv(no_exo_path)
            peak_data[muscle_name]["NoExo"] = get_peak_activation(df_no_exo, sensor_id, muscle_name)
        else:
            peak_data[muscle_name]["NoExo"] = np.nan
            
        # --- 2. Load NoAssi Peak ---
        no_assi_filename = f"{subject_name}_{walking_condition}_NoAssi_1.csv"
        no_assi_path = os.path.join(condition_base_dir, no_assi_filename)
        if os.path.exists(no_assi_path):
            df_no_assi = pd.read_csv(no_assi_path)
            peak_data[muscle_name]["NoAssi"] = get_peak_activation(df_no_assi, sensor_id, muscle_name)
        else:
            peak_data[muscle_name]["NoAssi"] = np.nan

        # --- 3. Load all Assistance Trial Peaks ---
        peak_data[muscle_name]["Assistance"] = {}
        for mag in magnitudes:
            peak_data[muscle_name]["Assistance"][mag] = {}
            data_folder = os.path.join(condition_base_dir, "Delay", mag)
            for delay in delays:
                filename = f"{subject_name}_{walking_condition}_{mag}{delay}_1.csv"
                file_path = os.path.join(data_folder, filename)
                if os.path.exists(file_path):
                    df_trial = pd.read_csv(file_path)
                    peak_trial = get_peak_activation(df_trial, sensor_id, muscle_name)
                    peak_data[muscle_name]["Assistance"][mag][delay] = peak_trial
                else:
                    peak_data[muscle_name]["Assistance"][mag][delay] = np.nan
                    
    return peak_data

def get_interpolated_surface(muscle_name, muscle_data, magnitudes, delays):
    """
    Helper to generate the RBF surface for a muscle, used for both finding global max
    and plotting.
    """
    peak_no_exo = muscle_data.get("NoExo", np.nan)
    if np.isnan(peak_no_exo):
        return None, None, None

    # --- 1. Flatten Data for RBF ---
    pts_x = [] # Delay (seconds)
    pts_y = [] # Magnitude (percent/100)
    pts_z = [] # % Change from NoExo

    # A. Add NoAssi Point (0 magnitude, 0 delay)
    peak_no_assi = muscle_data.get("NoAssi", np.nan)
    if not np.isnan(peak_no_assi):
        val = ((peak_no_assi - peak_no_exo) / peak_no_exo) * 100
        pts_x.append(0.0)
        pts_y.append(0.0)
        pts_z.append(val)
    
    # B. Add Assistance Points
    for mag_str in magnitudes:
        for delay_str in delays:
            peak_val = muscle_data["Assistance"][mag_str][delay_str]
            if not np.isnan(peak_val):
                mag_val = int(mag_str.replace('p', '')) / 100.0
                delay_val = int(delay_str.replace('ms', '')) / 1000.0
                val = ((peak_val - peak_no_exo) / peak_no_exo) * 100
                pts_x.append(delay_val)
                pts_y.append(mag_val)
                pts_z.append(val)

    if len(pts_z) < 4:
        return None, None, None

    pts_x = np.array(pts_x)
    pts_y = np.array(pts_y)
    pts_z = np.array(pts_z)

    # --- 2. RBF Interpolation ---
    # Create grid strictly 0 to 0.31 to match plot bounds
    nx, ny = 100, 100
    xi = np.linspace(0, 0.31, nx)
    yi = np.linspace(0, 0.31, ny)
    XI, YI = np.meshgrid(xi, yi)

    try:
        rbf = Rbf(pts_x, pts_y, pts_z, function='multiquadric', smooth=0.1)
        ZI = rbf(XI, YI)
        return XI, YI, ZI
    except Exception as e:
        # print(f"RBF Error for {muscle_name}: {e}")
        return None, None, None

def plot_muscle_contour(ax, muscle_name, muscle_data, magnitudes, delays, vmin, vmax):
    """
    Generates an RBF interpolated contour plot for a single muscle on the provided axes.
    Uses global vmin and vmax for consistent coloring across conditions.
    """
    
    XI, YI, ZI = get_interpolated_surface(muscle_name, muscle_data, magnitudes, delays)
    
    if ZI is None:
        ax.text(0.5, 0.5, 'Insufficient Data', ha='center', va='center')
        ax.set_title(muscle_name)
        return None

    # --- 3. Plotting with Global Scale ---
    levels = np.linspace(vmin, vmax, 25)
    
    # Filled contour strictly within 0-0.3 box
    cp = ax.contourf(XI, YI, ZI, levels=levels, cmap='Reds', alpha=0.9, vmin=vmin, vmax=vmax)
    ax.contour(XI, YI, ZI, levels=levels, colors='k', linewidths=0.5, alpha=0.3)

    # --- 4. Special Points ---
    
    # --- 1. Flatten Data for RBF ---
    pts_x = [] # Delay (seconds)
    pts_y = [] # Magnitude (percent/100)
    
    peak_no_exo = muscle_data.get("NoExo", np.nan)
    peak_no_assi = muscle_data.get("NoAssi", np.nan)
    if not np.isnan(peak_no_assi) and not np.isnan(peak_no_exo):
        pts_x.append(0.0)
        pts_y.append(0.0)
    
    # B. Add Assistance Points
    for mag_str in magnitudes:
        for delay_str in delays:
            peak_val = muscle_data["Assistance"][mag_str][delay_str]
            if not np.isnan(peak_val) and not np.isnan(peak_no_exo):
                mag_val = int(mag_str.replace('p', '')) / 100.0
                delay_val = int(delay_str.replace('ms', '')) / 1000.0
                pts_x.append(delay_val)
                pts_y.append(mag_val)
    
    # A. Raw Data Points (Hollow circles)
    ax.scatter(pts_x, pts_y, facecolors='none', edgecolors='black', linewidths=1.0, 
               s=40, zorder=5)
    
    # B. Theoretical Optimal (Min value)
    min_idx = np.unravel_index(np.argmin(ZI), ZI.shape)
    opt_x, opt_y = XI[min_idx], YI[min_idx]
    opt_val = ZI[min_idx]
    
    ax.scatter(opt_x, opt_y, facecolors='gold', edgecolors='black', s=50, zorder=10, 
               label=f'Theoretical Optimal: {opt_val:.1f}%')
    
    # C. Theoretical Worst (Max value)
    max_idx = np.unravel_index(np.argmax(ZI), ZI.shape)
    worst_x, worst_y = XI[max_idx], YI[max_idx]
    worst_val = ZI[max_idx]
    
    ax.scatter(worst_x, worst_y, facecolors='blue', edgecolors='black', s=50, zorder=10, 
               label=f'Theoretical Worst: {worst_val:.1f}%')

    # D. Previously Proposed Optimal (Fixed at 125ms, 20%)
    ax.scatter(0.125, 0.20, facecolors='black', edgecolors='black', s=50, zorder=10, 
               label='Previously Proposed Optimal')

    # --- 5. Formatting ---
    ax.set_title(f"{muscle_name}", fontsize=14, fontweight='bold')
    ax.set_xlabel("Assistance Delay (ms)")
    ax.set_ylabel("Assistance Magnitude (%)")
    
    # Ticks
    x_ticks = sorted(list(set([0.0, 0.1, 0.15, 0.2, 0.25, 0.3])))
    x_labels = ['NoAssi', '100ms', '150ms', '200ms', '250ms', '300ms']
    y_ticks = sorted(list(set([0.0, 0.1, 0.15, 0.2, 0.25, 0.3])))
    y_labels = ['NoAssi', '10%', '15%', '20%', '25%', '30%']
    
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)

    # Limits & Padding (Gap to top/right/left/bottom)
    # Adding negative margin so 0.0 is distinct from the frame
    ax.set_xlim(-0.02, 0.33)
    ax.set_ylim(-0.02, 0.33)
    
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # Legend - 'best' location
    ax.legend(loc='best', fontsize=8, framealpha=0.9, fancybox=True)
    
    return cp

def main():
    # --- Configuration ---
    subject_name = "AB16_Ilseung"
    
    # Select which condition to PLOT, but we will scan ALL for global color scaling
    target_walking_condition = "RD" 
    
    parent_dir = "N=8 Processed Data"
    
    # Subject Config
    SUBJECT_CONFIG = {
        "AB01_Jimin": {
            "folder": "AB01_Jimin_processed",
            "muscles": {
                "Tibialis Anterior": 82719, "Bicep Femoris": 83389,
                "Gastrocnemius": 82653, "Rectus Femoris": 83449
            }
        },
        "AB02_Rajiv": {
            "folder": "AB02_Rajiv_processed",
            "muscles": {
                "Tibialis Anterior": 83410, "Bicep Femoris": 83389,
                "Gastrocnemius": 82755, "Rectus Femoris": 83449
            }
        },
        "AB03_Amy": {
            "folder": "AB03_Amy_processed",
            "muscles": {
                "Tibialis Anterior": 82522, "Bicep Femoris": 83492,
                "Gastrocnemius": 83410, "Rectus Femoris": 82511
            }
        },
        "AB05_Maria": {
            "folder": "AB05_Maria_processed",
            "muscles": {
                "Tibialis Anterior": 82755, "Bicep Femoris": 83389,
                "Gastrocnemius": 83410, "Rectus Femoris": 83426
            }
        },
        "AB08_Adrian": {
            "folder": "AB08_Adrian_processed",
            "muscles": {
                "Tibialis Anterior": 82724, "Bicep Femoris": 83492,
                "Gastrocnemius": 82522, "Rectus Femoris": 82511
            }
        },
        "AB11_Ryan": {
            "folder": "AB11_Ryan_processed",
            "muscles": {
                "Tibialis Anterior": 82719, "Bicep Femoris": 83389,
                "Gastrocnemius": 82653, "Rectus Femoris": 83449
            }
        },
        "AB15_Daniel": {
            "folder": "AB15_Daniel_processed",
            "muscles": {
                "Tibialis Anterior": 82724, "Bicep Femoris": 83492,
                "Gastrocnemius": 82522, "Rectus Femoris": 82511
            }
        },
        "AB16_Ilseung": {
            "folder": "AB16_Ilseung_processed",
            "muscles": {
                "Tibialis Anterior": 82719, "Bicep Femoris": 83449,
                "Gastrocnemius": 82653, "Rectus Femoris": 83426
            }
        }
    }
    
    processed_dir = os.path.join(parent_dir, SUBJECT_CONFIG[subject_name]["folder"])
    
    muscles = SUBJECT_CONFIG[subject_name]["muscles"]
    
    magnitudes = ["30p", "25p", "20p", "15p", "10p"]
    delays = ["100ms", "150ms", "200ms", "250ms", "300ms"]

    # --- 1. ITERATE THROUGH ALL CONDITIONS TO FIND GLOBAL MAX/MIN ---
    all_conditions = ["LG", "RA", "RD"]
    all_percent_values = []
    
    # We will store the data for our target condition so we don't have to reload it
    target_condition_data = None
    
    print(f"Scanning all walking conditions {all_conditions} to determine Global Max/Min...")
    
    for condition in all_conditions:
        print(f"  Loading data for: {condition}...")
        
        # Load data for this condition
        cond_data = load_all_peak_data(processed_dir, subject_name, condition,
                                       muscles, magnitudes, delays)
        
        # If this is the one we want to plot later, save it
        if condition == target_walking_condition:
            target_condition_data = cond_data
            
        # Extract percent changes for Global Stats
        for muscle_name, muscle_data in cond_data.items():
            peak_no_exo = muscle_data.get("NoExo", np.nan)
            if np.isnan(peak_no_exo): continue

            # Check NoAssi
            peak_no_assi = muscle_data.get("NoAssi", np.nan)
            if not np.isnan(peak_no_assi):
                val = ((peak_no_assi - peak_no_exo) / peak_no_exo) * 100
                all_percent_values.append(val)
                
            # Check all Assistance conditions
            for mag in magnitudes:
                for delay in delays:
                    val = muscle_data["Assistance"][mag][delay]
                    if not np.isnan(val):
                        pct_diff = ((val - peak_no_exo) / peak_no_exo) * 100
                        all_percent_values.append(pct_diff)
            
            _, _, ZI = get_interpolated_surface(muscle_name, muscle_data, magnitudes, delays)
            if ZI is not None:
                all_percent_values.extend(ZI.flatten())
    
    # Determine Global Range
    if not all_percent_values:
        print("Error: No valid data found across any condition.")
        return
    
    global_vmin = min(all_percent_values)
    global_vmax = max(all_percent_values)
    
    print(f"Global Range determined across {all_conditions}: +/- {global_vmax:.2f}%")

    # --- 2. PLOT TARGET CONDITION USING GLOBAL RANGE ---
    if target_condition_data is None:
        print(f"Error: Could not load data for target condition {target_walking_condition}")
        return

    print(f"\nPlotting for target condition: {target_walking_condition}...")
    
    # Print table header
    print("\n" + "="*70)
    print(f"OPTIMAL AND WORST PARAMETERS ({subject_name} - {target_walking_condition})")
    print("="*70)
    print(f"{'Muscle':<20} | {'Optimal (Delay, Mag, %Diff)':<35} | {'Worst (Delay, Mag, %Diff)':<35}")
    print("-" * 95)

    fig, axes = plt.subplots(2, 2, figsize=(18, 11))
    fig.suptitle(f"Peak EMG Activation Contour Map - {target_walking_condition} Walking ({subject_name})", 
                 fontsize=20, fontweight='bold')
    
    axes_flat = axes.ravel()
    
    for ax, muscle_name in zip(axes_flat, muscles.keys()):
        muscle_data = target_condition_data[muscle_name]
        
        # Calculate stats for table printing
        XI, YI, ZI = get_interpolated_surface(muscle_name, muscle_data, magnitudes, delays)
        if ZI is not None:
            min_idx = np.unravel_index(np.argmin(ZI), ZI.shape)
            max_idx = np.unravel_index(np.argmax(ZI), ZI.shape)
            
            opt_delay = XI[min_idx] * 1000 # Convert s back to ms
            opt_mag = YI[min_idx] * 100 # Convert fraction back to %
            opt_val = ZI[min_idx]
            
            worst_delay = XI[max_idx] * 1000
            worst_mag = YI[max_idx] * 100
            worst_val = ZI[max_idx]
            
            # Format strings for the table
            opt_str = f"{opt_delay:.0f}ms, {opt_mag:.1f}%, {opt_val:.1f}%"
            worst_str = f"{worst_delay:.0f}ms, {worst_mag:.1f}%, {worst_val:.1f}%"
            print(f"{muscle_name:<20} | {opt_str:<35} | {worst_str:<35}")
        else:
            print(f"{muscle_name:<20} | {'Insufficient Data':<35} | {'Insufficient Data':<35}")

        # PASS GLOBAL MIN/MAX
        cp = plot_muscle_contour(ax, muscle_name, muscle_data, magnitudes, delays, 
                                 vmin=global_vmin, vmax=global_vmax)
        if cp:
            cbar = plt.colorbar(cp, ax=ax)
            cbar.set_label('% Change from NoExo')

    plt.tight_layout(rect=[0.0, 0.03, 1, 0.95])
    print("="*70 + "\n")
    
    # output_filename = f"{subject_name}_{target_walking_condition}_EMG_Contours_Unified.png"
    # plt.savefig(output_filename, dpi=300)
    # print(f"Plot saved to {output_filename}")
    plt.show()

if __name__ == "__main__":
    main()