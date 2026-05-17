import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import Rbf

def get_peak_activation(df: pd.DataFrame, sensor_id: int, muscle_name: str) -> float:
    """
    Extracts the peak (max) of the mean activation curve for a given sensor_id
    from a muscle-specific phase of the gait cycle.
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

def load_single_subject_peaks(processed_dir: str, subject_name: str, walking_condition: str, 
                              muscles: dict, magnitudes: list, delays: list) -> dict:
    """
    Loads raw peak activation data for a SINGLE subject.
    Returns nested dict: data[muscle][condition_key] = peak_value
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

def calculate_average_condition_data(parent_dir: str, subject_config: dict, walking_condition: str,
                                     muscle_names: list, magnitudes: list, delays: list):
    """
    Loads data for ALL subjects for a specific walking condition, calculates % change for each,
    and computes the AVERAGE % change across subjects.
    
    Returns: averaged_data dict structure similar to single subject but values are means.
    """
    # Initialize storage for lists of percent differences
    # Structure: aggregated[muscle]['NoAssi'] = [val1, val2...]
    # Structure: aggregated[muscle]['Assistance'][mag][delay] = [val1, val2...]
    aggregated = {m: {'NoAssi': [], 'Assistance': {mag: {d: [] for d in delays} for mag in magnitudes}} for m in muscle_names}

    # 1. Collect data from all subjects
    for subject_name, config in subject_config.items():
        processed_dir = os.path.join(parent_dir, config["folder"])
        subject_muscles = config["muscles"]
        
        # Load raw peaks
        subj_data = load_single_subject_peaks(processed_dir, subject_name, walking_condition,
                                              subject_muscles, magnitudes, delays)
        
        for muscle in muscle_names:
            peak_no_exo = subj_data[muscle].get("NoExo", np.nan)
            
            if np.isnan(peak_no_exo):
                continue # Cannot calculate % diff without baseline
            
            # NoAssi % Diff
            peak_no_assi = subj_data[muscle].get("NoAssi", np.nan)
            if not np.isnan(peak_no_assi):
                diff = ((peak_no_assi - peak_no_exo) / peak_no_exo) * 100
                aggregated[muscle]['NoAssi'].append(diff)
            
            # Assistance % Diffs
            for mag in magnitudes:
                for delay in delays:
                    peak_assist = subj_data[muscle]['Assistance'][mag][delay]
                    if not np.isnan(peak_assist):
                        diff = ((peak_assist - peak_no_exo) / peak_no_exo) * 100
                        aggregated[muscle]['Assistance'][mag][delay].append(diff)

    # 2. Compute Averages
    averaged_data = {m: {'NoAssi': np.nan, 'Assistance': {mag: {d: np.nan for d in delays} for mag in magnitudes}} for m in muscle_names}
    
    for muscle in muscle_names:
        # Average NoAssi
        if aggregated[muscle]['NoAssi']:
            averaged_data[muscle]['NoAssi'] = np.nanmean(aggregated[muscle]['NoAssi'])
        
        # Average Assistance
        for mag in magnitudes:
            for delay in delays:
                vals = aggregated[muscle]['Assistance'][mag][delay]
                if vals:
                    averaged_data[muscle]['Assistance'][mag][delay] = np.nanmean(vals)
                    
    return averaged_data

def get_interpolated_surface(avg_data, magnitudes, delays):
    """
    Helper to generate the RBF surface for a muscle, used for both finding global max
    and plotting.
    """
    # --- 1. Flatten Data for RBF ---
    pts_x = [] # Delay (seconds)
    pts_y = [] # Magnitude (percent/100)
    pts_z = [] # % Change from NoExo

    # A. Add NoAssi Point (0 magnitude, 0 delay)
    val_no_assist = avg_data.get("NoAssi", np.nan)
    if not np.isnan(val_no_assist):
        pts_x.append(0.0)
        pts_y.append(0.0)
        pts_z.append(val_no_assist)
    
    # B. Add Assistance Points
    for mag_str in magnitudes:
        for delay_str in delays:
            val = avg_data["Assistance"][mag_str][delay_str]
            if not np.isnan(val):
                mag_val = int(mag_str.replace('p', '')) / 100.0
                delay_val = int(delay_str.replace('ms', '')) / 1000.0
                pts_x.append(delay_val)
                pts_y.append(mag_val)
                pts_z.append(val)

    if len(pts_z) < 4:
        return None, None, None

    pts_x = np.array(pts_x)
    pts_y = np.array(pts_y)
    pts_z = np.array(pts_z)

    # --- 2. RBF Interpolation ---
    # Create grid strictly 0 to 0.3 to match data bounds
    nx, ny = 100, 100
    xi = np.linspace(0, 0.31, nx)
    yi = np.linspace(0, 0.31, ny)
    XI, YI = np.meshgrid(xi, yi)

    try:
        rbf = Rbf(pts_x, pts_y, pts_z, function='multiquadric', smooth=0.1)
        ZI = rbf(XI, YI)
        return XI, YI, ZI
    except Exception as e:
        return None, None, None

def plot_average_contour(ax, muscle_name, avg_data, magnitudes, delays, vmin, vmax):
    """
    Generates an RBF interpolated contour plot for a single muscle on the provided axes.
    Uses global vmin and vmax for consistent coloring across conditions.
    """
    
    XI, YI, ZI = get_interpolated_surface(avg_data, magnitudes, delays)
    
    if ZI is None:
        ax.text(0.5, 0.5, 'Insufficient Data', ha='center', va='center')
        ax.set_title(muscle_name)
        return None

    # --- 3. Plotting with Global Scale ---
    # Use global vmin/vmax to create shared levels
    levels = np.linspace(vmin, vmax, 25)
    
    # Filled contour strictly within 0-0.3 box
    # vmin and vmax passed here force the colormap to use the global scale
    cp = ax.contourf(XI, YI, ZI, levels=levels, cmap='Reds', alpha=0.9, vmin=vmin, vmax=vmax)
    ax.contour(XI, YI, ZI, levels=levels, colors='k', linewidths=0.5, alpha=0.3)

    # --- 4. Special Points ---
    
    # --- 1. Flatten Data for RBF ---
    pts_x = [] # Delay (seconds)
    pts_y = [] # Magnitude (percent/100)
    
    val_no_assi = avg_data.get("NoAssi", np.nan)
    if not np.isnan(val_no_assi):
        pts_x.append(0.0)
        pts_y.append(0.0)
    
    # B. Add Assistance Points
    for mag_str in magnitudes:
        for delay_str in delays:
            val = avg_data["Assistance"][mag_str][delay_str]
            if not np.isnan(val):
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
    PARENT_DIR = "N=8 Processed Data"
    
    # Select which condition to PLOT (Average Map)
    TARGET_WALKING_CONDITION = "RD" 

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
    
    MUSCLE_NAMES = ["Tibialis Anterior", "Bicep Femoris", "Gastrocnemius", "Rectus Femoris"]
    magnitudes = ["30p", "25p", "20p", "15p", "10p"]
    delays = ["100ms", "150ms", "200ms", "250ms", "300ms"]

    # --- 1. ITERATE THROUGH ALL CONDITIONS TO FIND GLOBAL MAX/MIN OF AVERAGES ---
    all_conditions = ["LG", "RA", "RD"]
    all_avg_percent_values = []
    
    # Store the averaged data for the target condition
    target_averaged_data = None
    
    print(f"Calculating averages across N={len(SUBJECT_CONFIG)} subjects for all conditions to determine Global Scale...")
    
    for condition in all_conditions:
        print(f"  Processing condition: {condition}...")
        
        # Calculate averages for this condition (already in percentage)
        cond_data = calculate_average_condition_data(PARENT_DIR, SUBJECT_CONFIG, condition,
                                                    MUSCLE_NAMES, magnitudes, delays)
        
        if condition == TARGET_WALKING_CONDITION:
            target_averaged_data = cond_data
            
        # Collect values for global range
        for muscle in MUSCLE_NAMES:
            # Check NoAssi average
            val_no_assi = cond_data[muscle].get('NoAssi', np.nan)
            if not np.isnan(val_no_assi):
                all_avg_percent_values.append(val_no_assi)
            
            # Check Assistance averages
            for mag in magnitudes:
                for delay in delays:
                    val = cond_data[muscle]['Assistance'][mag][delay]
                    if not np.isnan(val):
                        all_avg_percent_values.append(val)
            
            _, _, ZI = get_interpolated_surface(cond_data[muscle], magnitudes, delays)
            if ZI is not None:
                all_avg_percent_values.extend(ZI.flatten())

    # Determine Global Range from AVERAGES
    if not all_avg_percent_values:
        print("Error: No valid data found.")
        return
        
    global_min = min(all_avg_percent_values)
    global_max = max(all_avg_percent_values)
    
    print(f"Global Average Range determined: Min={global_min:.2f}%, Max={global_max:.2f}%")

    # --- 2. PLOT TARGET CONDITION USING GLOBAL RANGE ---
    if target_averaged_data is None:
        print(f"Error: Target condition {TARGET_WALKING_CONDITION} data missing.")
        return

    print(f"Generating Average Contour Plot for: {TARGET_WALKING_CONDITION}...")

    fig, axes = plt.subplots(2, 2, figsize=(18, 11))
    fig.suptitle(f"Average Peak EMG Contour Map - {TARGET_WALKING_CONDITION} Walking (N={len(SUBJECT_CONFIG)})", 
                 fontsize=20, fontweight='bold')
    
    axes_flat = axes.ravel()
    
    for ax, muscle_name in zip(axes_flat, MUSCLE_NAMES):
        muscle_data = target_averaged_data[muscle_name]
        
        # PASS GLOBAL MIN/MAX
        cp = plot_average_contour(ax, muscle_name, muscle_data, magnitudes, delays, 
                                 vmin=global_min, vmax=global_max)
        if cp:
            cbar = plt.colorbar(cp, ax=ax)
            cbar.set_label('% Change from NoExo (Average)')

    plt.tight_layout(rect=[0.0, 0.03, 1, 0.95])
    
    output_filename = f"Average_N{len(SUBJECT_CONFIG)}_{TARGET_WALKING_CONDITION}_EMG_Contours.png"
    plt.savefig(output_filename, dpi=300)
    print(f"Plot saved to {output_filename}")
    plt.show()

if __name__ == "__main__":
    main()