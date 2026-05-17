import os
import re
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

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
            print(f"Warning: Muscle '{muscle_name}' not in defined phases. Using full gait cycle (0-99%).")
            phase_data = mean_data

        if phase_data.size == 0:
            print(f"Warning: No data selected for phase for muscle '{muscle_name}'.")
            return np.nan
            
        return np.nanmax(phase_data)

    except KeyError:
        print(f"Error: Columns 'mean_00%'...'mean_99%' not found.")
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
            # print(f"Warning: NoExo file not found at {no_exo_path}")
            peak_data[muscle_name]["NoExo"] = np.nan
            
        # --- 2. Load NoAssi Peak ---
        no_assi_filename = f"{subject_name}_{walking_condition}_NoAssi_1.csv"
        no_assi_path = os.path.join(condition_base_dir, no_assi_filename)
        if os.path.exists(no_assi_path):
            df_no_assi = pd.read_csv(no_assi_path)
            peak_data[muscle_name]["NoAssi"] = get_peak_activation(df_no_assi, sensor_id, muscle_name)
        else:
            # print(f"Warning: NoAssi file not found at {no_assi_path}")
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
                    # print(f"Warning: Trial file not found at {file_path}")
                    peak_data[muscle_name]["Assistance"][mag][delay] = np.nan
                    
    return peak_data

def main():
    """
    Main function to generate the 2x2 AVERAGE heatmap plot with UNIFIED COLOR BARS.
    """
    # --- Configuration ---
    PARENT_DIR = "N=8 Processed Data"
    WALKING_CONDITION_TO_PLOT = "RD"

    # --- MUST BE CONFIGURED MANUALLY ---
    # Add all subjects and their unique info here
    SUBJECT_CONFIG = {
        # "AB01_Jimin": {
        #     "folder": "AB01_Jimin_processed",
        #     "muscles": {
        #         "Tibialis Anterior": 82719, "Bicep Femoris": 83389,
        #         "Gastrocnemius": 82653, "Rectus Femoris": 83449
        #     }
        # },
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
        # "AB11_Ryan": {
        #     "folder": "AB11_Ryan_processed",
        #     "muscles": {
        #         "Tibialis Anterior": 82719, "Bicep Femoris": 83389,
        #         "Gastrocnemius": 82653, "Rectus Femoris": 83449
        #     }
        # },
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
    
    # Standard lists
    MUSCLE_NAMES = ["Tibialis Anterior", "Bicep Femoris", "Gastrocnemius", "Rectus Femoris"]
    magnitudes = ["30p", "25p", "20p", "15p", "10p"]
    delays = ["100ms", "150ms", "200ms", "250ms", "300ms"]

    # --- Pre-Calculation: SCAN ALL CONDITIONS FOR GLOBAL MIN/MAX ---
    ALL_CONDITIONS = ["LG", "RA", "RD"]
    all_values_for_scale = []
    
    # We also need to store the data for the specific condition we want to plot later
    target_condition_avg_data = None
    target_condition_no_assi = None

    print(f"Scanning conditions {ALL_CONDITIONS} to find Global Color Scale for AVERAGES...")

    for condition in ALL_CONDITIONS:
        print(f"  ... processing condition: {condition}")
        
        # Temp storage for this condition
        cond_subject_diffs = {m: {ma: {d: [] for d in delays} for ma in magnitudes} for m in MUSCLE_NAMES}
        cond_no_assi_diffs = {m: [] for m in MUSCLE_NAMES}

        # 1. Load data for all subjects for this condition
        for subject_name, config in SUBJECT_CONFIG.items():
            processed_dir = os.path.join(PARENT_DIR, config["folder"])
            subject_muscles = config["muscles"]
            
            peak_data = load_all_peak_data(processed_dir, subject_name, condition,
                                           subject_muscles, magnitudes, delays)
            
            for muscle_name in MUSCLE_NAMES:
                peak_no_exo = peak_data[muscle_name].get("NoExo", np.nan)
                if np.isnan(peak_no_exo): continue
                
                # NoAssi
                peak_no_assi = peak_data[muscle_name].get("NoAssi", np.nan)
                if not np.isnan(peak_no_assi):
                    val = ((peak_no_assi - peak_no_exo) / peak_no_exo) * 100
                    cond_no_assi_diffs[muscle_name].append(val)
                    
                # Assistance
                for mag in magnitudes:
                    for delay in delays:
                        peak_assist = peak_data[muscle_name]["Assistance"][mag][delay]
                        if not np.isnan(peak_assist):
                            val = ((peak_assist - peak_no_exo) / peak_no_exo) * 100
                            cond_subject_diffs[muscle_name][mag][delay].append(val)

        # 2. Calculate averages for this condition and collect for global scale
        # Also store if it's the target condition
        current_cond_avgs = {}
        
        for muscle_name in MUSCLE_NAMES:
            # Avg NoAssi
            if cond_no_assi_diffs[muscle_name]:
                avg_no_assi = np.nanmean(cond_no_assi_diffs[muscle_name])
                # We generally don't include NoAssi in the heatmap range calculation unless explicitly desired,
                # but let's include it to ensure the title value isn't wildly off-scale if we were plotting it.
                # Actually, heatmap only shows the grid. The title shows NoAssi. 
                # Let's focus scaling on the grid values.
            
            # Avg Grid
            grid_df = pd.DataFrame(index=magnitudes, columns=delays, dtype=float)
            for mag in magnitudes:
                for delay in delays:
                    vals = cond_subject_diffs[muscle_name][mag][delay]
                    if vals:
                        avg_val = np.nanmean(vals)
                        grid_df.loc[mag, delay] = avg_val
                        all_values_for_scale.append(avg_val)
                    else:
                        grid_df.loc[mag, delay] = np.nan
            
            current_cond_avgs[muscle_name] = grid_df

        # Save if this is the target condition
        if condition == WALKING_CONDITION_TO_PLOT:
            target_condition_avg_data = current_cond_avgs
            target_condition_no_assi = cond_no_assi_diffs

    global_vmin = min(all_values_for_scale)
    global_vmax = max(all_values_for_scale)
    
    print(f"Global Max: {global_vmax:.2f}%")
    print(f"Global Min: {global_vmin:.2f}%")

    if target_condition_avg_data is None:
        print(f"Error: Target condition {WALKING_CONDITION_TO_PLOT} was not processed successfully.")
        return

    print(f"Generating heatmaps for {WALKING_CONDITION_TO_PLOT}...")

    # --- Loop 2: Plot the heatmaps using the Global Scale ---
    fig, axes = plt.subplots(2, 2, figsize=(18, 11))
    fig.suptitle(f"Average Peak EMG (% Change from NoExo) - {WALKING_CONDITION_TO_PLOT} (N={len(SUBJECT_CONFIG)}, Phased)", 
                 fontsize=20, fontweight='bold')
    
    for ax, muscle_name in zip(axes.ravel(), MUSCLE_NAMES):
        
        # Calculate average NoAssi diff for the title (specific to the target condition)
        no_assi_vals = target_condition_no_assi[muscle_name]
        if no_assi_vals:
            avg_no_assi_diff = np.nanmean(no_assi_vals)
            plot_title = f"{muscle_name} (NoAssi: {avg_no_assi_diff:+.1f}%)"
        else:
            plot_title = f"{muscle_name} (NoAssi: N/A)"
            
        ax.set_title(plot_title, fontsize=16, fontweight='bold')

        # Retrieve the pre-calculated dataframe for the target condition
        avg_heatmap_data = target_condition_avg_data[muscle_name]
        
        # --- Plot the Heatmap ---
        sns.heatmap(
            avg_heatmap_data,
            ax=ax,
            annot=True,
            fmt=".1f",
            cmap='RdYlGn_r',
            vmin=global_vmin,    # <--- Unified Min across all conditions
            vmax=global_vmax,    # <--- Unified Max across all conditions
            center=0,
            linewidths=.5,
            cbar_kws={'label': f'% Change from NoExo (N={len(SUBJECT_CONFIG)})'}
        )
        
        ax.set_ylabel("Assistance Magnitude", fontsize=12)
        ax.set_xlabel("Assistance Delay", fontsize=12)

    plt.tight_layout(rect=[0.0, 0.03, 1, 0.95])
    plt.show()

if __name__ == "__main__":
    main()