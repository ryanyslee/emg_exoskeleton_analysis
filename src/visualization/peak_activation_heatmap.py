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
    Main function to generate the 2x2 heatmap plot with UNIFIED color scaling.
    """
    # --- Configuration ---
    subject_name = "AB15_Daniel"
    target_walking_condition = "RD"  # The condition you want to PLOT
    # ---------------------
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

    # --- 1. SCAN ALL CONDITIONS FOR GLOBAL MIN/MAX ---
    all_conditions = ["LG", "RA", "RD"]
    all_percent_diffs = []
    target_data = None
    
    print(f"Scanning conditions {all_conditions} to find Global Color Scale...")
    
    for cond in all_conditions:
        print(f"  ... loading {cond}")
        cond_data = load_all_peak_data(processed_dir, subject_name, cond, muscles, magnitudes, delays)
        
        # Save the data if it matches our target plot condition
        if cond == target_walking_condition:
            target_data = cond_data
            
        # Extract all % differences to determine global range
        for muscle_name in muscles:
            peak_no_exo = cond_data[muscle_name].get("NoExo", np.nan)
            if np.isnan(peak_no_exo): continue
            
            # Check NoAssi
            peak_no_assi = cond_data[muscle_name].get("NoAssi", np.nan)
            if not np.isnan(peak_no_assi):
                diff = ((peak_no_assi - peak_no_exo) / peak_no_exo) * 100
                all_percent_diffs.append(diff)
            
            # Check Assistance
            for mag in magnitudes:
                for delay in delays:
                    val = cond_data[muscle_name]["Assistance"][mag][delay]
                    if not np.isnan(val):
                        diff = ((val - peak_no_exo) / peak_no_exo) * 100
                        all_percent_diffs.append(diff)

    # Determine Global Range
    if not all_percent_diffs:
        print("Error: No valid data found to determine color scale.")
        return

    # Use Symmetric Range for Divergent Colormap (RdYlGn_r)
    global_vmax = max(all_percent_diffs)
    global_vmin = min(all_percent_diffs)
    
    print(f"Global Max: {global_vmax:.2f}%")
    print(f"Global Min: {global_vmin:.2f}%")

    # --- 2. PLOT TARGET CONDITION ---
    if target_data is None:
        print(f"Error: Target data for {target_walking_condition} not found.")
        return

    print(f"Generating heatmaps for {target_walking_condition}...")

    fig, axes = plt.subplots(2, 2, figsize=(20, 10))
    fig.suptitle(f"Peak EMG Activation (% Change from NoExo) - {target_walking_condition} Walking (Phased)", 
                 fontsize=20, fontweight='bold')
    
    for ax, (muscle_name, sensor_id) in zip(axes.ravel(), muscles.items()):
        
        peak_no_exo = target_data[muscle_name].get("NoExo", np.nan)
        
        if np.isnan(peak_no_exo):
            print(f"Cannot generate heatmap for {muscle_name}: NoExo baseline data is missing.")
            ax.text(0.5, 0.5, 'NoExo Data Missing', ha='center', va='center', fontsize=12, color='red')
            ax.set_title(muscle_name, fontsize=16, fontweight='bold')
            continue

        peak_no_assi = target_data[muscle_name].get("NoAssi", np.nan)
        no_assi_percent_diff = ((peak_no_assi - peak_no_exo) / peak_no_exo) * 100
        
        # Build grid
        heatmap_data = pd.DataFrame(index=magnitudes, columns=delays)
        for mag in magnitudes:
            for delay in delays:
                peak_assist = target_data[muscle_name]["Assistance"][mag][delay]
                if np.isnan(peak_assist):
                    percent_diff = np.nan
                else:
                    percent_diff = ((peak_assist - peak_no_exo) / peak_no_exo) * 100
                heatmap_data.loc[mag, delay] = percent_diff
        
        heatmap_data = heatmap_data.astype(float)

        # Plot using GLOBAL VMIN/VMAX
        sns.heatmap(
            heatmap_data,
            ax=ax,
            annot=True,
            fmt=".1f",
            cmap='RdYlGn_r',
            vmin=global_vmin,    # <--- UNIFIED MIN
            vmax=global_vmax,    # <--- UNIFIED MAX
            center=0,
            linewidths=.5,
            cbar_kws={'label': '% Change from NoExo'}
        )
        
        ax.set_title(f'{muscle_name} (NoAssi = {round(no_assi_percent_diff, 1)}%)', fontsize=16, fontweight='bold')
        ax.set_ylabel("Assistance Magnitude", fontsize=12)
        ax.set_xlabel("Assistance Delay", fontsize=12)

    plt.tight_layout(rect=[0.0, 0.03, 1, 0.95])
    plt.show()

if __name__ == "__main__":
    main()