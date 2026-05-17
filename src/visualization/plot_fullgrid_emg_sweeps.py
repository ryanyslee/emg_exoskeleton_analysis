import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from processing_functions import (get_global_normalization_factors, normalize_data)

def load_fullgrid_sweep_data(sweep_dir: str, base_dir: str, sensor_id: int,
                             trial_files: list[str], no_exo_file: str,
                             no_assi_file: str, label_pattern: str,
                             global_stats: dict, is_normalize: bool = True) -> dict:
    """
    Loads mean EMG data and applies GLOBAL normalization on the fly.
    """
    sweep_data = {"no_exo": None, "no_assi": None, "trials": []}
    mean_cols = [f"mean_{p:02d}%" for p in range(100)]

    # --- Load NoExo Data from the base condition directory ---
    no_exo_path = os.path.join(base_dir, no_exo_file)
    if os.path.exists(no_exo_path):
        df_no_exo = pd.read_csv(no_exo_path)
        sensor_row = df_no_exo[df_no_exo['sensor_id'] == sensor_id]
        if not sensor_row.empty:
            original_data = sensor_row[mean_cols].to_numpy().ravel()
            if is_normalize: # Normalizing data
                sweep_data["no_exo"] = normalize_data(original_data, sensor_id, global_stats)
            else: # using original (un-normalized data)
                sweep_data["no_exo"] = original_data
    else:
        print(f"Warning: NoExo file not found at {no_exo_path}")
    
    # --- Load NoAssi Data ---
    no_assi_path = os.path.join(base_dir, no_assi_file)
    if os.path.exists(no_assi_path):
        df_no_assi = pd.read_csv(no_assi_path)
        sensor_row = df_no_assi[df_no_assi['sensor_id'] == sensor_id]
        if not sensor_row.empty:
            original_data = sensor_row[mean_cols].to_numpy().ravel()
            if is_normalize: # Normalizing data
                sweep_data["no_assi"] = normalize_data(original_data, sensor_id, global_stats)
            else: # using original (un-normalized data)
                sweep_data["no_assi"] = original_data
    else:
        print(f"Warning: NoAssi file not found at {no_assi_path}")

    # --- Load Sweep Trial Data ---
    for trial_file in sorted(trial_files):
        match = re.search(label_pattern, trial_file)
        if not match:
            # Add a check for the specific filename structure if needed
            # print(f"Debug: No label match for {trial_file} with pattern {label_pattern}")
            continue
        label = match.group(1).replace('p', '%').replace('ms', ' ms')

        file_path = os.path.join(sweep_dir, trial_file)
        if os.path.exists(file_path):
            df_trial = pd.read_csv(file_path)
            sensor_row = df_trial[df_trial['sensor_id'] == sensor_id]
            if not sensor_row.empty:
                original_data = sensor_row[mean_cols].to_numpy().ravel()
                if is_normalize: # Normalizing data
                    norm_data = normalize_data(original_data, sensor_id, global_stats)
                    sweep_data["trials"].append({"label": label, "data": norm_data})
                else: # using original (un-normalized data)
                    sweep_data["trials"].append({"label": label, "data": original_data})
        else:
            print(f"Warning: Trial file not found at {file_path}")

    return sweep_data


def plot_single_fullgrid_sweep(ax: plt.Axes, sweep_data: dict, color_map, title: str,
                               show_legend: bool = False, y_label: str | None = None):
    """
    Plots a single EMG sweep (one row of the 5x4 grid).
    """
    if not sweep_data["trials"] and sweep_data["no_exo"] is None:
         ax.text(0.5, 0.5, 'No Data Found', ha='center', va='center', fontsize=12, color='red')
         ax.set_title(title, fontsize=12, fontweight='bold') # Still show title if no data
         return # Stop plotting if no data

    # --- Plot Assistance Trials ---
    num_trials = len(sweep_data["trials"])
    colors = color_map(np.linspace(0.2, 1, num_trials))

    for i, trial in enumerate(sweep_data["trials"]):
        ax.plot(trial["data"], color=colors[i], linewidth=1.5, label=trial["label"])

    # --- Plot NoAssi Condition ---
    if sweep_data["no_assi"] is not None:
        ax.plot(sweep_data["no_assi"], linestyle=':', color="#FF0000", linewidth=1.5, label='NoAssi')
        
    # --- Plot NoExo Condition ---
    if sweep_data["no_exo"] is not None:
        ax.plot(sweep_data["no_exo"], color='#FF8C00', linewidth=1.5, label='NoExo')


    # --- Styling ---
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(direction='out', length=6, width=1, labelsize=10)
    ax.set_xlim(0, 99)
    ax.set_ylim(bottom=0)
    ax.grid(True, linestyle='--', alpha=0.5)

    if y_label:
        ax.set_ylabel(y_label, fontsize=12, fontweight='bold')

    if show_legend:
        ax.legend(
            frameon=False,
            fontsize='medium',
            loc='center left',
            bbox_to_anchor=(1.03, 0.5) # Anchors legend outside the axis
        )

def main():
    """
    Main function to generate the 5x4 plot matrix for EMG sweeps (full grid).
    """
    # --- Configuration ---
    subject_name = "AB01_Jimin"
    walking_condition = "RD"  # Options: "LG", "RA", "RD"
    sweep_type = "Delay"   # Options: "Magnitude", "Delay"
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

    # One peaks_log file for every subject (includes the max and min values for every trial)
    peaks_log_path = os.path.join(processed_dir, "subject_peaks_log.csv")

    muscles = SUBJECT_CONFIG[subject_name]["muscles"]

    magnitudes = ["10p", "15p", "20p", "25p", "30p"]
    delays = ["100ms", "150ms", "200ms", "250ms", "300ms"]

    print(f"Loading global peaks from: {peaks_log_path}")
    global_stats = get_global_normalization_factors(peaks_log_path)
    if not global_stats:
        print("Error: Could not load global stats. Check subject name and log file.")
        return

    # Define parameters based on sweep type
    if sweep_type == "Magnitude":
        fixed_params = delays      # Rows will be fixed delays
        varying_params = magnitudes # Trials within plot will be varying magnitudes
        sweep_subfolder = "Magnitude" # Top-level folder containing fixed delay subfolders
        trial_subfolder_prefix = ""    # No prefix needed for magnitude sweep structure
        trial_file_pattern = "{subj}_{cond}_{mag}{delay}_1.csv"
        label_pattern = r'(\d+p)' # To extract magnitude label
        color_map = plt.get_cmap('Blues')
        figure_title = f'EMG Magnitude Sweeps for {walking_condition} Walking (Normalized)'
        row_label_suffix = " Delay"
    elif sweep_type == "Delay":
        fixed_params = magnitudes  # Rows will be fixed magnitudes
        varying_params = delays    # Trials within plot will be varying delays
        sweep_subfolder = "Delay" # Top-level folder containing fixed magnitude subfolders
        trial_subfolder_prefix = "" # No prefix needed for delay sweep structure
        trial_file_pattern = "{subj}_{cond}_{mag}{delay}_1.csv"
        label_pattern = r'(\d+ms)' # To extract delay label
        color_map = plt.get_cmap('Greens')
        figure_title = f'EMG Delay Sweeps for {walking_condition} Walking (Normalized)'
        row_label_suffix = " Magnitude"
    else:
        raise ValueError("sweep_type must be either 'Magnitude' or 'Delay'")

    # --- Create the 5x4 Subplot Grid ---
    fig, axes = plt.subplots(5, 4, figsize=(22, 10), sharex=True, sharey=False) # 5 rows, 4 columns
    fig.suptitle(figure_title, fontsize=20, fontweight='bold')

    # Define base path for NoExo file (one level up)
    condition_base_dir = os.path.join(processed_dir, walking_condition)
    no_exo_filename = f"{subject_name}_{walking_condition}_NoExo_1.csv"
    no_assi_filename = f"{subject_name}_{walking_condition}_NoAssi_1.csv"

    # --- Loop through rows (fixed parameter) and columns (muscles) ---
    for row_idx, fixed_param in enumerate(fixed_params):
        # Define the specific directory for this row's fixed parameter
        row_sweep_dir = os.path.join(condition_base_dir, sweep_subfolder, fixed_param)

        for col_idx, (muscle_name, sensor_id) in enumerate(muscles.items()):
            is_last_column = (col_idx == len(muscles) - 1)
            ax = axes[row_idx, col_idx]

            # --- Generate list of trial filenames for this specific subplot ---
            trial_files = []
            if sweep_type == "Magnitude":
                # Varying magnitude, fixed delay (fixed_param)
                trial_files = [
                    trial_file_pattern.format(subj=subject_name, cond=walking_condition, mag=mag, delay=fixed_param)
                    for mag in varying_params
                ]
            else: # sweep_type == "Delay"
                # Varying delay, fixed magnitude (fixed_param)
                 trial_files = [
                    trial_file_pattern.format(subj=subject_name, cond=walking_condition, mag=fixed_param, delay=delay)
                    for delay in varying_params
                ]


            # --- Load data for this subplot ---
            sweep_data = load_fullgrid_sweep_data(
                row_sweep_dir, condition_base_dir, sensor_id,
                trial_files, no_exo_filename, no_assi_filename, 
                label_pattern, global_stats, is_normalize=True
            )

            # --- ADDED: Print Max Activations ---
            print(f"\n--- Peaks for {muscle_name} (Fixed Param: {fixed_param}) ---")
            if sweep_data["no_exo"] is not None:
                print(f"NoExo   : {np.max(sweep_data['no_exo']):.4f}")
            if sweep_data["no_assi"] is not None:
                print(f"NoAssi  : {np.max(sweep_data['no_assi']):.4f}")
            for trial in sweep_data["trials"]:
                print(f"{trial['label']:<7} : {np.max(trial['data']):.4f}")
            # --- End of Added Section ---

            # --- Plotting ---
            col_title = muscle_name if row_idx == 0 else "" # Only show muscle name on top row
            row_title = f"{fixed_param}{row_label_suffix}" if col_idx == 0 else None # Only show fixed param on first col

            plot_single_fullgrid_sweep(
                ax, sweep_data, color_map, col_title,
                show_legend=is_last_column, y_label=row_title
            )

            # Set x-label only on the bottom row
            if row_idx == len(fixed_params) - 1:
                ax.set_xlabel("Gait Cycle (%)", fontsize=12)
            else:
                ax.set_xlabel("")


    # --- Final Adjustments ---
    fig.subplots_adjust(
        left=0.08, right=0.90, # Adjust margins for labels and legend
        bottom=0.05, top=0.93,
        wspace=0.3, hspace=0.4 # Increase vertical space slightly
    )
    plt.show()

if __name__ == "__main__":
    main()
