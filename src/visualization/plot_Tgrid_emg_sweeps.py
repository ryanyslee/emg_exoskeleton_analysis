import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def load_sweep_data(sweep_dir: str, sensor_id: int,
                    trial_files: list[str], no_assi_file: str,
                    no_exo_file: str, label_pattern: str) -> dict:
    """
    Loads mean EMG data for NoExo, NoAssi, and assistance trials.
    """
    sweep_data = {"no_assi": None, "no_exo": None, "trials": []}
    mean_cols = [f"mean_{p:02d}%" for p in range(100)]

    # --- Load NoAssi from the sweep subfolder ---
    no_assi_path = os.path.join(sweep_dir, no_assi_file)  # <— changed from base_dir to sweep_dir
    if os.path.exists(no_assi_path):
        df_no_assi = pd.read_csv(no_assi_path)
        sensor_row = df_no_assi[df_no_assi['sensor_id'] == sensor_id]
        if not sensor_row.empty:
            sweep_data["no_assi"] = sensor_row[mean_cols].to_numpy().ravel()

    # --- Load NoExo from the sweep subfolder ---
    no_exo_path = os.path.join(sweep_dir, no_exo_file)    # <— changed from base_dir to sweep_dir
    if os.path.exists(no_exo_path):
        df_no_exo = pd.read_csv(no_exo_path)
        sensor_row = df_no_exo[df_no_exo['sensor_id'] == sensor_id]
        if not sensor_row.empty:
            sweep_data["no_exo"] = sensor_row[mean_cols].to_numpy().ravel()

    # --- Load Sweep Trial Data from the same subfolder ---
    for trial_file in sorted(trial_files):
        match = re.search(label_pattern, trial_file)
        if not match:
            continue
        label = match.group(1).replace('p', '%').replace('ms', ' ms')

        file_path = os.path.join(sweep_dir, trial_file)
        if os.path.exists(file_path):
            df_trial = pd.read_csv(file_path)
            sensor_row = df_trial[df_trial['sensor_id'] == sensor_id]
            if not sensor_row.empty:
                mean_data = sensor_row[mean_cols].to_numpy().ravel()
                sweep_data["trials"].append({"label": label, "data": mean_data})
        else:
            print(f"Warning: Trial file not found at {file_path}")

    return sweep_data


def plot_emg_sweep(ax: plt.Axes, sweep_data: dict, color_map, title: str, show_legend: bool = False):
    """
    Plots a single EMG sweep, now including NoExo and NoAssi conditions.
    """
    if not sweep_data["trials"] and sweep_data["no_assi"] is None and sweep_data["no_exo"] is None:
         ax.text(0.5, 0.5, 'No Data Found', ha='center', va='center', fontsize=12, color='red')
    
    if sweep_data["no_exo"] is not None:
        ax.plot(sweep_data["no_exo"], color='#FF8C00', linewidth=2.5, label='NoExo')
    
    if sweep_data["no_assi"] is not None:
        ax.plot(sweep_data["no_assi"], color='grey', linewidth=2.0, label='NoAssi')

    num_trials = len(sweep_data["trials"])
    colors = color_map(np.linspace(0.2, 1, num_trials)) 
    
    for i, trial in enumerate(sweep_data["trials"]):
        ax.plot(trial["data"], color=colors[i], linewidth=1.5, label=trial["label"])

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(direction='out', length=6, width=1)
    ax.set_xlim(0, 99)
    ax.set_ylim(bottom=0)
    ax.grid(True, linestyle='--', alpha=0.5)

    if show_legend:
        ax.legend(
            frameon=False, 
            fontsize='medium', 
            loc='center left', 
            bbox_to_anchor=(1.02, 0.5)
        )

def main():
    """
    Main function to generate the 2x4 plot matrix for EMG sweeps.
    """
    subject_name = "AB11_Ryan"
    
    processed_dir = f"{subject_name}_processed_last10"
    walking_condition = "LG" # <<< CHANGE THIS to "RA" or "RD" for other plots

    muscles = {
        "Tibialis Anterior": 83492, "Bicep Femoris": 78532, "Gastrocnemius": 82755, "Rectus Femoris": 78475
    }
    
    magnitudes = ["10p", "15p", "20p", "25p", "30p"]
    delays = ["100ms", "150ms", "200ms", "250ms", "300ms"]
    
    # --- Define all necessary filenames dynamically ---
    no_assi_filename = f"{subject_name}_{walking_condition}_NoAssi_1.csv"
    no_exo_filename = f"{subject_name}_{walking_condition}_NoExo_1.csv"
    magnitude_sweep_files = [f"{subject_name}_{walking_condition}_{mag}100ms_1.csv" for mag in magnitudes]
    delay_sweep_files = [f"{subject_name}_{walking_condition}_30p{delay}_1.csv" for delay in delays]
    
    fig, axes = plt.subplots(2, 4, figsize=(24, 10), sharex=True, sharey=False)
    fig.suptitle(f'EMG Activation Sweeps for {walking_condition} Walking', fontsize=20, fontweight='bold')

    mag_cmap = plt.get_cmap('Blues')
    delay_cmap = plt.get_cmap('Greens')

    for i, (muscle_name, sensor_id) in enumerate(muscles.items()):
        is_last_column = (i == len(muscles) - 1)

        # --- Define directory paths ---
        condition_base_dir = os.path.join(processed_dir, walking_condition)
        mag_sweep_dir = os.path.join(condition_base_dir, "Magnitude")
        delay_sweep_dir = os.path.join(condition_base_dir, "Delay")

        # --- Magnitude Sweep ---
        mag_data = load_sweep_data(mag_sweep_dir, sensor_id, magnitude_sweep_files, no_assi_filename, no_exo_filename, label_pattern=r'_(\d+p)')
        
        # --- ADDED: Print Max Activations for Magnitude Sweep ---
        print(f"\n--- Max Activations for {muscle_name} (Magnitude Sweep, {walking_condition}) ---")
        if mag_data["no_exo"] is not None:
            print(f"NoExo   : {np.max(mag_data['no_exo']):.4f}")
        if mag_data["no_assi"] is not None:
            print(f"NoAssi  : {np.max(mag_data['no_assi']):.4f}")
        for trial in mag_data["trials"]:
            print(f"{trial['label']:<7} : {np.max(trial['data']):.4f}")

        plot_emg_sweep(axes[0, i], mag_data, mag_cmap, muscle_name, show_legend=is_last_column)

        # --- Delay Sweep ---
        delay_data = load_sweep_data(delay_sweep_dir, sensor_id, delay_sweep_files, no_assi_filename, no_exo_filename, label_pattern=r'p(\d+ms)')
        
        # --- Print Max Activations for Delay Sweep ---
        print(f"\n--- Max Activations for {muscle_name} (Delay Sweep, {walking_condition}) ---")
        if delay_data["no_exo"] is not None:
            print(f"NoExo   : {np.max(delay_data['no_exo']):.4f}")
        if delay_data["no_assi"] is not None:
            print(f"NoAssi  : {np.max(delay_data['no_assi']):.4f}")
        for trial in delay_data["trials"]:
            print(f"{trial['label']:<7} : {np.max(trial['data']):.4f}")

        plot_emg_sweep(axes[1, i], delay_data, delay_cmap, "", show_legend=is_last_column)

    # --- Set Labels and Titles ---
    for row in range(2):
        axes[row, 0].set_ylabel("EMG Activation (mV)", fontsize=12)
    for col in range(4):
        axes[1, col].set_xlabel("Gait Cycle (%)", fontsize=12)
        
    fig.text(0.04, 0.7, 'Magnitude Sweep', va='center', rotation='vertical', fontsize=16, fontweight='bold')
    fig.text(0.04, 0.3, 'Delay Sweep', va='center', rotation='vertical', fontsize=16, fontweight='bold')

    fig.subplots_adjust(
        left=0.1, right=0.9, bottom=0.1, top=0.9, 
        wspace=0.3, hspace=0.25
    )
    plt.show()

if __name__ == "__main__":
    main()

