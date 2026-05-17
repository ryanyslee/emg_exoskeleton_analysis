import os
import re
import pandas as pd
import numpy as np
from processing_functions import (
    load_grf_and_trigger,
    load_emg,
    butterworth_filter,
    preprocess_emg,
    find_first_trigger_index,
    find_heel_contacts,
    resample_data,
    seg_and_resamp,
    find_emg_index # Assuming find_emg_index is in processing_functions
)

def emg_process(grf_path: str, emg_path: str, weight: float, output_path: str, is_no_exo_trial: bool = False, is_ramp_descent: bool = False):
    """
    Processes a single pair of GRF and EMG files and saves the summary CSV.
    Handles 'No Exo' trials by using the full 35s duration.
    """
    # --- 1. Load and Pre-process GRF Data ---
    # Only need data for one foot but needs to be held constant for the entire processing
    # Let's stick to RIGHT
    grf_l_raw, grf_r_raw, trigger = load_grf_and_trigger(grf_path, 130, 1000)
    
    # For RD conditions, Right and Left force plate switches
    if is_ramp_descent:
        grf_r_filt = butterworth_filter(grf_l_raw, 6, 1000)
    else:
        grf_r_filt = butterworth_filter(grf_r_raw, 6, 1000)
    
    # --- 2. Load and Pre-process EMG Data ---
    sensor_ids, emg_signals = load_emg(emg_path)
    emg_1_raw = emg_signals[sensor_ids[0]]
    emg_freq = len(emg_1_raw) / 130 # Each trial is 130 seconds long
    
    # Preprocessing (filter + rectify) the EMG signals
    # We are using LOW PASS, NOT RMS
    processed_emgs = {sid: preprocess_emg(data, emg_freq) for sid, data in emg_signals.items()}

    # --- 3. Conditional data slicing based on trial type ---
    if is_no_exo_trial:
        # NO trigger for no exo conditions -> need to use the full duration
        print("  -> NOTE: 'No Exo' trial detected. Using full duration.")
        grf_r_processed = grf_r_filt
        emgs_processed = processed_emgs
    else:
        # Original logic for all other trials
        grf_i0 = find_first_trigger_index(trigger)
        if grf_i0 is None:
            print(f"  -> SKIPPING: No trigger found in {grf_path}")
            return
        
        grf_i1 = grf_i0 + 119999 # to capture 120 second (2 min) data
        if grf_i1 >= len(grf_r_filt):
            print(f"  -> SKIPPING: Not enough data for a 120s slice from trigger in {grf_path}")
            return
            
        # Capturing the correct GRF window
        grf_r_processed = grf_r_filt[grf_i0: grf_i1 + 1]

        scaling_factor = len(emg_1_raw) / 130000 # 130s * 1000Hz
        emg_i0 = find_emg_index(grf_i0, scaling_factor)
        emg_i1 = find_emg_index(grf_i1, scaling_factor)
        
        # Capturing the correct EMG window
        emgs_processed = {sid: data[emg_i0:emg_i1+1] for sid, data in processed_emgs.items()}
    
    # --- 4. Upsample GRF and Find Heel Contacts ---
    # Upsamping GRF Data (1000 Hz -> 2148 Hz)
    grf_r_resamp = resample_data(grf_r_processed, 1000, len(emgs_processed[sensor_ids[0]]))
    
    # Finding the heel contacts in the upsampled GRF data
    hc_r = find_heel_contacts(grf_r_resamp, weight)
    
    # Logging the max and min value for every EMG trial
    trial_peaks = []
    for sensor_id, data in emgs_processed.items():
        # Use data between the first and last heel contacts
        # This removes data corrupted with noise at the two ends
        emg_sliced = data[hc_r[0] : hc_r[-1] + 1]
        
        # Using percentile instead of taking the absolute maximum
        true_max = np.percentile(emg_sliced, 99.5)
        true_min = np.min(emg_sliced)
        
        trial_peaks.append({
            "filename": os.path.basename(grf_path),
            "sensor_id": sensor_id,
            "trial_max": true_max,
            "trial_min": true_min
        })
       
    peaks_df = pd.DataFrame(trial_peaks)
    log_path = os.path.join(os.path.dirname(output_path), "..", "subject_peaks_log.csv")
    peaks_df.to_csv(log_path, mode='a', header=not os.path.exists(log_path), index=False)
        
    # --- 5. Segment, Normalize, and Average EMG Cycles ---
    # Array of Gait Phase Cycles [(100,), (100,),..] distinguished by heel contacts
    emg_gcs = {sid: seg_and_resamp(emg_data, hc_r)[0] for sid, emg_data in emgs_processed.items()}

    def avg_sd_100(cycle_list, fill=np.nan):
        valid = [c for c in cycle_list if c is not None and len(c) == 100]
        if not valid:
            return np.full(100, fill), np.full(100, fill)
        stack = np.vstack(valid)
        mean_100 = stack.mean(axis=0)
        std_100 = stack.std(axis=0, ddof=0)
        return mean_100, std_100

    # Calculated Mean and Std for all gait cycles identified in the trial
    stats = {sid: avg_sd_100(data[-10:]) for sid, data in emg_gcs.items()}
    
    # --- 6. Format and Save Results ---
    rows = []
    for sensor_id, (mu, sd) in stats.items():
        rows.append({
            "sensor_id": sensor_id,
            **{f"mean_{p:02d}%": mu[p] for p in range(100)},
            **{f"sd_{p:02d}%": sd[p] for p in range(100)},
        })

    df_emg = pd.DataFrame(rows)
    df_emg.to_csv(output_path, index=False)
    print(f"  -> SUCCESS: Saved processed file to {output_path}")

def main():
    """
    Main function to orchestrate the batch processing of all trials.
    """
    N_sub = 8
    subject_id = "AB15_Daniel"
    subject_path = os.path.join(f"N={N_sub} Original Data", subject_id)
    weight = 69.0  # kg
    
    root_dir = subject_path
    # output_dir = os.path.join(f"N={N_sub} Processed Data", f"{subject_id}_processed")

    output_dir = f"{subject_id}_processed"
    
    conditions = ["LG", "RA", "RD"]
    
    print(f"Starting batch processing for subject: {subject_id}")
    
    for condition in conditions:
        is_RD = (condition == "RD")
        grf_folder = os.path.join(root_dir, condition, "GRF")
        emg_folder = os.path.join(root_dir, condition, "EMG")
        output_condition_folder = os.path.join(output_dir, condition)
        
        os.makedirs(output_condition_folder, exist_ok=True)
        
        print(f"\nProcessing condition: {condition}...")
        
        if not os.path.isdir(grf_folder):
            print(f"  - WARNING: GRF folder not found at {grf_folder}. Skipping.")
            continue
        
        for grf_filename in os.listdir(grf_folder):
            if not grf_filename.endswith('.csv'):
                continue

            print(f"- Found GRF file: {grf_filename}")
            grf_path = os.path.join(grf_folder, grf_filename)
            
            emg_filename = ""
            is_no_exo = "NoExo" in grf_filename

            if "NoAssi" in grf_filename:
                emg_filename = "NoAssi.csv"
            elif is_no_exo:
                emg_filename = "NoExo.csv"
            else:
                match = re.search(r'_(\d+p)(\d+ms)_', grf_filename)
                if match:
                    magnitude, delay = match.groups()
                    emg_filename = f"{magnitude.replace('_', '')}{delay.replace('_', '')}.csv"

            if not emg_filename:
                print(f"  -> SKIPPING: Could not determine matching EMG file for {grf_filename}")
                continue

            emg_path = os.path.join(emg_folder, emg_filename)
            
            if not os.path.exists(emg_path):
                print(f"  -> SKIPPING: Matching EMG file not found at {emg_path}")
                continue
            
            output_filename = grf_filename
            output_path = os.path.join(output_condition_folder, output_filename)
            
            try:
                # Pass the is_no_exo_trial flag to the processing function
                emg_process(grf_path, emg_path, weight, output_path, is_no_exo_trial=is_no_exo, is_ramp_descent=is_RD)
            except Exception as e:
                print(f"  -> FAILED: An error occurred while processing {grf_filename}: {e}")

    print("\nBatch processing complete.")

if __name__ == "__main__":
    main()

