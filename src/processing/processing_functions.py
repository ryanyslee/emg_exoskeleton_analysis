import os
import re
import numpy as np
import pandas as pd
import scipy.signal as signal
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from io import StringIO
import matplotlib.pyplot as plt

# ==============================================================================
# DATA LOADING FUNCTIONS
# ==============================================================================

def load_grf_and_trigger(path: str, trial_time: int, samp_freq: int):
    """
    Loads GRF (Ground Reaction Force) and a trigger signal from a Vicon CSV file.

    Args:
        path (str): The file path to the CSV file.
        trial_time (int): The total duration of the trial in seconds.
        samp_freq (int): The sampling frequency in Hz (e.g., 1000).

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing three NumPy arrays:
        - left_fz: Vertical force for the left plate.
        - right_fz: Vertical force for the right plate.
        - trigger: The electric potential signal from the trigger column.
    """
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()

    # ── locate the first “Frame, …” header
    # This finds the start of the actual data table.
    try:
        hdr_idx = next(i for i, ln in enumerate(lines)
                       if ln.lstrip().startswith("Frame"))
    except StopIteration:
        raise ValueError("Could not find the 'Frame' header row in the file.")
        
    header  = lines[hdr_idx].strip().split(",")
    n_cols  = len(header)

    # Define the block of lines to read based on trial duration and sample rate
    start_line = hdr_idx + 1
    end_line = start_line + 1 + (trial_time * samp_freq) # +1 for units row
    block   = lines[start_line : end_line]

    # discard the units row (which contains 'N' in an early column)
    if block and len(block[0].split(",")) > 2 and block[0].split(",")[2].strip().upper() == "N":
        block = block[1:]

    # Clean up the data: keep only rows with the correct column count
    body = [",".join(r.strip().split(",")[:n_cols])
            for r in block if len(r.strip().split(",")) == n_cols]

    # Use pandas to read the cleaned data into a DataFrame
    df = pd.read_csv(
        StringIO("\n".join([",".join(header)] + body)),
        dtype=float,          # <───────── forces float64 everywhere
        low_memory=False,
        on_bad_lines="skip",  # silently drop any malformed row
    )

    # --- Data Extraction ---
    # The file stores downward force as negative, so we multiply by -1.
    # Because there are two 'Fz' columns, pandas automatically renames the second one to 'Fz.1'.
    # From your file, 'Fz' is Right Force and 'Fz.1' is Left Force.
    left_fz  = -df["Fz.1"].to_numpy()
    right_fz = -df["Fz"  ].to_numpy()
    
    # NEW: Extract the trigger signal from the last column, which is named 'jet'
    trigger = df["jet"].to_numpy()

    return left_fz, right_fz, trigger

def load_emg(path: str) -> tuple[list[int], dict[int, np.ndarray]]:
    """
    Loads EMG data from a Trigno Discover CSV file.

    This function reads the file, extracts the sensor IDs from the metadata,
    and loads the corresponding EMG signal for each sensor. It is designed
    to handle the specific format with metadata headers followed by data columns.

    Args:
        path (str): The file path to the EMG CSV file.

    Returns:
        A tuple containing two items:
        - ids (list[int]): A list of the integer sensor IDs found in the file header.
        - signals (dict[int, np.ndarray]): A dictionary where keys are the sensor IDs
          and values are the corresponding EMG signal data as NumPy arrays.
    """
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # --- Step 1: Extract Sensor IDs from Row 4 (0-indexed 3) ---
    # This line contains text like "Avanti Sensor 7 (82511)"
    if len(lines) < 4:
        raise ValueError("File is too short to contain sensor ID information.")
    
    sensor_line = lines[3].strip()
    # Use regular expression to find all numbers inside parentheses
    sensor_ids_str = re.findall(r'\((\d+)\)', sensor_line)
    if not sensor_ids_str:
        raise ValueError("Could not find any sensor IDs in the expected format, e.g., (12345).")
        
    sensor_ids = [int(sid) for sid in sensor_ids_str]

    # --- Step 2: Load the numerical data using pandas ---
    # The actual column headers are on row 6 (0-indexed 5).
    # We need to skip rows 7 and 8 (0-indexed 6 and 7) which contain frequency info.
    df = pd.read_csv(
        path,
        header=5,
        skiprows=[6, 7],
        low_memory=False,
        na_values=[' ', ''],
        usecols=range(8),
        on_bad_lines='skip'
    )

    # --- Step 3: Select only the signal (mV) columns ---
    # The signal columns are the 2nd, 4th, 6th, and 8th columns.
    # In a 0-indexed DataFrame, these are at positions 1, 3, 5, 7.
    signal_columns = df.iloc[:, [1, 3, 5, 7]]

    # Drop any rows at the end of the file that contain NaN values
    signal_columns = signal_columns.dropna()

    # --- Step 4: Create the dictionary mapping IDs to signal arrays ---
    signals = {}
    if len(sensor_ids) == signal_columns.shape[1]:
        for i, sensor_id in enumerate(sensor_ids):
            # Assign each signal column to its corresponding sensor ID
            signals[sensor_id] = signal_columns.iloc[:, i].to_numpy()
    else:
        raise ValueError(f"Mismatch between found IDs ({len(sensor_ids)}) and signal columns ({signal_columns.shape[1]}).")

    return sensor_ids, signals


# ==============================================================================
# SIGNAL PROCESSING FUNCTIONS
# ==============================================================================

def butterworth_filter(data, cutoff, samp_freq, order=5):
    b, a = signal.butter(order, cutoff / (samp_freq / 2), btype='low')
    
    data = np.asarray(data)
    if np.isnan(data).any():
        # split contiguous finite chunks, filter each
        y = np.full_like(data, np.nan)
        finite = np.isfinite(data).all(aixs=-1) if data.ndim == 2 else np.isfinite(data)
        edges = np.diff(np.concatenate(([0], finite.view(np.int8), [0])))
        starts = np.flatnonzero(edges == 1)
        ends = np.flatnonzero(edges == -1) - 1
        for s, e, in zip(starts, ends):
            if e > s:
                y[s:e+1] = signal.filtfilt(b, a, data[s:e+1], axis=0)
        return y
    else:
        return signal.filtfilt(b, a, data, axis=0)
    
def butter_bandpass(data, lo, hi, fs, order=4):
        b, a = butter(order, [lo/(fs/2), hi/(fs/2)], btype='band')
        return filtfilt(b, a, data, axis=0)

def butter_lowpass(data, fc, fs, order=4):
    b, a = butter(order, fc/(fs/2), btype='low')
    return filtfilt(b, a, data, axis=0)

# Band-pass filter, rectify, and (low-pass or RMS) envelope EMG
def preprocess_emg(raw_data, samp_freq, bp_lo=20, bp_hi=450, envelope_fc=8, order=4, use_rms=False):
    """
    Preprocesses EMG data.
    Returns ONLY the envelope, compatible with the emg_process loop.
    """
    # Remove DC offset
    demeaned = raw_data - np.nanmean(raw_data)
    
    # Band-pass filter (Zero-lag)
    bp = butter_bandpass(demeaned, bp_lo, bp_hi, samp_freq, order=order)
    
    # Rectify
    rect = np.abs(bp)
    
    # Envelope
    if use_rms:
        window_ms = 50
        window_samples = int((window_ms / 1000) * samp_freq)
        # Moving RMS calculation
        env = np.sqrt(np.convolve(rect**2, np.ones(window_samples)/window_samples, mode='same'))
    else:
        # Standard Linear Envelope (Lowpass)
        # THIS IS THE ONE WE ARE USING!!!
        env = butter_lowpass(rect, envelope_fc, samp_freq, order=order)
    
    return env

def resample_data(
    orig_data: np.ndarray,
    orig_freq: float,
    target_length: int
) -> np.ndarray:
    """
    Resamples a signal to a new length using cubic interpolation.

    This is ideal for upsampling a lower-frequency signal (like GRF) to match the
    length of a higher-frequency signal (like EMG).

    Args:
        original_signal (np.ndarray): The input signal array to be resampled.
        original_freq (float): The original sampling frequency of the input signal (in Hz).
        target_length (int): The desired number of samples in the output array.

    Returns:
        np.ndarray: The resampled signal with a length equal to target_length.
    """
    # 1. Calculate the total duration of the original signal
    original_duration = len(orig_data) / orig_freq

    # 2. Create the time vector for the original signal
    original_time = np.linspace(0, original_duration, len(orig_data))

    # 3. Create the new, high-resolution time vector for the target signal
    target_time = np.linspace(0, original_duration, target_length)

    # 4. Create an interpolation function from the original data
    # 'cubic' provides a smooth interpolation, which is great for GRF data.
    interpolator = interp1d(original_time, orig_data, kind='cubic', fill_value="extrapolate")

    # 5. Generate the new, resampled signal by applying the interpolator to the target time
    resampled_signal = interpolator(target_time)

    return resampled_signal


# ==============================================================================
# EVENT DETECTION & SEGMENTATION FUNCTIONS
# ==============================================================================

def find_first_trigger_index(trigger_signal: np.ndarray, threshold: float = 1.0) -> int | None:
    """
    Finds the index of the first occurrence in an array where the value exceeds a threshold.

    Args:
        trigger_signal (np.ndarray): The input signal array.
        threshold (float): The value the signal must exceed. Defaults to 1.0.

    Returns:
        int | None: The index of the first trigger occurrence. 
                    Returns None if no value exceeds the threshold.
    """
    # np.where returns a tuple of arrays, one for each dimension
    indices = np.where(trigger_signal > threshold)[0]
    
    if indices.size > 0:
        # We only need the very first index found
        return indices[0]
    else:
        # No trigger was found in the signal
        return None

def find_emg_index(grf_index, scaling_factor):
    return int(round(grf_index * scaling_factor))

def find_heel_contacts(grf_data, weight):
    '''
    Returns
    heel_contacts : list of int
        Indices in 'grf_data' where heel-strike occurs
    '''
    heel_contacts = []
    threshold = weight * 9.80665 * 0.05
    for i in range(1, len(grf_data)):
        if grf_data[i-1] < threshold and grf_data[i] >= threshold:
            heel_contacts.append(i)
    return heel_contacts

def seg_and_resamp(data, heel_contacts, resamp=True):
    '''
    Returns
    cycles : list
        Each element is either an ndarray of the cycle or None if NaNs were present
    nan_mask : list of bool
        True if that cycle contained NaNs
    '''
    data = np.asarray(data)
    cycles, nan_mask = [], []
    
    # Looks at the data between two consecutive heel contacts
    for start, end in zip(heel_contacts[:-1], heel_contacts[1:]):
        seg = data[start:end] # end is exclusive
        has_nan = np.isnan(seg).any()
        nan_mask.append(has_nan) # [True if NaN in cycle, False if clean]
        
        if has_nan:
            cycles.append(None)
        else:
            if resamp:
                cycles.append(signal.resample(seg, 100, axis=0))
            else:
                cycles.append(seg)
    return cycles, nan_mask


# ==============================================================================
# NORMALIZATION
# ==============================================================================

def get_global_normalization_factors(log_path: str) -> dict:
    """
    Reads the subject_peaks_log.csv and returns the global min and max
    for each sensor_id for the specified subject.
    """
    if not os.path.exists(log_path):
        print(f"Warning: Peaks log not found at {log_path}. Normalization will be skipped.")
        return {}
    
    df = pd.read_csv(log_path)
    
    # Group by sensor_id to find global min/max across all trials
    global_stats = df.groupby("sensor_id").agg({
        "trial_max": "max", # finds the highest value in the trial_max column
        "trial_min": "min" # finds the lowest value in the trial_min column
    }).to_dict('index')
    
    for sid, stats in global_stats.items():
        if stats['trial_min'] < 0:
            stats['trial_min'] = 0.0
    
    return global_stats # {82718: {'trial_max': 0.8, 'trial_min': 0.0},...}
    
def normalize_data(data: np.ndarray, sensor_id: int, global_stats: dict) -> np.ndarray:
    """
    Applies min-max normalization using global limits.
    """
    if sensor_id not in global_stats:
        return data # Returns raw if no stats found
        
    g_min = global_stats[sensor_id]['trial_min'] # global minimum
    g_max = global_stats[sensor_id]['trial_max'] # global maximum
    denom = g_max - g_min
    
    if denom == 0:
        return np.zeros_like(data)

    return (data - g_min) / denom

# ==============================================================================
# PLOTTING & TESTING
# ==============================================================================

def plot_emg_comparison(raw_signal: np.ndarray, filtered_signal: np.ndarray,
    samp_freq: float,title: str = "EMG Signal Comparison: 25p100ms"
):
    """
    Plots the raw EMG signal overlaid with its filtered version.

    Args:
        raw_signal (np.ndarray): The unprocessed EMG signal.
        filtered_signal (np.ndarray): The processed/filtered EMG signal.
        samp_freq (float): The sampling frequency of the signals in Hz.
        title (str): The title for the plot.
    """
    # --- Correctly create the time vector ---
    # It needs to have the same number of points as the signal arrays.
    total_duration_secs = len(raw_signal) / samp_freq
    time_vector = np.linspace(0, total_duration_secs, len(raw_signal))

    # --- Create the plot ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(15, 6))

    # Plot the raw signal in a lighter color and semi-transparent
    ax.plot(time_vector, raw_signal, color='lightgrey', alpha=0.8, label='Raw EMG')

    # Plot the filtered signal on top in a stronger color
    ax.plot(time_vector, filtered_signal, color='dodgerblue', linewidth=1.5, label='Filtered EMG')

    # --- Add labels and title for clarity ---
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Amplitude (mV)", fontsize=12)
    ax.legend()
    ax.grid(True)
    
    # Set reasonable limits to avoid extreme peaks from dominating the view
    ax.set_xlim(time_vector[0], time_vector[-1])

    plt.tight_layout()
    plt.show()

def plot_grf_unilateral(time_s, grf, heel_idx, *,
                        foot:str = "left",
                        title:str|None = None,
                        grf_kw:dict|None = None,
                        hc_kw:dict|None  = None):
    grf_kw = grf_kw or {}
    hc_kw  = hc_kw  or {}

    colour = "tab:blue" if foot.lower().startswith("l") else "tab:orange"
    default_grf = dict(color=colour, lw=1.2, label=f"GRF-{foot[0].upper()}")
    default_hc  = dict(color=colour, ls="--", lw=.8, alpha=.6)

    fig, ax = plt.subplots(figsize=(15,4))
    ax.plot(time_s, grf, **(default_grf | grf_kw))

    for idx in heel_idx:
        ax.axvline(time_s[idx], **(default_hc | hc_kw))

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Force (N)")
    ax.set_title(title or f"{foot.capitalize()} GRF with heel-strike markers")
    ax.grid(alpha=.2)
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.show()

def analyze_trial(emg_path, grf_path):
    
    print(f"--- Analyzing Trial ---")
    print(f"EMG: {os.path.basename(emg_path)}")
    print(f"GRF: {os.path.basename(grf_path)}")

    # 1. Get Slicing Indices from GRF
    grf_l_raw, grf_r_raw, trigger = load_grf_and_trigger(grf_path, 130, 1000)

    # 2. Load EMG
    sensor_ids, emg_signals = load_emg(emg_path)
    
    # Calculate Frequency and Scaling Factor
    # Assuming GRF is 130s and EMG is 130s, but different rates
    emg_1_raw = emg_signals[sensor_ids[0]]
    emg_freq = len(emg_1_raw) / 130
    scaling_factor = len(emg_1_raw) / 130000
    
    processed_emgs = {sid: preprocess_emg(data, emg_freq) for sid, data in emg_signals.items()}
    
    grf_i0 = find_first_trigger_index(trigger)
    grf_i1 = grf_i0 + 119999
    
    # Calculate EMG Slice Indices
    emg_i0 = find_emg_index(grf_i0, scaling_factor)
    emg_i1 = find_emg_index(grf_i1, scaling_factor)
    emgs_sliced = {sid: data[emg_i0:emg_i1+1] for sid, data in processed_emgs.items()}
    
    print(f"EMG Sampling Freq: {emg_freq:.1f} Hz")
    print(f"Slicing Interval: Index {emg_i0} to {emg_i1}")

    # 3. Process and Plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f"Filtered EMG & Min/Max Analysis\n{os.path.basename(emg_path)}", fontsize=16)
    axes = axes.flatten()
    
    # Define mapping based on Index: 0=TA, 1=GM, 2=RF, 3=BF
    index_map = {
        0: "Tibialis Anterior",
        1: "Gastrocnemius",
        2: "Rectus Femoris",
        3: "Bicep Femoris"
    }
    
    results = []

    # Iterate through the 4 subplots
    for i, ax in enumerate(axes):
        if i >= len(sensor_ids):
            ax.text(0.5, 0.5, "No Sensor Data", ha='center')
            continue
            
        sid = sensor_ids[i]
        muscle_name = index_map.get(i, f"Sensor {i}")
        
        data = emgs_sliced[sid]
        
        # --- Stats Calculation ---
        true_max = np.max(data)
        true_min = np.min(data)
        
        # Calculate Robust Max (99.0th Percentile)
        # You can adjust this to 99.5 if 99.0 is too aggressive
        robust_max = np.percentile(data, 99.5)
        
        results.append({
            "Muscle": muscle_name,
            "Abs Max": true_max,
            "99.5% Max": robust_max
        })
        
        # --- Plotting ---
        time_vector = np.arange(len(data)) / emg_freq
        
        # 1. Main Signal
        ax.plot(time_vector, data, label='Envelope', color='cornflowerblue', linewidth=0.8, alpha=0.8)
        
        # 2. Absolute Max Line (Orange Dotted)
        ax.axhline(true_max, color='orange', linestyle=':', linewidth=1.5, label='Abs Max')
        
        # 3. Robust Percentile Max Line (Red Dashed) - THIS IS WHAT YOU WANTED
        ax.axhline(robust_max, color='red', linestyle='--', linewidth=1.5, label='99.5% Max')
        
        # 4. Zero Line
        ax.axhline(0, color='black', linewidth=0.5)
        
        # 5. Highlight Min (to check for negatives)
        min_loc = np.argmin(data)
        ax.scatter(time_vector[min_loc], true_min, color='black', s=20, zorder=5)

        # Labels and Styling
        ax.set_title(f"{muscle_name} (ID: {sid})")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("mV")
        ax.legend(loc='upper right', fontsize='small', frameon=True)
        ax.grid(True, alpha=0.3)
        
        # Display values on plot
        stats_text = f"Max: {true_max:.3f}\n99.5%: {robust_max:.3f}\nMin: {true_min:.3f}"
        ax.text(0.02, 0.95, stats_text, transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.show()

    # 4. Print Console Summary
    print("\n--- Summary Statistics ---")
    print(f"{'Muscle':<20} | {'Abs Max':<10} | {'99.5% Max':<10} | {'Diff'}")
    print("-" * 55)
    for r in results:
        diff = r['Abs Max'] - r['99.5% Max']
        print(f"{r['Muscle']:<20} | {r['Abs Max']:.4f}     | {r['99.5% Max']:.4f}     | {diff:.4f}")

def emg_process(grf_path: str, emg_path: str, weight: float, output_path: str, is_no_exo_trial: bool = False, is_RD: bool = False):
    """
    Processes a single pair of GRF and EMG files and saves the summary CSV.
    Handles 'No Exo' trials by using the full 35s duration.
    """
    # --- 1. Load and Pre-process GRF Data ---
    grf_l_raw, grf_r_raw, trigger = load_grf_and_trigger(grf_path, 130, 1000)
    
    if is_RD:
        grf_r_filt = butterworth_filter(grf_l_raw, 6, 1000)
    else:
        grf_r_filt = butterworth_filter(grf_r_raw, 6, 1000)
    
    # --- 2. Load and Pre-process EMG Data ---
    sensor_ids, emg_signals = load_emg(emg_path)
    emg_1_raw = emg_signals[sensor_ids[0]]
    emg_freq = len(emg_1_raw) / 130 # sampling frequency of EMG data
    
    # Dictionary of processed EMG signals (Full 130-second trial)
    processed_emgs = {sid: preprocess_emg(data, emg_freq) for sid, data in emg_signals.items()}

    # --- 3. Conditional data slicing based on trial type ---
    if is_no_exo_trial:
        print("  -> NOTE: 'No Exo' trial detected. Using full 35-second duration.")
        grf_r_processed = grf_r_filt
        emgs_processed = processed_emgs
    else:
        # Original logic for all other trials
        grf_i0 = find_first_trigger_index(trigger)
        if grf_i0 is None:
            print(f"  -> SKIPPING: No trigger found in {grf_path}")
            return
        
        grf_i1 = grf_i0 + 119999 # Retrieving 120-second window using trigger
        if grf_i1 >= len(grf_r_filt):
            print(f"  -> SKIPPING: Not enough data for a 120s slice from trigger in {grf_path}")
            return
            
        grf_r_processed = grf_r_filt[grf_i0: grf_i1 + 1] # 120-second GRF

        scaling_factor = len(emg_1_raw) / 130000 # 130s * 1000Hz
        emg_i0 = find_emg_index(grf_i0, scaling_factor)
        emg_i1 = find_emg_index(grf_i1, scaling_factor)
        emgs_processed = {sid: data[emg_i0:emg_i1+1] for sid, data in processed_emgs.items()}
    
    # --- 4. Upsample GRF and Find Heel Contacts ---
    # 1000 Hz -> 2148 Hz
    grf_r_resamp = resample_data(grf_r_processed, 1000, len(emgs_processed[sensor_ids[0]]))
    hc_r = find_heel_contacts(grf_r_resamp, weight)
    
    # --- 5. Segment, Normalize, and Average EMG Cycles ---
    emg_gcs = {sid: seg_and_resamp(emg_data, hc_r)[0] for sid, emg_data in emgs_processed.items()}

    def avg_sd_100(cycle_list, fill=np.nan):
        valid = [c for c in cycle_list if c is not None and len(c) == 100]
        if not valid:
            return np.full(100, fill), np.full(100, fill)
        stack = np.vstack(valid)
        return stack.mean(axis=0), stack.std(axis=0, ddof=0)

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
    weight = 69.0 # [kg]
    
    input_emg_file = r"N=8 Original Data/AB01_Jimin/RD/EMG/30p300ms.csv"
    input_grf_file = r"N=8 Original Data/AB01_Jimin/RD/GRF/AB01_Jimin_RD_30p300ms_1.csv"
    output_path = "AB01_Jimin_RD_30p300ms_1.csv"
    
    emg_process(input_grf_file, input_emg_file, weight, output_path, is_no_exo_trial=False, is_RD=True)
    
    # analyze_trial(input_emg_file, input_grf_file)
    
    # _, grf_r_raw, trigger = load_grf_and_trigger(grf_path, 130, 1000)
    
    # print('grf_r:', len(grf_r_raw))
    
    # grf_r_filt = butterworth_filter(grf_r_raw, 6, 1000)
    
    # grf_i0 = find_first_trigger_index(trigger)
    # grf_i1 = grf_i0 + 119999
    
    # print('grf_i0:', grf_i0)
    # print('grf_i1:', grf_i1)
    
    # # Extracted 30-second grf data
    # grf_r_30 = grf_r_filt[grf_i0: grf_i1 + 1]
    
    # print('grf_r_2min:', len(grf_r_30))
    
    # sensor_ids, emg_signals = load_emg(emg_file)
    
    # emg_1 = emg_signals[sensor_ids[0]]
    # emg_2 = emg_signals[sensor_ids[1]]
    # emg_3 = emg_signals[sensor_ids[2]]
    # emg_4 = emg_signals[sensor_ids[3]]
    
    # print('emg_1:', len(emg_1))
    
    # emg_freq = len(emg_1) / 130
    
    # emg_1_filt = preprocess_emg(emg_1, emg_freq)
    # emg_2_filt = preprocess_emg(emg_2, emg_freq)
    # emg_3_filt = preprocess_emg(emg_3, emg_freq)
    # emg_4_filt = preprocess_emg(emg_4, emg_freq)
    
    # scaling_factor = len(emg_1) / 130000 # 130s * 1000Hz
    
    # print("scaling factor", scaling_factor)
    
    # emg_i0 = find_emg_index(grf_i0, scaling_factor)
    # emg_i1 = find_emg_index(grf_i1, scaling_factor)
    
    # emg1_30 = emg_1_filt[emg_i0: emg_i1 + 1]
    # emg2_30 = emg_2_filt[emg_i0: emg_i1 + 1]
    # emg3_30 = emg_3_filt[emg_i0: emg_i1 + 1]
    # emg4_30 = emg_4_filt[emg_i0: emg_i1 + 1]
    
    # emg_30 = emg_1[emg_i0: emg_i1 + 1]
    # demeaned_emg = emg_30 - np.nanmean(emg_30)
    # plot_emg_comparison(demeaned_emg, emg1_30, emg_freq)
    
    
    # print('emg_i0:', emg_i0)
    # print('emg_i1:', emg_i1)
    # print('emg1_30:', len(emg1_30))
    # print('emg2_30:', len(emg2_30))
    # print('emg3_30:', len(emg3_30))
    # print('emg4_30:', len(emg4_30))
    
    # # Upsample GRF data to EMG frequency
    # grf_r_resamp = resample_data(grf_r_30, 1000, len(emg1_30))
    
    # print('grf_r_resamp:', len(grf_r_resamp))
    
    # hc_r = find_heel_contacts(grf_r_resamp, weight)
    
    # emg1_gc, _ = seg_and_resamp(emg1_30, hc_r)
    # emg2_gc, _ = seg_and_resamp(emg2_30, hc_r)
    # emg3_gc, _ = seg_and_resamp(emg3_30, hc_r)
    # emg4_gc, _ = seg_and_resamp(emg4_30, hc_r)
    
    
    
    # time = np.linspace(0, 120, len(grf_r_resamp))
    # plot_grf_unilateral(time, grf_r_resamp, hc_r, foot='right')
    
   
if __name__ == "__main__":
    main()