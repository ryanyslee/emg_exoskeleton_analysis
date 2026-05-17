import numpy as np
import pandas as pd
import os
import scipy.signal as signal
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from io import StringIO
import matplotlib.pyplot as plt
import re

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
    Returns ONLY the envelope, compatible with your emg_process loop.
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
# PLOTTING HELPER FUNCTIONS
# ==============================================================================

def plot_emg_envelope(
    ax: plt.Axes,
    mean_signal: np.ndarray,
    std_signal: np.ndarray,
    title: str = "Average EMG Activation",
    color: str = 'dodgerblue'
):
    """
    Creates a plot of the average EMG activation envelope with a shaded
    standard deviation area on a given matplotlib Axes object.

    Args:
        ax (plt.Axes): The matplotlib Axes object to plot onto.
        mean_signal (np.ndarray): A 100-point array of the mean EMG activation.
        std_signal (np.ndarray): A 100-point array of the standard deviation.
        title (str): The title for the subplot.
        color (str): The base color for the plot elements.
    """
    # Check for valid data first
    is_valid_data = isinstance(mean_signal, np.ndarray) and \
                    isinstance(std_signal, np.ndarray) and \
                    mean_signal.shape == (100,) and \
                    std_signal.shape == (100,) and \
                    not np.isnan(mean_signal).all() and \
                    not np.isnan(std_signal).all()

    if not is_valid_data:
        # Handle cases where data might be missing or malformed on the given axis
        print(f"Warning: Invalid or missing data for '{title}'. Plotting placeholder.")
        ax.text(0.5, 0.5, 'Invalid or\nMissing Data', ha='center', va='center', fontsize=12, color='red', transform=ax.transAxes)
        ax.set_title(title, fontsize=14, fontweight='bold') # Still set title
        # Set basic limits and grid for consistency
        ax.set_xlim(0, 99)
        ax.set_ylim(0, 0.1) # Set a minimal default ylim
        ax.grid(True, linestyle='--', alpha=0.5)
        # Remove ticks if no data
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        return

    gait_cycle = np.arange(100)  # X-axis from 0% to 99%
    upper_bound = mean_signal + std_signal
    # Ensure lower bound doesn't go below zero
    lower_bound = np.maximum(mean_signal - std_signal, 0)

    # Plot the mean activation line
    ax.plot(gait_cycle, mean_signal, color=color, linewidth=2, label='Mean Activation')

    # Plot the shaded standard deviation area
    ax.fill_between(gait_cycle, lower_bound, upper_bound, color=color, alpha=0.2, label='± 1 Std. Dev.')

    # --- Add labels and title for clarity ---
    ax.set_title(title, fontsize=14, fontweight='bold') # Slightly smaller title for subplots
    ax.legend(frameon=False, fontsize='small') # Smaller legend
    ax.grid(True, linestyle='--', alpha=0.5)

    # --- Apply styling ---
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(direction='out', length=6, width=1, labelsize=10) # Smaller ticks
    ax.set_xlim(0, 99)
    ax.set_ylim(bottom=0) # EMG envelope is always positive
    # ax.set_ylim(top=np.max(upper_bound) * 1.1) # Auto-scale y-axis slightly above max std dev


# ==============================================================================
# MAIN PROCESSING PIPELINE
# ==============================================================================

def emg_process(grf_path: str, emg_path: str, weight: float, is_no_exo_trial: bool = False, is_RD: bool = False):
    """
    Processes GRF and EMG files and returns a dictionary of mean/std statistics.

    Args:
        grf_path (str): Path to the GRF file.
        emg_path (str): Path to the EMG file.
        weight (float): Participant weight in kg.
        is_no_exo_trial (bool): Flag for 'No Exo' trials (uses full duration).

    Returns:
        dict[int, tuple[np.ndarray, np.ndarray]]:
            A dictionary where keys are sensor IDs and values are a tuple
            containing (mean_100_point_array, std_100_point_array).
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
    emg_freq = len(emg_1_raw) / 130
    
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
        
        grf_i1 = grf_i0 + 119999
        if grf_i1 >= len(grf_r_filt):
            print(f"  -> SKIPPING: Not enough data for a 30s slice from trigger in {grf_path}")
            return
            
        grf_r_processed = grf_r_filt[grf_i0: grf_i1 + 1]

        scaling_factor = len(emg_1_raw) / 130000
        emg_i0 = find_emg_index(grf_i0, scaling_factor)
        emg_i1 = find_emg_index(grf_i1, scaling_factor)
        emgs_processed = {sid: data[emg_i0:emg_i1+1] for sid, data in processed_emgs.items()}
    
    # --- 4. Upsample GRF and Find Heel Contacts ---
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
    
    return stats

def main():
    """
    Main function to run the full processing and plotting pipeline.
    """
    
    WEIGHT = 80.5 #kg
    GRF_PATH = "AB16_Ilseung_RD_20p250ms_1.csv"
    EMG_PATH = "20p250ms.csv"
    IS_NO_EXO = False # set to True for 'No Exo' trials, False otherwise
    IS_RD = True
    
    # Define the muscles and their corresponding sensor IDs
    MUSCLES_TO_PLOT = {
        "Tibialis Anterior": 82719,
        "Bicep Femoris": 83449,
        "Gastrocnemius": 82653,
        "Rectus Femoris": 83426
    }

    print(f"Processing data for {GRF_PATH} and {EMG_PATH}...")
    try:
        # The 'stats' dict format is: {sensor_id: (mean_array, std_array)}
        stats = emg_process(
            GRF_PATH, 
            EMG_PATH, 
            WEIGHT, 
            is_no_exo_trial=IS_NO_EXO,
            is_RD = IS_RD
        )
        if not stats:
            print("Processing returned no data. Aborting plot.")
            return
    except Exception as e:
        print(f"An error occurred during data processing: {e}")
        import traceback
        traceback.print_exc()
        return

    # Generating Plots
    print("Processing complete. Generating plot...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel() # Flatten the 2x2 array for easy iteration

    plot_index = 0
    for muscle_name, sensor_id in MUSCLES_TO_PLOT.items():
        if plot_index >= len(axes):
            print(f"Warning: More muscles ({len(MUSCLES_TO_PLOT)}) than plot axes ({len(axes)}). Skipping '{muscle_name}'.")
            break
            
        current_ax = axes[plot_index]
        plot_title = f"{muscle_name}"

        # Get data directly from the stats dictionary
        mean_activation, std_activation = stats.get(
            sensor_id, 
            (np.full(100, np.nan), np.full(100, np.nan))
        )

        if np.isnan(mean_activation).all():
            print(f"\nWarning: Sensor ID {sensor_id} ({muscle_name}) not found in processed data.")

        # Generate the plot on the specific subplot axis
        plot_emg_envelope(current_ax, mean_activation, std_activation, title=plot_title)
        plot_index += 1

    # Hide any unused subplots
    for i in range(plot_index, len(axes)):
        axes[i].axis('off')

    # Finalize and show plot
    fig.text(0.5, 0.02, 'Gait Cycle (%)', ha='center', va='center', fontsize=14)
    fig.text(0.04, 0.5, 'EMG Activation (mV)', ha='center', va='center', rotation='vertical', fontsize=14)
    plot_title = f"EMG Activation Summary: {os.path.basename(GRF_PATH)}"
    fig.suptitle(plot_title, fontsize=18, fontweight='bold')

    plt.tight_layout(rect=[0.05, 0.05, 1, 0.95])
    plt.show()

    print("\nFinished plotting.")

if __name__ == "__main__":
    main()

