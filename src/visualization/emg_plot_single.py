import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_emg_envelope(
    mean_signal: np.ndarray,
    std_signal: np.ndarray,
    title: str = "Average EMG Activation",
    color: str = 'dodgerblue'
):
    """
    Creates a plot of the average EMG activation envelope with a shaded
    standard deviation area.

    Args:
        mean_signal (np.ndarray): A 100-point array of the mean EMG activation.
        std_signal (np.ndarray): A 100-point array of the standard deviation.
        title (str): The title for the plot.
        color (str): The base color for the plot elements.
    """
    if mean_signal.shape != (100,) or std_signal.shape != (100,):
        raise ValueError("Input signals must be 100-point NumPy arrays.")

    gait_cycle = np.arange(100)  # X-axis from 0% to 99%
    upper_bound = mean_signal + std_signal
    lower_bound = mean_signal - std_signal

    # --- Create the plot ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot the mean activation line
    ax.plot(gait_cycle, mean_signal, color=color, linewidth=2, label='Mean Activation')

    # Plot the shaded standard deviation area
    ax.fill_between(gait_cycle, lower_bound, upper_bound, color=color, alpha=0.2, label='± 1 std')

    # --- Add labels and title for clarity ---
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel("Gait Cycle (%)", fontsize=12)
    ax.set_ylabel("Amplitude", fontsize=12)
    ax.legend()
    ax.grid(True)
    
    # Set clear limits for the axes
    ax.set_xlim(0, 99)
    ax.set_ylim(bottom=0) # EMG envelope is always positive

    plt.tight_layout()
    plt.show()

def main():
    """
    Main function to load the processed EMG data and generate a plot
    for each muscle.
    """
    try:
        df = pd.read_csv("emg_processed.csv")
    except FileNotFoundError:
        print("Error: 'emg_processed.csv' not found. Please run the previous script first.")
        return

    # Loop through each row of the DataFrame (each row is one sensor)
    for index, row in df.iterrows():
        sensor_id = row['sensor_id']

        # Extract the mean and std values using their column prefixes
        # This creates 1D NumPy arrays of length 100
        mean_cols = [f"mean_{p:02d}%" for p in range(100)]
        std_cols = [f"sd_{p:02d}%" for p in range(100)]
        
        mean_activation = row[mean_cols].to_numpy()
        std_activation = row[std_cols].to_numpy()

        # Generate a plot for the current sensor
        plot_title = f"Sensor {int(sensor_id)}"
        plot_emg_envelope(mean_activation, std_activation, title=plot_title)

if __name__ == "__main__":
    main()
