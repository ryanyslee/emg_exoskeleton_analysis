import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- 1. Parsing Function (Same as before) ---
def parse_emg_csv(file_path):
    df_raw = pd.read_csv(file_path, header=None)
    muscle_indices = df_raw[df_raw[1].str.contains("EMG Results", na=False)].index
    parsed_data = []
    
    # Column mapping: Condition -> {Type -> (Mag_Col, Delay_Col)}
    col_map = {
        'LG': {'OO': (2, 3), 'OW': (4, 5), 'TO': (6, 7), 'TW': (8, 9)},
        'RA': {'OO': (11, 12), 'OW': (13, 14), 'TO': (15, 16), 'TW': (17, 18)},
        'RD': {'OO': (20, 21), 'OW': (22, 23), 'TO': (24, 25), 'TW': (26, 27)}
    }

    for idx in muscle_indices:
        muscle_abbr = df_raw.iloc[idx, 1].split(" - ")[-1]
        start_row = idx + 4
        end_row = start_row + 8 
        subset = df_raw.iloc[start_row:end_row]
        
        for _, row in subset.iterrows():
            subject = row[1]
            for cond, types in col_map.items():
                for type_key, cols in types.items():
                    mag = row[cols[0]]
                    delay = row[cols[1]]
                    if pd.notnull(mag) and pd.notnull(delay):
                        parsed_data.append({
                            'Subject': subject,
                            'Muscle': muscle_abbr,
                            'Condition': cond,
                            'Type': type_key,
                            'Magnitude': float(mag),
                            'Delay': float(delay)
                        })
    return pd.DataFrame(parsed_data)

def plot_scatter(df):
    conditions = ['LG', 'RA', 'RD']
    
    muscles = ['TA', 'BF', 'GM', 'RF']
    muscle_titles = {
        'TA': 'Tibialis Anterior (TA)',
        'BF': 'Bicep Femoris (BF)',
        'GM': 'Gastrocnemius (GM)',
        'RF': 'Rectus Femoris (RF)'
    }
    
    colors = {'OO': 'darkgreen', 'OW': 'darkred', 'TO': 'darkblue', 'TW': 'darkorange'}
    base_labels = {'OO': 'Observed Optimal', 
                   'OW': 'Observed Worst', 
                   'TO': 'Theoretical Optimal', 
                   'TW': 'Theoretical Worst'
    }
    
    for cond in conditions:
        fig, axes = plt.subplots(2, 2, figsize=(18, 11))
        fig.suptitle(f'EMG Summary: Walking Condition - {cond}', fontsize=20, fontweight='bold')
        
        axes_flat = axes.flatten()
        
        # We need to collect legend handles from one of the plots to show at the bottom
        global_handles = []
        global_labels = []
        
        for i, muscle in enumerate(muscles):
            ax = axes_flat[i]
            data = df[(df['Condition'] == cond) & (df['Muscle'] == muscle)]
            
            # Grid
            ax.grid(True, linestyle='--', alpha=0.5, zorder=0)
        
            plot_handles = []
            plot_labels = []

            for type_key, color in colors.items():
                subset = data[data['Type'] == type_key]
                if subset.empty:
                    continue
                
                # 1. Individual Points (Small, light)
                scatter_individual = ax.scatter(subset['Delay'], subset['Magnitude'], 
                                     color=color, alpha=0.6, s=50, zorder=2, edgecolors='none')
                
                # 2. Statistics
                avg_delay = subset['Delay'].mean()
                avg_mag = subset['Magnitude'].mean()
                std_delay = subset['Delay'].std()
                std_mag = subset['Magnitude'].std()
                
                # 3. Error Bars (Standard Deviation)
                # fmt='none' means points aren't drawn by errorbar (we draw them manually below)
                ax.errorbar(avg_delay, avg_mag, 
                            xerr=std_delay, yerr=std_mag, 
                            fmt='none', ecolor=color, elinewidth=1, capsize=3, alpha=0.5, zorder=3)
                
                # 4. Average Point (Large, translucent, dark edge)
                scatter_h = ax.scatter(avg_delay, avg_mag, 
                           color=color, alpha=0.6, s=150, zorder=4)
                
                # Create label string with coordinates
                label_str = f"{type_key} ({avg_delay:.0f}ms, {avg_mag:.1f}%)"
                
                # Collect handles for this subplot's legend
                plot_handles.append(scatter_h)
                plot_labels.append(label_str)
                
                if i == 0:
                    global_handles.append(scatter_individual)
                    global_labels.append(base_labels[type_key])

            # Axis formatting
            ax.set_title(muscle_titles[muscle], fontsize=14, fontweight='bold')
            ax.set_xlabel('Delay (ms)', fontsize=11)
            ax.set_ylabel('Magnitude (%)', fontsize=11)
            ax.set_xticks([100, 150, 200, 250, 300])
            ax.set_yticks([10, 15, 20, 25, 30])
            ax.set_xlim(0, 320)
            ax.set_ylim(-2, 35)
            
            # This shows the specific coordinates for THIS muscle.
            ax.legend(plot_handles, plot_labels, 
                      loc='best', 
                      fontsize=8,
                      frameon=False, 
                      framealpha=0.2, 
                      labelspacing=1.2, 
                      handletextpad=0.5)

        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        
        fig.legend(global_handles, global_labels,
                   loc='lower center',
                   ncol=4,
                   fontsize=10,
                   bbox_to_anchor=(0.5, 0.03))

        plt.savefig(f'Scatter_{cond}.png', dpi=300)
        plt.show()

# --- Run ---
file_path = 'magnitude_delay_sweep_summary2.csv'
df_parsed = parse_emg_csv(file_path)
plot_scatter(df_parsed)