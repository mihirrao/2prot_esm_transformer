import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import random
from scipy.signal import correlate
from scipy.stats import iqr
from matplotlib.ticker import ScalarFormatter
from tqdm import tqdm

# Set font sizes for plots
plt.rcParams.update({'font.size': 14})  # Set default font size for all text
plt.rcParams.update({'axes.titlesize': 16})  # Title font size
plt.rcParams.update({'axes.labelsize': 16})  # Axis label font size
plt.rcParams.update({'xtick.labelsize': 14})  # X-tick label font size
plt.rcParams.update({'ytick.labelsize': 14})  # Y-tick label font size

#! load the joseph_group style
########################################################################################
plt.style.use('./joseph_group.mplstyle') # load the joseph_group style
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = '\n'.join([r'\usepackage{sansmath}', r'\sansmath'])
default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color'] # obtain the colors into a list
order = [11, 5, 17, 15, 16, 10, 4, 9, 3, 14, 8, 2, 13, 7, 1, 12, 6, 18, 19, 0]  #order
order_dict = {index: color for index, color in enumerate(default_colors)}
ordered_colors = [order_dict[i] for i in order]
########################################################################################

# Create acf directory if it doesn't exist
os.makedirs('acf', exist_ok=True)

ids = [i for i in range(200)]  # Extended to all systems
exclude = []

def extract_pe_timeseries(filepath):
    """Extract potential energy timeseries from DCS log file"""
    relaxation_pe = []
    sampling_pe = []
    
    with open(filepath, 'r') as file:
        lines = file.readlines()
        for i, line in enumerate(lines):
            # Parse timestep data
            try:
                values = line.strip().split()
                if len(values) == 7:  # Only process lines with timestep data
                    step = int(values[0])
                    pe = float(values[1])
                    
                    # Separate relaxation and sampling phases
                    if step <= 6000000:  # Relaxation phase
                        relaxation_pe.append(pe)
                    elif step > 6000000:  # Sampling phase
                        sampling_pe.append(pe)
            except (ValueError, IndexError):
                continue
    
    return relaxation_pe, sampling_pe

# Store successful extractions
successful_systems = []
incomplete_systems = []
pe_data = {}

# Extract data from all systems
for id in ids:
    if id in exclude:
        continue
    
    filepath = f'../simulations/System{id}/dcs/dcs_log.lammps'
    try:
        relaxation_pe, sampling_pe = extract_pe_timeseries(filepath)
        total_steps = len(relaxation_pe) + len(sampling_pe)
        
        # Check if run is complete (8M steps = 8000 dumps at 1000 step intervals)
        if total_steps < 8000:
            print(f"  WARNING: Incomplete run System {id} - only {total_steps}/8000 timesteps")
            incomplete_systems.append(id)
        
        if len(relaxation_pe) > 0:  # Only store if we got data
            successful_systems.append(id)
            pe_data[id] = {
                'relaxation': relaxation_pe,
                'sampling': sampling_pe
            }
            
    except FileNotFoundError:
        print(f"Could not find log file for System {id}")

print("\nSummary:")
print(f"Total systems processed: {len(successful_systems)}")
print(f"Incomplete runs: {len(incomplete_systems)}")
if incomplete_systems:
    print("Incomplete system IDs:", incomplete_systems)

# Create gallery plot
if successful_systems:
    # Randomly sample 6 systems for 2x3 grid, excluding incomplete runs
    complete_systems = [s for s in successful_systems if s not in incomplete_systems]
    if complete_systems:
        plot_systems = random.sample(complete_systems, min(6, len(complete_systems)))
        plot_systems.sort()  # Sort systems for ordered display
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        for idx, system_id in enumerate(plot_systems):
            row = idx // 3
            col = idx % 3
            
            # Plot relaxation and sampling phases with different colors
            relaxation = pe_data[system_id]['relaxation']
            sampling = pe_data[system_id]['sampling']
            
            # Convert timesteps to ns (10fs * 1000 steps = 10ps per dump)
            relaxation_times = np.arange(len(relaxation)) * 0.01
            sampling_times = np.arange(len(sampling)) * 0.01 + 60  # Start at 60ns
            
            # Plot both phases
            axes[row, col].plot(relaxation_times, relaxation, default_colors[0], alpha=1, linewidth=2)
            axes[row, col].plot(sampling_times, sampling, color=default_colors[1], alpha=1, linewidth=2)  # Mid green color
            
            axes[row, col].set_title(f'System {system_id}', fontsize=16)
            
            # Set unique ticks for each subplot
            xticks = np.linspace(0, 80, 5)
            axes[row, col].set_xticks(xticks, xticks, fontsize=14)  # Example: 5 ticks from 0 to 80 ns
            yticks = np.linspace(min(relaxation + sampling), max(relaxation + sampling), 5)
            axes[row, col].set_yticks(yticks, yticks, fontsize=14)  # 5 ticks
            
            # Only set x-labels for the bottom row
            if row == 1:
                axes[row, col].set_xlabel('Time [ns]', fontsize=16)
            
            # Only set y-labels for the leftmost column
            if col == 0:
                axes[row, col].set_ylabel('Potential Energy [kcal/mol]', fontsize=16)
            
            axes[row, col].grid(True, alpha=0.3)
            
            # Apply scientific notation to y-axis
            formatter = ScalarFormatter(useMathText=True)
            formatter.set_scientific(True)
            formatter.set_powerlimits((0, 0))
            axes[row, col].yaxis.set_major_formatter(formatter)
            # Set the offset text (scientific notation) font size
            axes[row, col].yaxis.get_offset_text().set_fontsize(14)
        
        plt.tight_layout()
        plt.savefig('acf/pe_trajectories_gallery.png', dpi=300, bbox_inches='tight')
        plt.close()

def calculate_acf(x):
    """Calculate autocorrelation function"""
    x = np.array(x)
    x = x - np.mean(x)
    acf = correlate(x, x, mode='full') / len(x)
    acf = acf[len(x)-1:]
    return acf / acf[0]

# def calculate_iat(acf, dt=0.01):
#     """Calculate integrated autocorrelation time"""
#     # Integrate until first crossing of 0 or e^-1
#     cutoff = np.exp(-1)
#     for i, value in enumerate(acf):
#         if value <= cutoff or value <= 0:
#             break
    
#     iat = 2 * np.sum(acf[:i]) * dt
#     return iat

def calculate_iat(acf, dt=0.01, c=5):
    """
    Canonical integrated autocorrelation time (IAT) with automatic windowing.
    Uses: τ_int = 0.5 + ∑ ρ(t), with window where t <= c * τ_int
    """
    tau_int = 0.5
    for t in range(1, len(acf)):
        tau_int += acf[t]
        if t > c * tau_int:
            break
    return tau_int * dt

def block_average_analysis(data, input_block_sizes):
    """
    Perform block averaging analysis on the data.
    Returns block sizes and corresponding standard errors with their uncertainties.
    """
    n = len(data)
    block_sizes = []
    std_errs = []
    std_err_errs = []  # Uncertainties in the standard errors
    
    print(block_sizes)
    for block_size in tqdm(input_block_sizes):
        print(block_size)
        if n // block_size < 2:  # Need at least 2 blocks
            break
            
        # Create blocks and calculate their means
        n_blocks = n // block_size
        blocks = np.array_split(data[:n_blocks * block_size], n_blocks)
        block_means = np.array([np.mean(block) for block in blocks])
        
        # Calculate standard error and its uncertainty
        std_err = np.std(block_means) / np.sqrt(n_blocks)
        # Error in standard error (from chi-square distribution)
        std_err_err = std_err / np.sqrt(2 * (n_blocks - 1))
        
        block_sizes.append(block_size)
        std_errs.append(std_err)
        std_err_errs.append(std_err_err)
    
    return np.array(block_sizes), np.array(std_errs), np.array(std_err_errs)

# Calculate IAT as function of start time for all valid systems
n_samples = 100

# Create array of start times (in timesteps)
start_times = np.linspace(0, 8000, n_samples, dtype=int)  # Leave room for window at end
start_times_ns = start_times * 0.01  # Convert to ns

iat_vs_time = {system_id: [] for system_id in complete_systems}

for system_id in complete_systems:
    # Combine relaxation and sampling phases
    full_trajectory = pe_data[system_id]['relaxation'] + pe_data[system_id]['sampling']
    
    # Calculate IAT for each start time
    for start_idx in start_times:
        window = full_trajectory[start_idx:]
        acf = calculate_acf(window)
        iat = calculate_iat(acf)
        iat_vs_time[system_id].append(iat)

# Calculate statistics across systems for each start time
medians = []
q1s = []
q3s = []

for i in range(n_samples):
    iats_at_time = [iat_vs_time[system_id][i] for system_id in complete_systems]
    medians.append(np.median(iats_at_time))
    q1s.append(np.percentile(iats_at_time, 25))
    q3s.append(np.percentile(iats_at_time, 75))

# Create combined figure for IAT and block averaging analysis
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

# Plot IAT vs time on first subplot
ax1.errorbar(start_times_ns, medians, 
            yerr=[np.array(medians)-np.array(q1s), np.array(q3s)-np.array(medians)],
            fmt='o', color=default_colors[0], capsize=5, alpha=1, markersize=4)

ax1.axvline(x=60, color=default_colors[0], linestyle='--', alpha=0.5)  # Mark relaxation/sampling transition
ax1.set_xlabel('Start Time [ns]', fontsize=16)
ax1.set_ylabel('Integrated Autocorrelation Time [ns]', fontsize=16)
ax1.tick_params(labelsize=14)
ax1.grid(True, alpha=0.3)

print("Computing average ACF across systems...")

# Max lag in timesteps to compute for all systems (e.g., 1000 = 10 ns)
max_lag = 1000

# Store truncated ACFs per system
acf_matrix = []

for system_id in complete_systems:
    full_trajectory = pe_data[system_id]['relaxation'] + pe_data[system_id]['sampling']
    acf = calculate_acf(full_trajectory)[:max_lag]
    acf_matrix.append(acf)

# Convert to array: shape (num_systems, max_lag)
acf_matrix = np.array(acf_matrix)
lags_ns = np.arange(max_lag) * 0.01  # Convert to ns

# Compute statistics
acf_median = np.median(acf_matrix, axis=0)
acf_q1 = np.percentile(acf_matrix, 25, axis=0)
acf_q3 = np.percentile(acf_matrix, 75, axis=0)
acf_lower = acf_median - acf_q1
acf_upper = acf_q3 - acf_median

# Plot median ACF with IQR error bars
plt.figure(figsize=(10, 6))
plt.errorbar(lags_ns, acf_median, yerr=[acf_lower, acf_upper],
             fmt='o', color=default_colors[0], capsize=3, markersize=3, alpha=0.9)

plt.axhline(y=0, linestyle='--', color='gray', alpha=0.5)
plt.xlabel('Lag Time [ns]', fontsize=16)
plt.ylabel('Normalized ACF', fontsize=16)
plt.title('Average ACF of Potential Energy Across Systems', fontsize=18)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('acf/avg_acf.png', dpi=300)
plt.close()

print('Block averaging analysis...')
# Perform block averaging analysis on the sampling phase data
all_sampling_data = []
for system_id in complete_systems:
    sampling_data = pe_data[system_id]['sampling']
    all_sampling_data.extend(sampling_data)

# Convert input block sizes from ns to number of dumps
# 1 ns = 100 dumps (since 1000 steps × 10fs = 10ps per dump)
input_block_sizes_ns = [i for i in range(1, 70, 2)]  # in nanoseconds
input_block_sizes_dumps = [int(size * 100) for size in input_block_sizes_ns]  # convert to number of dumps

block_sizes_dumps, std_errs, std_err_errs = block_average_analysis(all_sampling_data, input_block_sizes_dumps)

# Convert block sizes back to ns for plotting
block_sizes_ns = block_sizes_dumps / 100  # convert dumps to ns

# Plot block averaging analysis on second subplot
ax2.errorbar(block_sizes_ns, std_errs, yerr=std_err_errs,
            fmt='o', color=default_colors[8], capsize=5, alpha=1, markersize=4)
ax2.set_xlabel('Block Size [ns]', fontsize=16)
ax2.set_ylabel('Standard Error [kcal/mol]', fontsize=16)
ax2.tick_params(labelsize=14)
ax2.grid(True, alpha=0.3)

# Set regular linear scale with appropriate ticks
max_ns = max(block_sizes_ns)
tick_spacing = 10 if max_ns > 50 else 5  # Use 10 ns spacing for large ranges, 5 ns for smaller
xticks = np.arange(0, max_ns + tick_spacing, tick_spacing)
ax2.set_xticks(xticks)
ax2.set_xlim(0, max_ns * 1.05)  # Add 5% padding to the right

# Apply scientific notation to y-axis for both plots
for ax in [ax1, ax2]:
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((0, 0))
    ax.yaxis.set_major_formatter(formatter)
    ax.yaxis.get_offset_text().set_fontsize(14)

plt.tight_layout()
plt.savefig('acf/combined_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

# Also save the original IAT plot separately for backward compatibility
plt.figure(figsize=(10, 6))
plt.errorbar(start_times_ns, medians, 
            yerr=[np.array(medians)-np.array(q1s), np.array(q3s)-np.array(medians)],
            fmt='o', color=default_colors[0], capsize=5, alpha=1, markersize=4)

plt.axvline(x=60, color=default_colors[1], linestyle='--', alpha=1, lw=2)  # Mark relaxation/sampling transition
plt.xlabel('Start Time [ns]', fontsize=16)
plt.ylabel('Integrated Autocorrelation Time [ns]', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('acf/iat_vs_time.png', dpi=300, bbox_inches='tight')
plt.close()
