import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
from ast import literal_eval
import os
import numpy as np
from tqdm import tqdm

# Load the joseph_group style first
plt.style.use('./joseph_group.mplstyle')

# Override only specific settings we need to change
mpl.rcParams.update({
    'figure.figsize': (10, 6),  # Override for larger plots
    'figure.dpi': 300,  # Set DPI for output
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'text.usetex': True,
    'text.latex.preamble': '\n'.join([
        r'\usepackage[utf8]{inputenc}',
        r'\usepackage{sansmath}',
        r'\sansmath'
    ]),
    'figure.constrained_layout.use': False  # Disable constrained layout
})

# Get default colors from style
default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

# Create output directories
os.makedirs('density-matrices', exist_ok=True)
os.makedirs('radial-matrices', exist_ok=True)
os.makedirs('density-profiles', exist_ok=True)
os.makedirs('radial-profiles', exist_ok=True)

def create_matrix_plot(data, system_id, plot_type, title_suffix, is_radial=False):
    """Helper function to create matrix plots with consistent formatting."""
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    # Set extent for both plot types (0 to 1 for both z/Lz and r/R)
    extent = [0, 1, 2.5, 0.5]  # x from 0 to 1, sequence from 2.5 to 0.5
    
    # For radial plots, no need to flip the data since it's already ordered from r=0 to r=R
    im = ax.imshow(data, aspect='auto', cmap='viridis', extent=extent)
    
    # Add colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = fig.colorbar(im, cax=cax)
    
    # Set labels and title
    ax.set_xlabel('r/R' if is_radial else 'z/$L_z$')
    ax.set_ylabel('Sequence')
    ax.set_title(f'System {system_id}: {title_suffix}' + (' (Radial)' if is_radial else ''))
    cbar.set_label('Density [cm$^{-3}$]')
    
    # Set y-axis ticks to just 1 and 2
    ax.set_yticks([1, 2])
    
    # Set x-axis ticks (same for both plot types now)
    ax.set_xticks(np.linspace(0, 1, 5))
    
    # Format colorbar
    formatter = mpl.ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((0, 0))
    cbar.formatter = formatter
    cbar.update_ticks()
    
    # Save plot
    plt.savefig(
        f'{"radial" if is_radial else "density"}-matrices/System{system_id}_{plot_type}.png',
        bbox_inches='tight'
    )
    plt.close()

def create_line_profile(avg_data, std_data, system_id, is_radial=False):
    """Create line plot of density profile with error bands."""
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    # Create x coordinates
    x = np.linspace(0, 1, avg_data.shape[1])
    
    # Plot both sequences with error bands
    colors = [default_colors[1], default_colors[5]]  # Use colors 1 and 5
    for i, (sequence, color) in enumerate(zip(['Sequence 1', 'Sequence 2'], colors)):
        # Plot mean line
        ax.plot(x, avg_data[i], color=color, label=sequence, linewidth=2)
        # Add error bands
        ax.fill_between(x, 
                       avg_data[i] - std_data[i],
                       avg_data[i] + std_data[i],
                       color=color, alpha=0.2)
    
    # Set labels and title
    ax.set_xlabel('r/R' if is_radial else 'z/$L_z$')
    ax.set_ylabel('Density [cm$^{-3}$]')
    ax.set_title(f'System {system_id}')
    
    # Add legend
    ax.legend()
    
    # Set x-axis limits and ticks
    ax.set_xlim(0, 1)
    ax.set_xticks(np.linspace(0, 1, 5))
    
    # Format y-axis with scientific notation
    ax.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter(useMathText=True))
    ax.yaxis.get_major_formatter().set_scientific(True)
    ax.yaxis.get_major_formatter().set_powerlimits((0, 0))
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Save plot
    plt.savefig(
        f'{"radial" if is_radial else "density"}-profiles/System{system_id}_profile.png',
        bbox_inches='tight'
    )
    plt.close()

#skips = [8, 92, 130, 137, 142, 164]
skips = []

with open('dataset.txt', 'r') as file:
    lines = file.readlines()
    print(f"Total lines in dataset.txt: {len(lines)}")
    print("\nFirst few lines of dataset.txt:")
    for i in range(min(10, len(lines))):
        print(f"Line {i}: {lines[i].strip()}")
    
    for system_id in tqdm(range(0,200,1)):
        if system_id in skips:
            continue
        try:
            # Each system has 5 lines: header, Avg, Std, RadialAvg, RadialStd, blank line
            base_idx = system_id * 6  # Changed from 5 to 6 to account for blank line
            if base_idx + 5 >= len(lines):  # Check if we have enough lines
                print(f"Warning: Not enough data for System {system_id}")
                continue
            
            # Debug print for first system
            if system_id == 0:
                print(f"\nDebug - System {system_id} lines:")
                for i in range(6):
                    print(f"Line {base_idx + i}: {lines[base_idx + i].strip()}")
                
            # Read density and radial data from dataset.txt
            avg_line = lines[base_idx + 1].strip()
            std_line = lines[base_idx + 2].strip()
            radial_avg_line = lines[base_idx + 3].strip()
            radial_std_line = lines[base_idx + 4].strip()
            
            if system_id == 0:
                print("\nDebug - Split lines:")
                print(f"Avg line: {avg_line}")
                print(f"Std line: {std_line}")
                print(f"RadialAvg line: {radial_avg_line}")
                print(f"RadialStd line: {radial_std_line}")
            
            avg_mat = literal_eval(avg_line.split('Avg: ')[1])
            std_mat = literal_eval(std_line.split('Std: ')[1])
            radial_avg_mat = literal_eval(radial_avg_line.split('RadialAvg: ')[1])
            radial_std_mat = literal_eval(radial_std_line.split('RadialStd: ')[1])
            
            if system_id == 0:
                print("\nDebug - Array shapes:")
                print(f"avg_mat shape: {np.array(avg_mat).shape}")
                print(f"std_mat shape: {np.array(std_mat).shape}")
                print(f"radial_avg_mat shape: {np.array(radial_avg_mat).shape}")
                print(f"radial_std_mat shape: {np.array(radial_std_mat).shape}")
            
            # Convert to numpy arrays
            avg_mat = np.array(avg_mat)
            std_mat = np.array(std_mat)
            radial_avg_mat = np.array(radial_avg_mat)
            radial_std_mat = np.array(radial_std_mat)
            
            # Create density matrix plots
            for data, plot_type, title_suffix in [
                (avg_mat, 'avg', r'$\mu$'),
                (std_mat, 'std', r'$\sigma$')
            ]:
                create_matrix_plot(data, system_id, plot_type, title_suffix, is_radial=False)
            
            # Create radial matrix plots
            for data, plot_type, title_suffix in [
                (radial_avg_mat, 'avg', r'$\mu$'),
                (radial_std_mat, 'std', r'$\sigma$')
            ]:
                create_matrix_plot(data, system_id, plot_type, title_suffix, is_radial=True)
            
            # Create density and radial profile line plots
            create_line_profile(avg_mat, std_mat, system_id, is_radial=False)
            create_line_profile(radial_avg_mat, radial_std_mat, system_id, is_radial=True)
            
        except Exception as e:
            print(f"Error plotting system {system_id}: {str(e)}")
            if system_id == 0:
                import traceback
                traceback.print_exc()
            continue