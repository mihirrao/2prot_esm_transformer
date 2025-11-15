from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

num_chunks = 50
num_sampling_steps = 2000000
dump_freq = 50000
num_dumps = num_sampling_steps // dump_freq

def load_timeseries(fpath):
    out = []
    with open(fpath, 'r') as file:
        lines = file.readlines()
        lines = lines[3:]
        for dump_idx in range(num_dumps):
            start = dump_idx * (num_chunks+1) + 1
            end = (dump_idx+1) * (num_chunks+1)
            sub = lines[start:end]
            sub_data = [float(x.split()[4]) for x in sub]
            out.append(sub_data)
    file.close()
    out = np.array(out)
    return out

def compute_shift_timeseries(all_ts):
    shifts = []
    for t in range(all_ts.shape[0]):
        row = all_ts[t,:]
        COM = np.sum(row * np.arange(len(row))) / np.sum(row)
        mode = np.argmax(row)
        center = 1*mode + 0*COM
        shift = int(round(len(row)/2 - center))
        shifts.append(shift)
    return np.array(shifts)

def apply_shifts(ts, shifts):
    for t in range(ts.shape[0]):
        ts[t,:] = np.roll(ts[t,:], shifts[t])
    return ts

def calculate_radial_profile(matrix):
    n_cols = matrix.shape[1]
    center = n_cols // 2
    
    if n_cols % 2 == 0:
        left_half = matrix[:, :center]
        right_half = matrix[:, center:][:, ::-1]
        return ((left_half + right_half) / 2)[:, ::-1]
    else:
        left_half = matrix[:, :center]
        right_half = matrix[:, (center+1):][:, ::-1]
        center_col = matrix[:, center].reshape(-1, 1)
        averaged_sides = (left_half + right_half) / 2
        return np.hstack([center_col, averaged_sides[:, ::-1]])

def calculate_radial_profiles(seq1_ts, seq2_ts):
    n_timesteps = seq1_ts.shape[0]
    n_cols = seq1_ts.shape[1]
    n_output_cols = (n_cols + 1) // 2
    
    radial_seq1_ts = np.zeros((n_timesteps, n_output_cols))
    radial_seq2_ts = np.zeros((n_timesteps, n_output_cols))
    
    for t in range(n_timesteps):
        timestep_data = np.stack([seq1_ts[t], seq2_ts[t]], axis=0)
        radial_data = calculate_radial_profile(timestep_data)
        radial_seq1_ts[t] = radial_data[0]
        radial_seq2_ts[t] = radial_data[1]
    
    return radial_seq1_ts, radial_seq2_ts

with open('dataset.txt', 'w') as file:
    for system_id in tqdm(range(0,200,1)):
        try:
            all_ts = load_timeseries(f'../simulations/System{system_id}/dcs/all_densities_chunked.dat')
            seq1_ts = load_timeseries(f'../simulations/System{system_id}/dcs/seq1_densities_chunked.dat')
            seq2_ts = load_timeseries(f'../simulations/System{system_id}/dcs/seq2_densities_chunked.dat')
            shifts = compute_shift_timeseries(all_ts)
            seq1_ts = apply_shifts(seq1_ts, shifts)
            seq2_ts = apply_shifts(seq2_ts, shifts)
            
            avg_seq1 = np.mean(seq1_ts, axis=0)
            avg_seq2 = np.mean(seq2_ts, axis=0)
            avg_mat = np.stack([avg_seq1, avg_seq2], axis=0)
            std_seq1 = np.std(seq1_ts, axis=0)
            std_seq2 = np.std(seq2_ts, axis=0)
            std_mat = np.stack([std_seq1, std_seq2], axis=0)
            
            radial_seq1_ts, radial_seq2_ts = calculate_radial_profiles(seq1_ts, seq2_ts)
            radial_avg_seq1 = np.mean(radial_seq1_ts, axis=0)
            radial_avg_seq2 = np.mean(radial_seq2_ts, axis=0)
            radial_avg_mat = np.stack([radial_avg_seq1, radial_avg_seq2], axis=0)
            radial_std_seq1 = np.std(radial_seq1_ts, axis=0)
            radial_std_seq2 = np.std(radial_seq2_ts, axis=0)
            radial_std_mat = np.stack([radial_std_seq1, radial_std_seq2], axis=0)
            
            file.write(f'System {system_id}\n')
            file.write(f'Avg: {avg_mat.tolist()}\n')
            file.write(f'Std: {std_mat.tolist()}\n')
            file.write(f'RadialAvg: {radial_avg_mat.tolist()}\n')
            file.write(f'RadialStd: {radial_std_mat.tolist()}\n\n')
        except Exception as e:
            print(f'Error loading System{system_id}: {e}')
file.close()