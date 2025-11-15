import os
import numpy as np
import random
import subprocess
import sys
import matplotlib.pyplot as plt

class TwoProteinPhaseCoexistence:
    def __init__(self, id, seq1, seq2, temp=300):
        self.id = id
        self.seq1 = seq1
        self.seq2 = seq2
        self.temp = temp

    def _create_directory(self):
        base_dir = f'./simulations/System{self.id}'
        subdirs = ['single_chains', 'replicate', 'compress', 'dcs', 'plots']
        for subdir in subdirs:
            os.makedirs(f'{base_dir}/{subdir}', exist_ok=True)

    def _run_lammps_command_(self, input_file, out_dir, srun=True, lammps_executable='lmp_d9_double_aocc'):
        command = f"""{'srun ' if srun else ''}$HOME/.local/bin/{lammps_executable} -log {out_dir}/unfiltered_log.lammps -in {input_file} && grep -v WARNING {out_dir}/unfiltered_log.lammps > {out_dir}/{input_file.split('/')[-1].split('.')[0]}_log.lammps && rm {out_dir}/unfiltered_log.lammps"""
        print(command)
        result = subprocess.run(command, shell=True)
        return result

    def _create_sc_ne(self, seq, seq_id):
        masses = np.loadtxt('./mpipi/masses.txt', dtype=str)
        types = np.loadtxt('./mpipi/aa_types.txt', dtype=str)
        charges = np.loadtxt('./mpipi/aa_charges.txt', dtype=str)
        type_lookup = {aa: int(t) for aa, t in types}
        charge_lookup = {aa: float(c) for aa, c in charges}
        with open(f'./simulations/System{self.id}/single_chains/seq{seq_id}_ne.dat', 'w') as f:
            f.write('LAMMPS data file\n\n')
            n_atoms = len(seq)
            f.write(f'{n_atoms} atoms\n')
            f.write(f'{n_atoms-1} bonds\n')
            f.write(f'{n_atoms-2} angles\n\n')
            f.write(f'{len(masses)} atom types\n')
            f.write('2 bond types\n')
            f.write('1 angle types\n\n')
            box_size = n_atoms * 4
            f.write(f'{-box_size} {box_size} xlo xhi\n')
            f.write(f'{-box_size} {box_size} ylo yhi\n')
            f.write(f'{-box_size} {box_size} zlo zhi\n\n')
            f.write('Masses\n\n')
            for i, mass in enumerate(masses):
                f.write(f'{i+1} {float(mass[1])}\n')
            f.write('\nAtoms # full\n\n')
            for i, aa in enumerate(seq):
                atom_type = type_lookup[aa]  
                charge = charge_lookup[aa]
                f.write(f'{i+1} 1 {atom_type} {charge} {i*4} 0.0 0.0\n')
            f.write('\nBonds\n\n')
            for i in range(n_atoms-1):
                f.write(f'{i+1} 2 {i+1} {i+2}\n')
            f.write('\nAngles\n\n')
            for i in range(n_atoms-2):
                f.write(f'{i+1} 1 {i+1} {i+2} {i+3}\n')

    def _equilibrate_sc(self, seq_id):
        with open(f'./simulations/System{self.id}/single_chains/seq{seq_id}_equilibrate.dat', 'w') as f:
            f.write(f"""variable temperature equal {self.temp}
variable randomSeed equal {random.randint(1, 999999)}

units       real
dimension   3
boundary    p p p
atom_style  full

read_data       ./simulations/System{self.id}/single_chains/seq{seq_id}_ne.dat

include         ./mpipi/potentials.dat

velocity        all create ${{temperature}} ${{randomSeed}}

comm_style      tiled

timestep        10

neighbor  15.0 bin
neigh_modify    every 10 delay 0

fix fxnve all nve
fix fxlange all langevin ${{temperature}} ${{temperature}} 100000.0 ${{randomSeed}}

fix             fxbal all balance 1000 1.05 rcb

thermo          1000
thermo_style    custom step pe ecoul ke temp press density
thermo_modify   flush yes

run             100000

write_data      ./simulations/System{self.id}/single_chains/seq{seq_id}_eq.dat nocoeff
""")
        self._run_lammps_command_(f'./simulations/System{self.id}/single_chains/seq{seq_id}_equilibrate.dat', f'./simulations/System{self.id}/single_chains/')

    def _equilibrate_replicate(self):
        def parse_lammps_data(filepath):
            with open(filepath, 'r') as f:
                lines = f.readlines()
            sections = {}
            current_section = None
            for line in lines:
                if line.strip() == '':
                    continue
                elif any(header in line for header in ['Atoms', 'Velocities', 'Bonds', 'Angles', 'Masses']):
                    current_section = line.strip()
                    sections[current_section] = []
                elif current_section:
                    sections[current_section].append(line)
                else:
                    sections.setdefault('header', []).append(line)
            return sections

        def shift_atom_ids(atom_lines, id_offset, mol_offset):
            shifted = []
            for line in atom_lines:
                tokens = line.strip().split()
                tokens[0] = str(int(tokens[0]) + id_offset)
                tokens[1] = str(int(tokens[1]) + mol_offset)
                shifted.append(' '.join(tokens) + '\n')
            return shifted

        def shift_velocity_ids(vel_lines, id_offset):
            shifted = []
            for line in vel_lines:
                tokens = line.strip().split()
                tokens[0] = str(int(tokens[0]) + id_offset)
                shifted.append(' '.join(tokens) + '\n')
            return shifted

        def shift_bond_ids(bond_lines, id_offset, atom_offset):
            shifted = []
            for i, line in enumerate(bond_lines):
                tokens = line.strip().split()
                tokens[0] = str(int(tokens[0]) + id_offset)
                tokens[2] = str(int(tokens[2]) + atom_offset)
                tokens[3] = str(int(tokens[3]) + atom_offset)
                shifted.append(' '.join(tokens) + '\n')
            return shifted

        def shift_angle_ids(angle_lines, id_offset, atom_offset):
            shifted = []
            for i, line in enumerate(angle_lines):
                tokens = line.strip().split()
                tokens[0] = str(int(tokens[0]) + id_offset)
                tokens[2] = str(int(tokens[2]) + atom_offset)
                tokens[3] = str(int(tokens[3]) + atom_offset)
                tokens[4] = str(int(tokens[4]) + atom_offset)
                shifted.append(' '.join(tokens) + '\n')
            return shifted

        seq1 = parse_lammps_data(f'./simulations/System{self.id}/single_chains/seq1_eq.dat')
        seq2 = parse_lammps_data(f'./simulations/System{self.id}/single_chains/seq2_eq.dat')
        n_atoms1 = len(seq1['Atoms # full'])
        n_bonds1 = len(seq1['Bonds'])
        n_angles1 = len(seq1['Angles'])
        atoms2 = shift_atom_ids(seq2['Atoms # full'], id_offset=n_atoms1, mol_offset=1)
        velocities2 = shift_velocity_ids(seq2['Velocities'], id_offset=n_atoms1)
        bonds2 = shift_bond_ids(seq2['Bonds'], id_offset=n_bonds1, atom_offset=n_atoms1)
        angles2 = shift_angle_ids(seq2['Angles'], id_offset=n_angles1, atom_offset=n_atoms1)
        combined_data = []
        header = seq1['header']
        combined_data.extend(header)
        combined_data.append('\nMasses\n\n')
        combined_data.extend(seq1['Masses'])
        combined_data.append('\nAtoms # full\n\n')
        combined_data.extend(seq1['Atoms # full'])
        combined_data.extend(atoms2)
        combined_data.append('\nVelocities\n\n')
        combined_data.extend(seq1['Velocities'])
        combined_data.extend(velocities2)
        combined_data.append('\nBonds\n\n')
        combined_data.extend(seq1['Bonds'])
        combined_data.extend(bonds2)
        combined_data.append('\nAngles\n\n')
        combined_data.extend(seq1['Angles'])
        combined_data.extend(angles2)
        for i, line in enumerate(combined_data):
            if 'atoms' in line:
                combined_data[i] = f"{n_atoms1 * 2} atoms\n"
            elif 'bonds' in line:
                combined_data[i] = f"{n_bonds1 * 2} bonds\n"
            elif 'angles' in line:
                combined_data[i] = f"{n_angles1 * 2} angles\n"
        with open(f'./simulations/System{self.id}/combined.dat', 'w') as f:
            f.writelines(combined_data)
        replicate_input = f"""variable temperature equal {self.temp}
variable randomSeed equal {random.randint(1, 999999)}

units       real
dimension   3
boundary    p p p
atom_style  full

read_data ./simulations/System{self.id}/combined.dat
replicate 4 4 4

include         ./mpipi/potentials.dat

velocity        all create ${{temperature}} ${{randomSeed}}

comm_style      tiled

timestep        10

neighbor  15.0 bin
neigh_modify    every 10 delay 0

fix fxnve all nve
fix fxlange all langevin ${{temperature}} ${{temperature}} 100000.0 ${{randomSeed}}

fix             fxbal all balance 1000 1.05 rcb

thermo          1000
thermo_style    custom step pe ecoul ke temp press density
thermo_modify   flush yes

run             100000

write_data ./simulations/System{self.id}/replicate/replicate_eq.dat nocoeff
"""
        with open(f'./simulations/System{self.id}/replicate/replicate.dat', 'w') as f:
            f.write(replicate_input)
        self._run_lammps_command_(f'./simulations/System{self.id}/replicate/replicate.dat', f'./simulations/System{self.id}/replicate/')
    
    def _compress(self):
        with open(f'./simulations/System{self.id}/compress/compress.dat', 'w') as f:
            f.write(f"""variable temperature equal {self.temp}
variable randomSeed equal {random.randint(1, 999999)}
variable startingPressure equal 100
variable pressureDampingParam equal 10000

units       real
dimension   3
boundary    p p p
atom_style  full

read_data ./simulations/System{self.id}/replicate/replicate_eq.dat

include         ./mpipi/potentials.dat

velocity        all create ${{temperature}} ${{randomSeed}}

comm_style      tiled

timestep        10

neighbor  15.0 bin
neigh_modify    every 10 delay 0

thermo          1000
thermo_style    custom step pe ecoul ke temp press density
thermo_modify   flush yes

fix             2 all langevin ${{temperature}} ${{temperature}} 1000 ${{randomSeed}}
fix             3 all nve

fix             1 all press/berendsen iso ${{startingPressure}} ${{startingPressure}} ${{pressureDampingParam}}

run             25000

write_data ./simulations/System{self.id}/compress/compress_eq.dat nocoeff
""")
        self._run_lammps_command_(f'./simulations/System{self.id}/compress/compress.dat', f'./simulations/System{self.id}/compress/')

    def _dcs(self):
        def extend_z(input_file, output_file):
            with open(input_file, 'r') as f:
                lines = f.readlines()
            for i, line in enumerate(lines):
                if 'xlo xhi' in line:
                    xbounds = list(map(float, line.strip().split()[:2]))
                elif 'ylo yhi' in line:
                    ybounds = list(map(float, line.strip().split()[:2]))
                elif 'zlo zhi' in line:
                    zlo, zhi = map(float, line.strip().split()[:2])
                    zlen = zhi - zlo
                    new_zlo = zlo - 2*zlen
                    new_zhi = zhi + 2*zlen
                    new_zlen = new_zhi - new_zlo
                    lines[i] = f"{new_zlo} {new_zhi} zlo zhi\n"
                    break
            atom_start = None
            for i, line in enumerate(lines):
                if line.strip().startswith("Atoms"):
                    atom_start = i
                    break
            atom_data_start = atom_start + 2
            atom_data = []
            for i in range(atom_data_start, len(lines)):
                if lines[i].strip() == '' or lines[i][0].isalpha():
                    break
                atom_data.append(lines[i])
            new_atom_data = []
            for line in atom_data:
                parts = line.strip().split()
                atom_id = int(parts[0])
                mol_id = int(parts[1])
                atom_type = int(parts[2])
                charge = float(parts[3])
                x, y, z = map(float, parts[4:7])
                ix, iy, iz = map(int, parts[7:10])
                z_unwrapped = z + iz * zlen
                z_centered = z_unwrapped - (new_zlo + new_zlen / 2)
                iz_new = int((z_unwrapped - new_zlo) // new_zlen)
                z_wrapped = z_unwrapped - iz_new * new_zlen
                new_line = f"{atom_id} {mol_id} {atom_type} {charge:.6f} {x:.6f} {y:.6f} {z_wrapped:.6f} {ix} {iy} {iz_new}\n"
                new_atom_data.append(new_line)
            output_lines = lines[:atom_data_start] + new_atom_data + lines[atom_data_start+len(atom_data):]
            with open(output_file, 'w') as f:
                f.writelines(output_lines)
        extend_z(f'./simulations/System{self.id}/compress/compress_eq.dat', f'./simulations/System{self.id}/dcs/dcs_ne.dat')

        n_copies = 64
        seq1_pattern = ' '.join(str(i) for i in range(1, n_copies * 2, 2))
        seq2_pattern = ' '.join(str(i) for i in range(2, n_copies * 2 + 1, 2))

        with open(f'./simulations/System{self.id}/dcs/dcs.dat', 'w') as f:
            f.write(f"""variable temperature equal {self.temp}
variable randomSeed equal {random.randint(1, 999999)}

units       real
dimension   3
boundary    p p p
atom_style  full

read_data ./simulations/System{self.id}/dcs/dcs_ne.dat

include         ./mpipi/potentials.dat

velocity        all create ${{temperature}} ${{randomSeed}}

comm_style      tiled

timestep        10

neighbor  15.0 bin
neigh_modify    every 10 delay 0

fix fxnve all nve
fix fxlange all langevin ${{temperature}} ${{temperature}} 100000.0 ${{randomSeed}}

fix             fxbal all balance 1000 1.05 rcb

thermo          1000
thermo_style    custom step pe ecoul ke temp press density
thermo_modify   flush yes

run             6000000

variable seq1_pattern string "{seq1_pattern}"
variable seq2_pattern string "{seq2_pattern}"
group seq1_mols molecule ${{seq1_pattern}}
group seq2_mols molecule ${{seq2_pattern}}

compute myChunk all chunk/atom bin/1d z lower 0.02 units reduced

fix seq1_density seq1_mols ave/chunk 1000 40 50000 myChunk density/mass density/number file ./simulations/System{self.id}/dcs/seq1_densities_chunked.dat
fix seq2_density seq2_mols ave/chunk 1000 40 50000 myChunk density/mass density/number file ./simulations/System{self.id}/dcs/seq2_densities_chunked.dat
fix all_density all ave/chunk 1000 40 50000 myChunk density/mass density/number file ./simulations/System{self.id}/dcs/all_densities_chunked.dat

fix fixCOM all recenter INIT INIT INIT

dump            1 all custom 1000 ./simulations/System{self.id}/dcs/dcs_sample.lammpstrj id mol type q xu yu zu
dump_modify     1 sort id

run             2000000

write_data ./simulations/System{self.id}/dcs/dcs_eq.dat nocoeff
""")

        self._run_lammps_command_(f'./simulations/System{self.id}/dcs/dcs.dat', f'./simulations/System{self.id}/dcs/')

        def get_profile_data(fpath):
            profile_data = {}
            with open(fpath, 'r') as file:
                data = file.read()[3:]
                data = data.split('\n')
                header_idxs = [idx for idx in range(len(data)) if len(data[idx].split(' ')) == 3]
                try:
                    num_chunks = header_idxs[1] - header_idxs[0] - 1
                except:
                    num_chunks = len(data) - header_idxs[0] - 2
                profile_data = {chunk:[] for chunk in range(1, num_chunks+1)}
                for idx in header_idxs:
                    chunk_data = data[idx+1:idx+num_chunks+1]
                    chunk_data = [[float(y) for y in x.split(' ')[2:]] for x in chunk_data]
                    chunk_data = np.array(chunk_data)
                    for chunk_idx in range(num_chunks):
                        profile_data[chunk_idx+1].append(chunk_data[chunk_idx][4])
                    if header_idxs.index(idx) == 0:
                        coords = chunk_data[:,1]
            profile_data_mat = np.array(list(profile_data.values()))
            avg_profile = list({chunk: sum(profile_data[chunk])/profile_data_mat.shape[1] for chunk in range(1, profile_data_mat.shape[0]+1)}.values())
            
            return coords, profile_data_mat, avg_profile

        fig, ax = plt.subplots()

        coords, profile_data_mat, avg_profile = get_profile_data(f'./simulations/System{self.id}/dcs/seq1_densities_chunked.dat')
        for timestep_idx in range(profile_data_mat.shape[1]):
            timestep_profile = profile_data_mat[:,timestep_idx]
            ax.plot(coords, timestep_profile, color='blue', ls='-', alpha=0.1)
        ax.plot(coords, avg_profile, label='Seq1', color='blue', ls='-', alpha=1)

        coords, profile_data_mat, avg_profile = get_profile_data(f'./simulations/System{self.id}/dcs/seq2_densities_chunked.dat')
        for timestep_idx in range(profile_data_mat.shape[1]):
            timestep_profile = profile_data_mat[:,timestep_idx]
            ax.plot(coords, timestep_profile, color='red', ls='-', alpha=0.1)
        ax.plot(coords, avg_profile, label='Seq2', color='red', ls='-', alpha=1)

        coords, profile_data_mat, avg_profile = get_profile_data(f'./simulations/System{self.id}/dcs/all_densities_chunked.dat')
        for timestep_idx in range(profile_data_mat.shape[1]):
            timestep_profile = profile_data_mat[:,timestep_idx]
            ax.plot(coords, timestep_profile, color='black', ls='-', alpha=0.1)
        ax.plot(coords, avg_profile, label='Total', color='black', ls='--', alpha=0.4)

        ax.set_xticks([i/10 for i in range(0,11,2)])
        ax.set_xlabel('z / $L_{z}$')
        ax.set_ylabel(r'$\rho$ [$cm^{-3}$]')
        ax.legend(loc='best', fontsize=9)
        ax.set_title(f'Temperature = {self.temp}K')
        plt.subplots_adjust(left=0.2, bottom=0.15)

        fig.savefig(f'./simulations/System{self.id}/plots/density_profile.png')
        plt.close()

    def run(self):
        self._create_directory()
        self._create_sc_ne(self.seq1, 1)
        self._create_sc_ne(self.seq2, 2)
        self._equilibrate_sc(1)
        self._equilibrate_sc(2)
        self._equilibrate_replicate()
        self._compress()
        self._dcs()
    
if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: python workflow.py <idx> <seq1> <seq2>")
        sys.exit(1)
    idx = int(sys.argv[1])
    seq1 = sys.argv[2]
    seq2 = sys.argv[3]
    workflow = TwoProteinPhaseCoexistence(idx, seq1, seq2)
    workflow.run()