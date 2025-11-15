import os
import subprocess

SBATCH_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=job_{idx}       
#SBATCH --nodes=1                
#SBATCH --ntasks=8               
#SBATCH --cpus-per-task=1        
#SBATCH --mem-per-cpu=4G         
#SBATCH --time=08:00:00          
#SBATCH --mail-user=mr7123@princeton.edu

module purge
module load gcc-toolset/14
module load aocc/5.0.0
module load aocl/aocc/5.0.0
module load openmpi/aocc-5.0.0/4.1.6

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK

python3 workflow.py {idx} "{seq1}" "{seq2}"
"""

os.makedirs("logs", exist_ok=True)
os.makedirs("slurm_jobs", exist_ok=True)

with open("2prot_sequences.txt", "r") as file:
    lines = [line.strip().split("\t") for line in file]

job_ids = []

for line in [lines[idx] for idx in [8, 92, 130, 131, 132, 133, 134, 135, 136, 137, 142, 164]]:
    idx, seq1, seq2 = int(line[0]), line[1], line[2]
    job_script_path = f"slurm_jobs/job_{idx}.sh"
    with open(job_script_path, "w") as job_file:
        job_file.write(SBATCH_TEMPLATE.format(idx=idx, seq1=seq1, seq2=seq2))
    result = subprocess.run(["sbatch", job_script_path], capture_output=True, text=True)
    if "Submitted batch job" in result.stdout:
        job_id = result.stdout.strip().split()[-1]
        job_ids.append(job_id)
        print(f"Submitted job {idx} with Job ID: {job_id}")
    else:
        print(f"Failed to submit job {idx}: {result.stderr}")
with open("submitted_jobs.txt", "w") as f:
    for job_id in job_ids:
        f.write(f"{job_id}\n")

print(f"Successfully submitted {len(job_ids)} jobs!")
