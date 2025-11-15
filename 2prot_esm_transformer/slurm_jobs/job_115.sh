#!/bin/bash
#SBATCH --job-name=job_115       # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=8               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G is default)
#SBATCH --time=08:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-user=mr7123@princeton.edu

module purge
module load gcc-toolset/14
module load aocc/5.0.0
module load aocl/aocc/5.0.0
module load openmpi/aocc-5.0.0/4.1.6

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK

python3 workflow.py 115 "YHLFLWEQGMAIPGEKHHQTCALAARVDACFEQATRALLRFGLPGWRFEK" "DWIWCPGCCGFVFCEVEILMQFWGVDPAGLFRDVAPESIIYDGYEICLDS"
