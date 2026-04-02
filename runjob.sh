#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=10:00:00
#SBATCH --error=sl_%j.errFile
#SBATCH --output=sl_%j.outFile
#SBATCH --partition=normal
#SBATCH --collectors=off
#SBATCH --clustercockpit=off
#
# modules
#
module reset
export PATH=/scratch/pc2-mitarbeiter/xw/mybin/julia/v1.13.0-beta3/bin:$PATH
ncpus=4

lscpu

taskset --cpu-list 15,31,47,63 \
  julia --cpu-target=znver5 --optimize=3 --check-bounds=no \
        --math-mode=user --project=@. -t ${ncpus},0 -- matmul.jl
