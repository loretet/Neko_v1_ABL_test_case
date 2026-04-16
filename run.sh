#!/bin/bash -l
#SBATCH -J neko_test
#SBATCH -t 01:00:00
#SBATCH --ntasks-per-node=128
#SBATCH --nodes 1
#SBATCH -p main
#SBATCH -A naiss2025-1-5
#SBATCH --mail-type=BEGIN,END,FAIL

if [ ! -d logfiles ]; then
    mkdir logfiles
fi
if [ ! -d output ]; then
    mkdir output
fi

conda activate pysem

d="$(date +%F_%H-%M-%S)"
srun -u -n 128 ./neko abl_test.case > logfiles/log.run_${SLURM_JOB_ID}_${d} 2>&1
mv *0.* output/
srun  -n 1 python compare_to_ref.py > logfiles/log.compare_to_ref_${SLURM_JOB_ID}_${d} 2>&1
