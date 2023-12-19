#!/bin/bash
#SBATCH --job-name=attack_evaluate               # name
#SBATCH --nodes=1                    # nodes
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=10           # number of cores per tasks
#SBATCH --gres=gpu:1                 # number of gpus
#SBATCH --time 7:00:00               # maximum execution time (HH:MM:SS)
#SBATCH --output=%x-%j.out           # output file name

export offset=$1

# sbatch slurm/run.slurm
srun --jobid $SLURM_JOBID bash -c 'bash tmp.sh $offset'
