#!/bin/bash
#SBATCH --job-name=prompter_validation               # name
#SBATCH --nodes=1                    # nodes
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=2           # number of cores per tasks
#SBATCH --gres=gpu:1                 # number of gpus
#SBATCH --time 24:00:00               # maximum execution time (HH:MM:SS)
#SBATCH --output=%x-%j.out           # output file name
#SBATCH --partition=preemptible

# sbatch slurm/run.slurm
srun --jobid $SLURM_JOBID bash -c 'bash run_cli_test.sh'
