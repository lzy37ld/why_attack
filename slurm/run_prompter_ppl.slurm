#!/bin/bash
#SBATCH --job-name=ppl_attack_prompter               # name
#SBATCH --nodes=1                                # nodes
#SBATCH --ntasks-per-node=1                      # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=10                       # number of cores per tasks
#SBATCH --gres=gpu:4                             # number of gpus
#SBATCH --time 24:00:00                          # maximum execution time (HH:MM:SS)
#SBATCH --output=%x-%j.out                       # output file name
#SBATCH --mail-type=ALL                          # send email for all notifications
#SBATCH --mail-user=lzy37ld@gmail.com       # your email
#SBATCH --mem=300G
###SBATCH --partition=preemptible
#SBATCH --exclude=a0018,a0017,a0006

# sbatch slurm/run.slurm
srun --jobid $SLURM_JOBID bash -c 'bash tmp_prompter_ascend_ppl.sh'