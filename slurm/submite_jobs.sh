#!/bin/bash

# forget offset = 510
for offset in $(seq 0 10 500); do
    sbatch slurm/run.slurm $offset
done