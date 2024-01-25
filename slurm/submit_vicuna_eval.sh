# #!/bin/bash


# train_offsets=(250 320 390)
# for offset in "${train_offsets[@]}"
# do
#     sbatch slurm/vicuna_eval.slurm $offset
# done

# for offset in $(seq 0 10 310);
# do
#     sbatch slurm/vicuna_eval.slurm $offset
#     # echo $offset
# done

for offset in $(seq 100 10 190);
do
    sbatch slurm/vicuna13b_eval.slurm $offset
    # echo $offset
done

for offset in $(seq 100 10 190);
do
    sbatch slurm/vicuna_eval.slurm $offset
    # echo $offset
done

for offset in $(seq 100 10 190);
do
    sbatch slurm/guanaco_eval.slurm $offset
    # echo $offset
done

for offset in $(seq 100 10 190);
do
    sbatch slurm/guanaco13b_eval.slurm $offset
    # echo $offset
done