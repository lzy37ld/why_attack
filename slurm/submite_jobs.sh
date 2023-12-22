# #!/bin/bash

# # forget offset = 510
# # vicuna forget offset = 510


# # 定义要跳过的offset值
# SKIP_OFFSETS=(150 220 230 350)

# for offset in $(seq 0 10 510); do
#     # 检查当前的offset是否应该跳过
#     if [[ " ${SKIP_OFFSETS[@]} " =~ " ${offset} " ]]; then
#         # 如果当前offset在跳过列表中，就跳过这次循环
#         echo "Skipping offset $offset"
#         continue
#     fi

#     # 如果不跳过，执行sbatch命令
#     sbatch slurm/run.slurm $offset
# done
#!/bin/bash


train_offsets=(0 10 20 30 40 50 60 70 80 90 100 110 120 130 140 160 170 180 190 200 210 240 250 260 270 280 290 300 310 320 330 340 360 370 380 390 400 420 430 440)

for offset in "${train_offsets[@]}"
do
    sbatch slurm/run.slurm $offset
done



