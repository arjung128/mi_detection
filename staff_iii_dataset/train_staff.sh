#!/bin/bash

# arrays
declare -a seeds=("39")
# declare -a channels=("i" "ii" "iii" "avr" "avl" "avf" "v1" "v2" "v3" "v4" "v5" "v6" "vx" "vy" "vz")
# declare -a runs=("1")

# FIND BEST CHANNEL
# for seed in ${seeds[@]}; do
#     for ((run=0; run<1; run+=1)); do
#         for ((channel=0; channel<9; channel+=1)); do
#             echo $seed $run $channel
#             python train_staff.py --channel=$channel --run=$run
#         done
#     done
# done


# FIND BEST SEED
for ((seed=0; run<10; seed+=1)); do
    for ((run=0; run<3; run+=1)); do
        echo $seed $run
        python train_staff.py --channel=1 --run=$run --seed=$seed
    done
done