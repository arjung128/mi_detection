#!/bin/bash

# arrays
declare -a seeds=("30" "39")
# declare -a seeds=("39")
# declare -a channels=("i" "ii" "iii" "avr" "avl" "avf" "v1" "v2" "v3" "v4" "v5" "v6" "vx" "vy" "vz")
# declare -a best_pairs=("i" "vz" "ii" "v1" "ii" "vz" "avr" "v6" "avf" "vz" "v1" "v6" "v3" "vz" "v4" "vz" "v5" "v6" "v5" "vx" "v6" "vx" "v6" "vz" "vy" "vz")
declare -a best_pairs=("v6" "vz" "ii")
# declare -a best_pairs=("i" "vz" "ii" "v1")
declare -a runs=("1" "2" "3" "4" "5")
# declare -a runs=("1")

for seed in ${seeds[@]}; do
    for run in ${runs[@]}; do
        for ((i=0; i<${#best_pairs[@]}; i+=2)); do
            # echo $seed $run ${best_pairs[i]} ${best_pairs[i+1]} ${best_pairs[i+2]}
            python CNQ_model_split_by_patient_3_channel.py $seed $run ${best_pairs[i]} ${best_pairs[i+1]} ${best_pairs[i+2]}
        done
    done
done
