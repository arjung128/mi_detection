#!/bin/bash

# arrays
declare -a seeds=("30" "39")
declare -a channels=("i" "ii" "iii" "avr" "avl" "avf" "v1" "v2" "v3" "v4" "v5" "v6" "vx" "vy" "vz")
declare -a runs=("1" "2" "3" "4" "5")

for seed in ${seeds[@]}; do
    for run in ${runs[@]}; do
        for channel_one in ${channels[@]}; do
            for channel_two in ${channels[@]}; do
                if [ "$channel_one" = "$channel_two" ]; then
                    continue
                fi
                # echo $seed $run $channel_one $channel_two	
                python CNQ_model_split_by_patient.py $seed $run $channel_one $channel_two
            done
        done
    done
done
