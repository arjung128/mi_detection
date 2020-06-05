#!/bin/bash

# arrays
# declare -a seeds=("5" "10" "11" "13" "23" "30" "36" "39" "41" "46" "49")
declare -a seeds=("49")
# declare -a seeds=("39")
# declare -a channels=("i" "ii" "iii" "avr" "avl" "avf" "v1" "v2" "v3" "v4" "v5" "v6" "vx" "vy" "vz")
# declare -a best_pairs=("i" "vz" "ii" "v1" "ii" "vz" "avr" "v6" "avf" "vz" "v1" "v6" "v3" "vz" "v4" "vz" "v5" "v6" "v5" "vx" "v6" "vx" "v6" "vz" "vy" "vz")
declare -a best_pairs=("v6" "vz" "ii")
# declare -a best_pairs=("i" "vz" "ii" "v1")
declare -a runs=("1" "2" "3" "4" "5" "6" "7" "8" "9" "10")
# declare -a runs=("1")

for seed in ${seeds[@]}; do
    for run in ${runs[@]}; do
        python liu_et_al.py $seed $run
    done
done
