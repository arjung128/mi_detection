#!/bin/bash

# make directories
for ((seed=1; seed<7; seed+=1)); do
    mkdir runs_staff_v1/seed$seed
done

# start jobs
for ((channel=0; channel<9; channel+=1)); do
    
    # gpu0
    for seed in 1 2; do
        screen -d -m -S chan${channel}_seed$seed bash -c "source /home/arjung2/project/anaconda3/etc/profile.d/conda.sh && conda activate py27 && echo \"$channel\" \"$seed\" && python train_staff.py --channel=\"$channel\" --seed=\"$seed\" --gpu=2"
    done
 
    # gpu1
    seed=3
    screen -d -m -S chan${channel}_seed$seed bash -c "source /home/arjung2/project/anaconda3/etc/profile.d/conda.sh && conda activate py27 && echo \"$channel\" \"$seed\" && python train_staff.py --channel=\"$channel\" --seed=\"$seed\" --gpu=0"

    # gpu2
    for seed in 4 5 6; do
        screen -d -m -S chan${channel}_seed$seed bash -c "source /home/arjung2/project/anaconda3/etc/profile.d/conda.sh && conda activate py27 && echo \"$channel\" \"$seed\" && python train_staff.py --channel=\"$channel\" --seed=\"$seed\" --gpu=1"
    done

done

