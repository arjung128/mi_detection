#!/bin/bash 
#PBS -l nodes=nano1:ppn=1:gpus=1,walltime=96:00:00
#PBS -N CNQ_model
#PBS -o log_CNQ_model.txt
#PBS -e err_CNQ_model.txt 

# cd $PBS_O_WORKDIR
# source activate python2.7

# 5, 10, 11, 13, 23, 30, 36, 39, 41, 46, 49

python CNQ_model_split_by_patient.py 5 0
python CNQ_model_split_by_patient.py 5 1
python CNQ_model_split_by_patient.py 5 2
python CNQ_model_split_by_patient.py 5 3
python CNQ_model_split_by_patient.py 5 4
python CNQ_model_split_by_patient.py 10 0
python CNQ_model_split_by_patient.py 10 1
python CNQ_model_split_by_patient.py 10 2
python CNQ_model_split_by_patient.py 10 3
python CNQ_model_split_by_patient.py 10 4
python CNQ_model_split_by_patient.py 11 0
python CNQ_model_split_by_patient.py 11 1
python CNQ_model_split_by_patient.py 11 2
python CNQ_model_split_by_patient.py 11 3
python CNQ_model_split_by_patient.py 11 4
python CNQ_model_split_by_patient.py 13 0
python CNQ_model_split_by_patient.py 13 1
python CNQ_model_split_by_patient.py 13 2
python CNQ_model_split_by_patient.py 13 3
python CNQ_model_split_by_patient.py 13 4
python CNQ_model_split_by_patient.py 23 0
python CNQ_model_split_by_patient.py 23 1
python CNQ_model_split_by_patient.py 23 2
python CNQ_model_split_by_patient.py 23 3
python CNQ_model_split_by_patient.py 23 4
python CNQ_model_split_by_patient.py 30 0
python CNQ_model_split_by_patient.py 30 1
python CNQ_model_split_by_patient.py 30 2
python CNQ_model_split_by_patient.py 30 3
python CNQ_model_split_by_patient.py 30 4
python CNQ_model_split_by_patient.py 36 0
python CNQ_model_split_by_patient.py 36 1
python CNQ_model_split_by_patient.py 36 2
python CNQ_model_split_by_patient.py 36 3
python CNQ_model_split_by_patient.py 36 4
python CNQ_model_split_by_patient.py 39 0
python CNQ_model_split_by_patient.py 39 1
python CNQ_model_split_by_patient.py 39 2
python CNQ_model_split_by_patient.py 39 3
python CNQ_model_split_by_patient.py 39 4
python CNQ_model_split_by_patient.py 41 0
python CNQ_model_split_by_patient.py 41 1
python CNQ_model_split_by_patient.py 41 2
python CNQ_model_split_by_patient.py 41 3
python CNQ_model_split_by_patient.py 41 4
python CNQ_model_split_by_patient.py 46 0
python CNQ_model_split_by_patient.py 46 1
python CNQ_model_split_by_patient.py 46 2
python CNQ_model_split_by_patient.py 46 3
python CNQ_model_split_by_patient.py 46 4
python CNQ_model_split_by_patient.py 49 0
python CNQ_model_split_by_patient.py 49 1
python CNQ_model_split_by_patient.py 49 2
python CNQ_model_split_by_patient.py 49 3
python CNQ_model_split_by_patient.py 49 4




