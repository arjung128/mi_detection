#!/bin/bash 
#PBS -l nodes=nano1:ppn=1:gpus=1,walltime=96:00:00
#PBS -N CNQ_model
#PBS -o log_CNQ_model.txt
#PBS -e err_CNQ_model.txt 

# cd $PBS_O_WORKDIR
# source activate python2.7

python CNQ_model_split_by_patient.py 1
python CNQ_model_split_by_patient.py 2
python CNQ_model_split_by_patient.py 3
python CNQ_model_split_by_patient.py 4
python CNQ_model_split_by_patient.py 5
python CNQ_model_split_by_patient.py 6
python CNQ_model_split_by_patient.py 7
python CNQ_model_split_by_patient.py 8
python CNQ_model_split_by_patient.py 9
python CNQ_model_split_by_patient.py 10
python CNQ_model_split_by_patient.py 11
python CNQ_model_split_by_patient.py 12
python CNQ_model_split_by_patient.py 13
python CNQ_model_split_by_patient.py 14
python CNQ_model_split_by_patient.py 15
python CNQ_model_split_by_patient.py 16
python CNQ_model_split_by_patient.py 17
python CNQ_model_split_by_patient.py 18
python CNQ_model_split_by_patient.py 19
python CNQ_model_split_by_patient.py 20
python CNQ_model_split_by_patient.py 21
python CNQ_model_split_by_patient.py 22
python CNQ_model_split_by_patient.py 23
python CNQ_model_split_by_patient.py 24
python CNQ_model_split_by_patient.py 25
python CNQ_model_split_by_patient.py 26
python CNQ_model_split_by_patient.py 27
python CNQ_model_split_by_patient.py 28
python CNQ_model_split_by_patient.py 29
python CNQ_model_split_by_patient.py 30
python CNQ_model_split_by_patient.py 31
python CNQ_model_split_by_patient.py 32
python CNQ_model_split_by_patient.py 33
python CNQ_model_split_by_patient.py 34
python CNQ_model_split_by_patient.py 35
python CNQ_model_split_by_patient.py 36
python CNQ_model_split_by_patient.py 37
python CNQ_model_split_by_patient.py 38
python CNQ_model_split_by_patient.py 39
python CNQ_model_split_by_patient.py 40
python CNQ_model_split_by_patient.py 41
python CNQ_model_split_by_patient.py 42
python CNQ_model_split_by_patient.py 43
python CNQ_model_split_by_patient.py 44
python CNQ_model_split_by_patient.py 45
python CNQ_model_split_by_patient.py 46
python CNQ_model_split_by_patient.py 47
python CNQ_model_split_by_patient.py 48
python CNQ_model_split_by_patient.py 49
python CNQ_model_split_by_patient.py 50
