#!/bin/bash 
#PBS -l nodes=nano1:ppn=1:gpus=1,walltime=96:00:00
#PBS -N CNQ_model
#PBS -o log_CNQ_model.txt
#PBS -e err_CNQ_model.txt 

cd $PBS_O_WORKDIR
source activate python2.7

python CNQ_model.py v6 vz