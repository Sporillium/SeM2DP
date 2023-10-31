#!/bin/bash

while getopts e: flag
do
    case "${flag}" in
        e) experiment=${OPTARG};;
    esac
done

logfile=$(basename "$0")_run.log

programfile=/home/march/devel/SeM2DP

# Implementing Fallthrough switch statement
case "$experiment" in 
    0)
        > $logfile # Clear the previous run log before executing
    ;&
    1)
    # Experiment 1: Parameter Tuning for Circles
        echo "$(date) Starting Experiment: Parameter Tuning Circles Normal" >> $logfile 
        python $programfile/parameter_tuning.py -e display -n 7 -m normal -f circle_params.txt -s results/circle_tune_norm >> $logfile 
        echo "$(date) Experiment Complete" >> $logfile  
    ;&
    2)
    # Experiment 2: Parameter Tuning for Bins
        echo "$(date) Starting Experiment: Parameter Tuning Bins Normal" >> $logfile 
        python $programfile/parameter_tuning.py -e display -n 7 -m normal -f bin_params.txt -s results/bin_tune_norm >> $logfile 
        echo "$(date) Experiment Complete" >> $logfile  
    ;&
    3)
    # Experiment 3: Parameter Tuning for Circles, Superclasses
        echo "$(date) Starting Experiment: Parameter Tuning Circles Super" >> $logfile 
        python $programfile/parameter_tuning.py -e display -n 7 -m super -f circle_params.txt -s results/circle_tune_super >> $logfile 
        echo "$(date) Experiment Complete" >> $logfile  
    ;&
    4)
    # Experiment 2: Parameter Tuning for Bins, Superclasses
        echo "$(date) Starting Experiment: Parameter Tuning Bins Super" >> $logfile 
        python $programfile/parameter_tuning.py -e display -n 7 -m super -f bin_params.txt -s results/bin_tune_super >> $logfile 
        echo "$(date) Experiment Complete" >> $logfile  
    ;;
    *)
    ;;
esac