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
    # Experiment 1: Compare Normal and Super Seq 07
        echo "$(date) Starting Experiment: Comparing Normal and Super SeM2DP Seq07" >> $logfile 
        python $programfile/parameter_tuning.py -e display -n 7 -m compare -f comp_params.txt -s results/comp_07 >> $logfile 
        echo "$(date) Experiment Complete" >> $logfile  
    ;&
    2)
    # Experiment 2: Parameter Tuning for Bins
        echo "$(date) Starting Experiment: Comparing Normal and Super SeM2DP Seq06" >> $logfile 
        python $programfile/parameter_tuning.py -e display -n 6 -m compare -f comp_params.txt -s results/comp_06 >> $logfile 
        echo "$(date) Experiment Complete" >> $logfile    
    ;&
    3)
    # Experiment 3: Parameter Tuning for Circles, Superclasses
        echo "$(date) Starting Experiment: Comparing Normal and Super SeM2DP Seq05" >> $logfile 
        python $programfile/parameter_tuning.py -e display -n 5 -m compare -f comp_params.txt -s results/comp_05 >> $logfile 
        echo "$(date) Experiment Complete" >> $logfile    
    ;&
    4)
    # Experiment 2: Parameter Tuning for Bins, Superclasses
        echo "$(date) Starting Experiment: Comparing Normal and Super SeM2DP Seq00" >> $logfile 
        python $programfile/parameter_tuning.py -e display -n 0 -m compare -f comp_params.txt -s results/comp_00 >> $logfile 
        echo "$(date) Experiment Complete" >> $logfile    
    ;;
    *)
    ;;
esac