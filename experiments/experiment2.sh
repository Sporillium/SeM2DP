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
    # Experiment 1: Compare Normal and Super, Seq 06
        echo "$(date) Starting Experiment 1" >> $logfile
        python $programfile/parameter_tuning.py -e display -n 6 -m all -f inputs/all_comp_params.txt -s results/comparisons/all_seq06 >> $logfile
        echo "$(date) Experiment 1 Complete" >> $logfile
    ;&
    2)
    # Experiment 2: Compare Normal and Super, Seq 05
        echo "$(date) Starting Experiment 2" >> $logfile
        python $programfile/parameter_tuning.py -e display -n 5 -m all -f inputs/all_comp_params.txt -s results/comparisons/all_seq05 >> $logfile 
        echo "$(date) Experiment 2 Complete" >> $logfile 
    ;&
    3)
    # Experiment 3: Compare Normal and Super, Seq 00
        echo "$(date) Starting Experiment 3" >> $logfile 
        python $programfile/parameter_tuning.py -e display -n 0 -m all -f inputs/all_comp_params.txt -s results/comparisons/all_seq00 >> $logfile 
        echo "$(date) Experiment 3 Complete" >> $logfile 
    ;&
    4)
    # Experiment 4: Compare Normal and Super, Seq 07
        echo "$(date) Starting Experiment 4" >> $logfile 
        python $programfile/parameter_tuning.py -e display -n 7 -m all -f inputs/all_comp_params.txt -s results/comparisons/all_seq07 >> $logfile 
        echo "$(date) Experiment 4 Complete" >> $logfile  
    ;;
    *)
    ;;
esac