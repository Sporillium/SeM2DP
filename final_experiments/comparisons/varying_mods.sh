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
    # Experiment 1: SIFT Features, MobileNetV2, Superclasses, All Sequences
        echo "$(date) Starting Experiment 1" >> $logfile
        python $programfile/parameter_tuning.py -e display -n 7 -m super -f params.txt --sem_model 6 >> $logfile
        python $programfile/parameter_tuning.py -e display -n 6 -m super -f params.txt --sem_model 6 >> $logfile
        python $programfile/parameter_tuning.py -e display -n 5 -m super -f params.txt --sem_model 6 >> $logfile
        python $programfile/parameter_tuning.py -e display -n 0 -m super -f params.txt --sem_model 6 >> $logfile
        echo "$(date) Experiment 1 Complete" >> $logfile
        echo "-------------------------------------"
    ;&
    2)
    # Experiment 2: SIFT Features, ResNet18, Superclasses, All Sequences
        echo "$(date) Starting Experiment 2" >> $logfile
        python $programfile/parameter_tuning.py -e display -n 7 -m super -f params.txt --sem_model 4 >> $logfile
        python $programfile/parameter_tuning.py -e display -n 6 -m super -f params.txt --sem_model 4 >> $logfile
        python $programfile/parameter_tuning.py -e display -n 5 -m super -f params.txt --sem_model 4 >> $logfile
        python $programfile/parameter_tuning.py -e display -n 0 -m super -f params.txt --sem_model 4 >> $logfile 
        echo "$(date) Experiment 2 Complete" >> $logfile
        echo "-------------------------------------"
    ;&
    3)
    # Experiment 3: SIFT Features, ResNet50, Superclasses, All Sequences
        echo "$(date) Starting Experiment 3" >> $logfile 
        python $programfile/parameter_tuning.py -e display -n 7 -m super -f params.txt --sem_model 2 >> $logfile
        python $programfile/parameter_tuning.py -e display -n 6 -m super -f params.txt --sem_model 2 >> $logfile
        python $programfile/parameter_tuning.py -e display -n 5 -m super -f params.txt --sem_model 2 >> $logfile
        python $programfile/parameter_tuning.py -e display -n 0 -m super -f params.txt --sem_model 2 >> $logfile 
        echo "$(date) Experiment 3 Complete" >> $logfile
        echo "-------------------------------------"
    ;&
    4)
    # Experiment 4: SIFT Features, ResNet101, Normal, All Sequences
        echo "$(date) Starting Experiment 4" >> $logfile
        python $programfile/parameter_tuning.py -e display -n 7 -m normal -f params.txt --sem_model 0 >> $logfile
        python $programfile/parameter_tuning.py -e display -n 6 -m normal -f params.txt --sem_model 0 >> $logfile
        python $programfile/parameter_tuning.py -e display -n 5 -m normal -f params.txt --sem_model 0 >> $logfile
        python $programfile/parameter_tuning.py -e display -n 0 -m normal -f params.txt --sem_model 0 >> $logfile
        echo "$(date) Experiment 4 Complete" >> $logfile
        echo "-------------------------------------"
    ;&
    5)
    # Experiment 5: SIFT Features, MobileNetV2, Normal, All Sequences
        echo "$(date) Starting Experiment 4" >> $logfile
        python $programfile/parameter_tuning.py -e display -n 7 -m normal -f params.txt --sem_model 6 >> $logfile
        python $programfile/parameter_tuning.py -e display -n 6 -m normal -f params.txt --sem_model 6 >> $logfile
        python $programfile/parameter_tuning.py -e display -n 5 -m normal -f params.txt --sem_model 6 >> $logfile
        python $programfile/parameter_tuning.py -e display -n 0 -m normal -f params.txt --sem_model 6 >> $logfile
        echo "$(date) Experiment 4 Complete" >> $logfile
        echo "-------------------------------------"
    ;&
    6)
    # Experiment 6: SIFT Features, ResNet18, Normal, All Sequences
        echo "$(date) Starting Experiment 5" >> $logfile
        python $programfile/parameter_tuning.py -e display -n 7 -m normal -f params.txt --sem_model 4 >> $logfile
        python $programfile/parameter_tuning.py -e display -n 6 -m normal -f params.txt --sem_model 4 >> $logfile
        python $programfile/parameter_tuning.py -e display -n 5 -m normal -f params.txt --sem_model 4 >> $logfile
        python $programfile/parameter_tuning.py -e display -n 0 -m normal -f params.txt --sem_model 4 >> $logfile 
        echo "$(date) Experiment 5 Complete" >> $logfile
        echo "-------------------------------------"
    ;&
    7)
    # Experiment 7: SIFT Features, ResNet50, Normal, All Sequences
        echo "$(date) Starting Experiment 6" >> $logfile 
        python $programfile/parameter_tuning.py -e display -n 7 -m normal -f params.txt --sem_model 2 >> $logfile
        python $programfile/parameter_tuning.py -e display -n 6 -m normal -f params.txt --sem_model 2 >> $logfile
        python $programfile/parameter_tuning.py -e display -n 5 -m normal -f params.txt --sem_model 2 >> $logfile
        python $programfile/parameter_tuning.py -e display -n 0 -m normal -f params.txt --sem_model 2 >> $logfile 
        echo "$(date) Experiment 6 Complete" >> $logfile
        echo "-------------------------------------"
    ;;
    8)
    # Experiment 8: SIFT Features, ResNet101, Normal, All Sequences
        echo "$(date) Starting Experiment 6" >> $logfile 
        python $programfile/parameter_tuning.py -e display -n 7 -m normal -f params.txt --sem_model 0 >> $logfile
        python $programfile/parameter_tuning.py -e display -n 6 -m normal -f params.txt --sem_model 0 >> $logfile
        python $programfile/parameter_tuning.py -e display -n 5 -m normal -f params.txt --sem_model 0 >> $logfile
        python $programfile/parameter_tuning.py -e display -n 0 -m normal -f params.txt --sem_model 0 >> $logfile 
        echo "$(date) Experiment 6 Complete" >> $logfile
        echo "-------------------------------------"
    ;;
    *)
    ;;
esac