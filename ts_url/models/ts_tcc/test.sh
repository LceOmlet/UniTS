#!/bin/bash
datasets=(ArticularyWordRecognition AtrialFibrillation BasicMotions Epilepsy ERing HandMovementDirection Libras NATOPS "PEMS-SF" PenDigits StandWalkJump UWaveGestureLibrary)
datasets=("PEMS-SF")

# datasets=(StandWalkJump)
# rm -f res_file_paths.txt
for data_set in ${datasets[*]}; do
    # data_set=BasicMotions
    # echo $data_set
    # echo $data_set >> res_file_paths.txt
    for i in $(seq 5); do
        python main.py --experiment_description exp1 --run_description run_1 --seed `expr 123 + $i` --training_mode self_supervised --selected_dataset $data_set --rank $i
    done
done