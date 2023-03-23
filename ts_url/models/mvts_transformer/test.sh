#!/bin/bash
datasets=(StandWalkJump)
# datasets=(StandWalkJump)
rm -f res_file_paths.txt
for data_set in ${datasets[*]}; do
    # data_set=BasicMotions
    echo $data_set
    echo $data_set >> res_file_paths.txt
    for i in $(seq 5); do
        python src/main.py --output_dir experiments --comment "pretraining through imputation" --name ${data_set}_pretrained --records_file Imputation_records.xls \
        --data_dir /home/liangchen/Desktop/3liang/mvts_transformer/src/datasets/classify/${data_set} \
         --data_class tsra --pattern TRAIN --val_pattern TEST \
        --epochs 700 --lr 0.001 --optimizer RAdam --pos_encoding learnable --task imputation \
        --batch_size 1 --gpu 0 --d_model 320 --seed `expr 2077 + $i`
    done
done