#!/bin/bash
datasets=(ArticularyWordRecognition AtrialFibrillation BasicMotions Epilepsy ERing HandMovementDirection Libras NATOPS "'PEMS-SF'" PenDigits StandWalkJump UWaveGestureLibrary)
datasets=("'PEMS-SF'")
rm -f result_record.txt
for data_set in ${datasets[*]}; do
    # data_set=BasicMotions
    echo $data_set
    # 
    # echo "ts2vec" >> result_record_$data_set.txt
    echo $data_set >> result_record.txt
    for i in $(seq 0 4); do
        # echo $i
        cmd="python uea.py --dataset "$data_set"  --path /home/liangchen/Desktop/3liang/ts2vec/datasets/UEA \
        --save_path /home/liangchen/Desktop/3liang/UnsupervisedScalableRepresentationLearningTimeSeries/models \
        --hyper default_hyperparameters.json --cuda --seed "`expr 2077 + $i`" --rank  "$i
        $cmd 
        pid=$!
    done
    wait
    for i in $(seq 0 4); do
        cat $i.txt >> result_record.txt
        echo >> result_record.txt
        rm -f $i.txt
    done
done