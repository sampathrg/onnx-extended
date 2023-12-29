#!/bin/bash

export TEST_LLC_EXE=/home/srajendra/repos/llvm-project/build/bin/llc
export TB_DIR=/home/srajendra/repos/llvm-project/mlir/examples/treebeard/src/python

BATCH_SIZES=(512)
#BATCH_SIZES=(512 1024 2048 4096)
# MODEL_NAMES=("abalone_xgb_model_save")
MODEL_NAMES=("abalone_xgb_model_save" "airline_xgb_model_save" "airline-ohe_xgb_model_save" "epsilon_xgb_model_save" "higgs_xgb_model_save" "year_prediction_msd_xgb_model_save")
NUM_RUNS=500
N_REPEATS=5

for model_name in "${MODEL_NAMES[@]}"; do
    echo "Running model_name=$model_name"
    echo
    for batch_size in "${BATCH_SIZES[@]}"; do
        echo "Running batch_size=$batch_size"
        python3 /home/srajendra/repos/onnx-extended/_doc/examples/plot_op_tree_ensemble_implementations.py -n $NUM_RUNS -r $N_REPEATS --batch_size $batch_size --onnx_model $model_name
    done
done
