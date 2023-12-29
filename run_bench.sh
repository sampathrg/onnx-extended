#!/bin/bash

conda activate base
export TEST_LLC_EXE=/home/srajendra/repos/llvm-project/build/bin/llc
export TB_DIR=/home/srajendra/repos/llvm-project/mlir/examples/treebeard/src/python

BATCH_SIZES=(512 1024 2048 4096)
N_TREES=(256 512 1024)
N_FEATURES=(256 512 1024)
NUM_RUNS=50
N_REPEATS=5

for batch_size in "${BATCH_SIZES[@]}"; do
    for n_trees in "${N_TREES[@]}"; do
        for n_features in "${N_FEATURES[@]}"; do
            echo "Running batch_size=$batch_size, n_trees=$n_trees, n_features=$n_features"
            python3 /home/srajendra/repos/onnx-extended/_doc/examples/plot_op_tree_ensemble_implementations.py -n $NUM_RUNS -r $N_REPEATS --n_features $n_features --n_trees $n_trees --batch_size $batch_size
        done
    done
done
