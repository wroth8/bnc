#!/bin/bash

export CUDA_VISIBLE_DEVICES="" # run on cpu

# Note: In case the CPU usage of the experiment goes crazy, try uncommenting the following lines.
# export MKL_NUM_THREADS=1
# export NUMEXPR_NUM_THREADS=1
# export OMP_NUM_THREADS=1

if [ -z ${SLURM_ARRAY_TASK_ID+x} ]; then
  echo "SLURM_ARRAY_TASK_ID not set. Using task id 1"
  TASK_ID="1"
else
  TASK_ID=`printf "%d" "$SLURM_ARRAY_TASK_ID"`
fi

#-----------------------------------------------------------------------
# Use the following lines to run the experiments from the paper
# This experiment was run with SLURM for TASK_ID \in {1, ..., 30} for letter and TASK_ID \in {1, ..., 72} for satimage
# The purpose of this experiment is to check whether the structure learning method can recover the Chow-Liu structure
# Note: We were able to recover the Chow-Liu structure for learning rate 3e-2 that is used in every experiment with an odd TASK_ID

python experiment_recover_chow_liu.py --taskid=$TASK_ID --experiment=letter/recover_chow_liu --dataset=letter --n-folds=1 --n-epochs=500 --batch-size=100 --learning-rate-structure=1e-3 --gumbel-softmax-temperature-start=1e1 --gumbel-softmax-temperature-end=1e-1
# python experiment_recover_chow_liu.py --taskid=$TASK_ID --experiment=satimage/recover_chow_liu --dataset=satimage --n-folds=5 --n-epochs=500 --batch-size=50 --learning-rate-structure=1e-3 --gumbel-softmax-temperature-start=1e1 --gumbel-softmax-temperature-end=1e-1
