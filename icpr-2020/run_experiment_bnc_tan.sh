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
# This experiment was run with SLURM for TASK_ID \in {1, ..., 9600}
#
# In the paper, this experiment was used to obtain results shown in
#  - Fig 6: - BNC TAN
#  - Fig 7: - BNC TAN
# Using the option --init-ml the initial parameters can be set to the maximum likelihood parameters.
# However, note that is not reported in the paper.
#-----------------------------------------------------------------------

python experiment_bnc_tan.py --taskid=$TASK_ID --experiment-dir=letter/bnc_tan --dataset=letter --n-folds=1 --n-epochs=500 --batch-size=100
# python experiment_bnc_tan.py --taskid=$TASK_ID --experiment-dir=satimage/bnc_tan --dataset=satimage --n-folds=5 --n-epochs=500 --batch-size=50
# python experiment_bnc_tan.py --taskid=$TASK_ID --experiment-dir=usps/bnc_tan --dataset=usps --n-folds=1 --n-epochs=500 --batch-size=100
# python experiment_bnc_tan.py --taskid=$TASK_ID --experiment-dir=mnist/bnc_tan --dataset=mnist --n-folds=1 --n-epochs=500 --batch-size=250

# maximum-likelihood initialization (not reported in the paper)
# python experiment_bnc_tan.py --taskid=$TASK_ID --experiment-dir=letter/bnc_tan --dataset=letter --n-folds=1 --n-epochs=500 --batch-size=100 --init-ml
# python experiment_bnc_tan.py --taskid=$TASK_ID --experiment-dir=satimage/bnc_tan --dataset=satimage --n-folds=5 --n-epochs=500 --batch-size=50 --init-ml
# python experiment_bnc_tan.py --taskid=$TASK_ID --experiment-dir=usps/bnc_tan --dataset=usps --n-folds=1 --n-epochs=500 --batch-size=100 --init-ml
# python experiment_bnc_tan.py --taskid=$TASK_ID --experiment-dir=mnist/bnc_tan --dataset=mnist --n-folds=1 --n-epochs=500 --batch-size=250 --init-ml
