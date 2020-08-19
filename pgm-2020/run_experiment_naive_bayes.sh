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
# This experiment was run with SLURM for TASK_ID \in {1, ..., 1000}
# In the paper, this experiment was used to obtain results for L_HYB / NB in Table 1

python experiment_naive_bayes.py --taskid=$TASK_ID --experiment-dir=letter/naive_bayes --dataset=letter --n-folds=1 --n-epochs=500 --batch-size=100
# python experiment_naive_bayes.py --taskid=$TASK_ID --experiment-dir=satimage/naive_bayes --dataset=satimage --n-folds=5 --n-epochs=500 --batch-size=50
# python experiment_naive_bayes.py --taskid=$TASK_ID --experiment-dir=usps/naive_bayes --dataset=usps --n-folds=1 --n-epochs=500 --batch-size=100
# python experiment_naive_bayes.py --taskid=$TASK_ID --experiment-dir=mnist/naive_bayes --dataset=mnist --n-folds=1 --n-epochs=500 --batch-size=250

#-----------------------------------------------------------------------
# Use the following lines to initialize parameters with ML parameters
# The initial test error is reported in the paper as L_NLL / NB in Table 1

# python experiment_naive_bayes.py --taskid=$TASK_ID --experiment-dir=letter/naive_bayes --dataset=letter --n-folds=1 --n-epochs=500 --batch-size=100 --init-ml
# python experiment_naive_bayes.py --taskid=$TASK_ID --experiment-dir=satimage/naive_bayes --dataset=satimage --n-folds=5 --n-epochs=500 --batch-size=50 --init-ml
# python experiment_naive_bayes.py --taskid=$TASK_ID --experiment-dir=usps/naive_bayes --dataset=usps --n-folds=1 --n-epochs=500 --batch-size=100 --init-ml
# python experiment_naive_bayes.py --taskid=$TASK_ID --experiment-dir=mnist/naive_bayes --dataset=mnist --n-folds=1 --n-epochs=500 --batch-size=250 --init-ml

