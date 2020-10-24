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
#  - Fig 3: - BNC NB STE (ours)
#  - Fig 4: - BNC NB
#  - Fig 6: - BNC NB
#  - Fig 7: - BNC NB
# Using the option --init-ml the initial parameters can be set to the maximum likelihood parameters.
# However, note that is not reported in the paper.
#-----------------------------------------------------------------------

python experiment_bnc_nb.py --taskid=$TASK_ID --experiment-dir=letter/bnc_nb --dataset=letter --n-folds=1 --n-epochs=500 --batch-size=100
# python experiment_bnc_nb.py --taskid=$TASK_ID --experiment-dir=satimage/bnc_nb --dataset=satimage --n-folds=5 --n-epochs=500 --batch-size=50
# python experiment_bnc_nb.py --taskid=$TASK_ID --experiment-dir=usps/bnc_nb --dataset=usps --n-folds=1 --n-epochs=500 --batch-size=100
# python experiment_bnc_nb.py --taskid=$TASK_ID --experiment-dir=mnist/bnc_nb --dataset=mnist --n-folds=1 --n-epochs=500 --batch-size=250

# maximum-likelihood initialization (not reported in the paper)
# python experiment_bnc_nb.py --taskid=$TASK_ID --experiment-dir=letter/bnc_nb --dataset=letter --n-folds=1 --n-epochs=500 --batch-size=100 --init-ml
# python experiment_bnc_nb.py --taskid=$TASK_ID --experiment-dir=satimage/bnc_nb --dataset=satimage --n-folds=5 --n-epochs=500 --batch-size=50 --init-ml
# python experiment_bnc_nb.py --taskid=$TASK_ID --experiment-dir=usps/bnc_nb --dataset=usps --n-folds=1 --n-epochs=500 --batch-size=100 --init-ml
# python experiment_bnc_nb.py --taskid=$TASK_ID --experiment-dir=mnist/bnc_nb --dataset=mnist --n-folds=1 --n-epochs=500 --batch-size=250 --init-ml
