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
# This experiment was run with SLURM for TASK_ID \in {1, ..., 410}
#
# In the paper, this experiment was used to obtain results shown in
#  - Fig 5
# Using the option --init-ml the initial parameters can be set to the maximum likelihood parameters.
# However, note that --init-ml results are not reported in the paper.
#-----------------------------------------------------------------------

# Experiments using the "TAN Subset" setting from the PGM-2020 paper. This setting is used in the ICPR-2020 paper.
python experiment_tan_structure_learning_subset.py --taskid=$TASK_ID --experiment-dir=letter/tan_structure_learning_subset --dataset=letter --n-folds=1 --n-epochs=500 --batch-size=100 --learning-rate-structure=1e-3 --gumbel-softmax-temperature-start=1e1 --gumbel-softmax-temperature-end=1e-1
# python experiment_tan_structure_learning_subset.py --taskid=$TASK_ID --experiment-dir=satimage/tan_structure_learning_subset --dataset=satimage --n-folds=5 --n-epochs=500 --batch-size=50 --learning-rate-structure=1e-3 --gumbel-softmax-temperature-start=1e1 --gumbel-softmax-temperature-end=1e-1
# python experiment_tan_structure_learning_subset.py --taskid=$TASK_ID --experiment-dir=usps/tan_structure_learning_subset --dataset=usps --n-folds=1 --n-epochs=500 --batch-size=100 --learning-rate-structure=1e-3 --gumbel-softmax-temperature-start=1e1 --gumbel-softmax-temperature-end=1e-1
# python experiment_tan_structure_learning_subset.py --taskid=$TASK_ID --experiment-dir=mnist/tan_structure_learning_subset --dataset=mnist --n-folds=1 --n-epochs=500 --batch-size=250 --learning-rate-structure=1e-3 --gumbel-softmax-temperature-start=1e1 --gumbel-softmax-temperature-end=1e-1

# maximum-likelihood initialization (not reported in the paper)
# python experiment_tan_structure_learning_subset.py --taskid=$TASK_ID --experiment-dir=letter/tan_structure_learning_subset --dataset=letter --n-folds=1 --n-epochs=500 --batch-size=100 --learning-rate-structure=1e-3 --gumbel-softmax-temperature-start=1e1 --gumbel-softmax-temperature-end=1e-1 --init-ml
# python experiment_tan_structure_learning_subset.py --taskid=$TASK_ID --experiment-dir=satimage/tan_structure_learning_subset --dataset=satimage --n-folds=5 --n-epochs=500 --batch-size=50 --learning-rate-structure=1e-3 --gumbel-softmax-temperature-start=1e1 --gumbel-softmax-temperature-end=1e-1 --init-ml
# python experiment_tan_structure_learning_subset.py --taskid=$TASK_ID --experiment-dir=usps/tan_structure_learning_subset --dataset=usps --n-folds=1 --n-epochs=500 --batch-size=100 --learning-rate-structure=1e-3 --gumbel-softmax-temperature-start=1e1 --gumbel-softmax-temperature-end=1e-1 --init-ml
# python experiment_tan_structure_learning_subset.py --taskid=$TASK_ID --experiment-dir=mnist/tan_structure_learning_subset --dataset=mnist --n-folds=1 --n-epochs=500 --batch-size=250 --learning-rate-structure=1e-3 --gumbel-softmax-temperature-start=1e1 --gumbel-softmax-temperature-end=1e-1 --init-ml

#-----------------------------------------------------------------------
# Experiments using the "TAN All" setting from the PGM-2020 paper. This setting is not reported in the ICPR-2020 paper.
# Only implemented for letter and satimage. The other datasets have too many input features.
# python experiment_tan_structure_learning_all.py --taskid=$TASK_ID --experiment-dir=letter/tan_structure_learning_all --dataset=letter --n-folds=1 --n-epochs=500 --batch-size=100 --learning-rate-structure=1e-3 --gumbel-softmax-temperature-start=1e1 --gumbel-softmax-temperature-end=1e-1
# python experiment_tan_structure_learning_all.py --taskid=$TASK_ID --experiment-dir=satimage/tan_structure_learning_all --dataset=satimage --n-folds=5 --n-epochs=500 --batch-size=50 --learning-rate-structure=1e-3 --gumbel-softmax-temperature-start=1e1 --gumbel-softmax-temperature-end=1e-1

# maximum-likelihood initialization (not reported in the paper)
# python experiment_tan_structure_learning_all.py --taskid=$TASK_ID --experiment-dir=letter/tan_structure_learning_all --dataset=letter --n-folds=1 --n-epochs=500 --batch-size=100 --learning-rate-structure=1e-3 --gumbel-softmax-temperature-start=1e1 --gumbel-softmax-temperature-end=1e-1 --init-ml
# python experiment_tan_structure_learning_all.py --taskid=$TASK_ID --experiment-dir=satimage/tan_structure_learning_all --dataset=satimage --n-folds=5 --n-epochs=500 --batch-size=50 --learning-rate-structure=1e-3 --gumbel-softmax-temperature-start=1e1 --gumbel-softmax-temperature-end=1e-1 --init-ml
