#!/bin/bash

export CUDA_VISIBLE_DEVICES="" # run on cpu (for this experiment it could make sense to comment this line and use the GPU)

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

export PYTHONPATH="${PYTHONPATH}:./nn-discrete-tf"

#-----------------------------------------------------------------------
# Use the following lines to run the experiments from the paper
# This experiment was run with SLURM for TASK_ID \in {1, ..., 1920}
#
# In the paper, this experiment was used to obtain results shown in
#  - Fig 3: - CNN ReLU
#           - CNN sign
#  - Fig 7: - CNN (comprises CNN ReLU, CNN sign)
# In the following lines, use the argument --activation=<relu,sign> to set the activation function
# Note: There is no option for one-hot encoded features
#-----------------------------------------------------------------------

# ReLU
python experiment_cnn_memory.py --taskid=$TASK_ID --experiment-dir=usps/cnn_relu_memory --dataset=usps --input-image-size=16 --n-input-channels=1 --n-folds=1 --pgm-n-parameters=8650 --activation=relu --n-epochs=500 --batch-size=100
# python experiment_cnn_memory.py --taskid=$TASK_ID --experiment-dir=mnist/cnn_relu_memory --dataset=mnist --input-image-size=14 --n-input-channels=1 --n-folds=1 --pgm-n-parameters=25800 --activation=relu --n-epochs=500 --batch-size=250

# sign
# python experiment_cnn_memory.py --taskid=$TASK_ID --experiment-dir=usps/cnn_sign_memory --dataset=usps --input-image-size=16 --n-input-channels=1 --n-folds=1 --pgm-n-parameters=8650 --activation=sign --n-epochs=500 --batch-size=100
# python experiment_cnn_memory.py --taskid=$TASK_ID --experiment-dir=mnist/cnn_sign_memory --dataset=mnist --input-image-size=14 --n-input-channels=1 --n-folds=1 --pgm-n-parameters=25800 --activation=sign --n-epochs=500 --batch-size=250
