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

export PYTHONPATH="${PYTHONPATH}:./nn-discrete-tf"

#-----------------------------------------------------------------------
# Use the following lines to run the experiments from the paper
# This experiment was run with SLURM for TASK_ID \in {1, ..., 9600}
# Using a higher TASK_ID \in {9601, ..., 12480} uses model size multipliers \in {16,24,32,48,64} but these results are
# not reported in the paper.
#
# In the paper, this experiment was used to obtain results shown in
#  - Fig 3: - FC NN ReLU
#           - FC NN sign
#           - FC NN ReLU (one-hot)
#           - FC NN sign (one-hot)
#  - Fig 7: - FC NN (comprises FC NN ReLU, FC NN sign)
# In the following lines, use the argument --activation=<relu,sign> to set the activation function
#                         use --one-hot to run the experiment on one-hot features (default: don't use one-hot features)
#-----------------------------------------------------------------------

# ReLU
python experiment_fc_dnn.py --taskid=$TASK_ID --experiment-dir=letter/fcnn_relu --dataset=letter --n-folds=1 --pgm-n-parameters=3822 --activation=relu --n-epochs=500 --batch-size=100
# python experiment_fc_dnn.py --taskid=$TASK_ID --experiment-dir=satimage/fcnn_relu --dataset=satimage --n-folds=5 --pgm-n-parameters=2508 --activation=relu --n-epochs=500 --batch-size=50
# python experiment_fc_dnn.py --taskid=$TASK_ID --experiment-dir=usps/fcnn_relu --dataset=usps --n-folds=1 --pgm-n-parameters=8650 --activation=relu --n-epochs=500 --batch-size=100
# python experiment_fc_dnn.py --taskid=$TASK_ID --experiment-dir=mnist/fcnn_relu --dataset=mnist --n-folds=1 --pgm-n-parameters=25800 --activation=relu --n-epochs=500 --batch-size=250

# sign
# python experiment_fc_dnn.py --taskid=$TASK_ID --experiment-dir=letter/fcnn_sign --dataset=letter --n-folds=1 --pgm-n-parameters=3822 --activation=sign --n-epochs=500 --batch-size=100
# python experiment_fc_dnn.py --taskid=$TASK_ID --experiment-dir=satimage/fcnn_sign --dataset=satimage --n-folds=5 --pgm-n-parameters=2508 --activation=sign --n-epochs=500 --batch-size=50
# python experiment_fc_dnn.py --taskid=$TASK_ID --experiment-dir=usps/fcnn_sign --dataset=usps --n-folds=1 --pgm-n-parameters=8650 --activation=sign --n-epochs=500 --batch-size=100
# python experiment_fc_dnn.py --taskid=$TASK_ID --experiment-dir=mnist/fcnn_sign --dataset=mnist --n-folds=1 --pgm-n-parameters=25800 --activation=sign --n-epochs=500 --batch-size=250

# ReLU one-hot
# python experiment_fc_dnn.py --taskid=$TASK_ID --experiment-dir=letter/fcnn_relu_onehot --dataset=letter --one-hot --n-folds=1 --pgm-n-parameters=3822 --activation=relu --n-epochs=500 --batch-size=100
# python experiment_fc_dnn.py --taskid=$TASK_ID --experiment-dir=satimage/fcnn_relu_onehot --dataset=satimage --one-hot --n-folds=5 --pgm-n-parameters=2508 --activation=relu --n-epochs=500 --batch-size=50
# python experiment_fc_dnn.py --taskid=$TASK_ID --experiment-dir=usps/fcnn_relu_onehot --dataset=usps --one-hot --n-folds=1 --pgm-n-parameters=8650 --activation=relu --n-epochs=500 --batch-size=100
# python experiment_fc_dnn.py --taskid=$TASK_ID --experiment-dir=mnist/fcnn_relu_onehot --dataset=mnist --one-hot --n-folds=1 --pgm-n-parameters=25800 --activation=relu --n-epochs=500 --batch-size=250

# sign one-hot
# python experiment_fc_dnn.py --taskid=$TASK_ID --experiment-dir=letter/fcnn_sign_onehot --dataset=letter --one-hot --n-folds=1 --pgm-n-parameters=3822 --activation=sign --n-epochs=500 --batch-size=100
# python experiment_fc_dnn.py --taskid=$TASK_ID --experiment-dir=satimage/fcnn_sign_onehot --dataset=satimage --one-hot --n-folds=5 --pgm-n-parameters=2508 --activation=sign --n-epochs=500 --batch-size=50
# python experiment_fc_dnn.py --taskid=$TASK_ID --experiment-dir=usps/fcnn_sign_onehot --dataset=usps --one-hot --n-folds=1 --pgm-n-parameters=8650 --activation=sign --n-epochs=500 --batch-size=100
# python experiment_fc_dnn.py --taskid=$TASK_ID --experiment-dir=mnist/fcnn_sign_onehot --dataset=mnist --one-hot --n-folds=1 --pgm-n-parameters=25800 --activation=sign --n-epochs=500 --batch-size=250
