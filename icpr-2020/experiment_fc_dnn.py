import tensorflow as tf
import numpy as np

from time import time
from optparse import OptionParser
from itertools import product

from tensorflow.keras import Model
from tensorflow.keras.layers import Softmax

from layers.DistDropout import DistDropout
from layers.DistDense import DistDense
from layers.DistBatchNormalization import DistBatchNormalization
from layers.DistReLU import DistReLU
from layers.DistSign import DistSign

from layers.weights.RealWeights import RealWeights
from layers.weights.QuantizedWeightsStraightThrough import QuantizedWeightsStraightThrough

from layers.ste import sign_ste_id, sign_dorefa, linear_quantizer, linear_quantizer_dorefa, tanh_quantizer_dorefa


class FCNN(Model):
    def __init__(self,
                 layout,
                 n_bits_per_weight=32,
                 activation='relu',
                 regularize_weights_l1=0.0,
                 regularize_weights_l2=0.0,
                 enable_batchnorm=True,
                 batchnorm_momentum=0.99,
                 dropout_rate=None):
        super(FCNN, self).__init__()
        '''
          layout: Layout does not include the number of input features
        '''
        
        if dropout_rate is None:
            dropout_rate = [0.0] * len(layout)
        assert len(dropout_rate) == len(layout)

        self.batchnorm_momentum = batchnorm_momentum
        if n_bits_per_weight == 32:
            create_weights = lambda : RealWeights(regularize_l1=regularize_weights_l1,
                                                  regularize_l2=regularize_weights_l2)
        else:
            if n_bits_per_weight == 1:
                quantizer = sign_ste_id
            else:
                quantizer = lambda x : linear_quantizer_dorefa(x, float(n_bits_per_weight), -1.0, 1.0)

            create_weights = lambda : QuantizedWeightsStraightThrough(quantizer,
                                                                      regularize_l1=regularize_weights_l1,
                                                                      regularize_l2=regularize_weights_l2)

        if activation == 'relu':
            create_activation = lambda : DistReLU()
        elif activation == 'sign':
            create_activation = lambda : DistSign(has_zero_output=False, straight_through_type='tanh', stochastic=True)
        else:
            raise NotImplementedError('Activation \'{}\' not implemented'.format(activation))
        
        create_batchnorm = lambda : DistBatchNormalization(momentum=batchnorm_momentum) if enable_batchnorm else None

        create_dropout = lambda dropout_rate : DistDropout(dropout_rate=dropout_rate, scale_at_training=True) if dropout_rate > 0.0 else None

        self.layout = layout
        self.dropout = []
        self.dense = []
        self.batchnorm = []
        self.act = []
        for layer_idx, n_neurons in enumerate(layout):
            is_last_layer = layer_idx == len(layout) - 1
            self.dropout.append(create_dropout(dropout_rate[layer_idx]))
            self.dense.append(DistDense(n_neurons, create_weights(), use_bias=is_last_layer))
            if is_last_layer:
                self.softmax = Softmax()
            else:
                self.batchnorm.append(create_batchnorm())
                self.act.append(create_activation())

    
    def call(self, x, training):
        for layer_idx in range(len(self.layout)):
            is_last_layer = layer_idx == len(self.layout) - 1
            if self.dropout[layer_idx] is not None:
                x = self.dropout[layer_idx](x, training)
            x = self.dense[layer_idx](x, training)
            if is_last_layer:
                x = self.softmax(x)
            else:
                if self.batchnorm[layer_idx] is not None:
                    x = self.batchnorm[layer_idx](x, training)
                x = self.act[layer_idx](x, training)
        return x


def createFCNN(n_target_bits,
               n_input_features,
               n_classes,
               n_bits_per_weight,
               n_layers,
               enable_batchnorm,
               activation,
               n_bits_per_batchnorm=64,
               n_bits_per_bias=32,
               dropout_rate=0.0):
    '''
    Calculates the number of hidden units to match the number of bits for the parameters to a given target number of bits.
    n_target_bits: The target number of bits
    n_input_features: The number of input features
    n_classes: The number of classes
    n_bits_per_weight: The number of bits used for each weight
    n_layers: The number of layers, i.e., the number of weight matrices.
    enable_batchnorm: Determines whether batch normalization should be used
    activation: The activation function ('relu' or 'sign')
    n_bits_per_batchnorm: The number of bits consumed by each neuron for batch normalization
    n_bis_per_bias: The number of bits consumed by each bias. A bias is used in the last layer.
    dropout_rate: A dropout rate used for all layers.
    '''
    assert n_layers > 1
    if not enable_batchnorm:
        n_bits_per_batchnorm = 0
    if n_layers == 2:
        n_bits_per_hidden = (n_input_features + n_classes) * n_bits_per_weight + n_bits_per_batchnorm
        n_hidden_real = (n_target_bits - n_classes * n_bits_per_bias) / n_bits_per_hidden
        n_hidden = round(n_hidden_real)
        layout = [n_hidden, n_classes]
    else:
        a = n_bits_per_weight * (n_layers - 2)
        b = n_bits_per_weight * (n_input_features + n_classes) + (n_layers - 1) * n_bits_per_batchnorm
        c = n_bits_per_bias * n_classes - n_target_bits
        n_hidden1 = (-b - (b ** 2.0 - 4.0 * a * c) ** 0.5) / (2.0 * a)
        n_hidden2 = (-b + (b ** 2.0 - 4.0 * a * c) ** 0.5) / (2.0 * a)
        n_hidden_real = max(n_hidden1, n_hidden2)
        n_hidden = round(n_hidden_real)
        layout = [n_hidden] * (n_layers - 1) + [n_classes]
    n_hidden = max(n_hidden, 1) # we need at least one neuron
    model = FCNN(layout=layout,
         n_bits_per_weight=n_bits_per_weight,
         activation=activation,
         enable_batchnorm=enable_batchnorm,
         batchnorm_momentum=0.9,
         regularize_weights_l2=0.0,
         dropout_rate=[dropout_rate] * n_layers)

    # This is to verify our implementation
    n_bits_total_real = (n_layers - 1) * n_hidden_real * n_bits_per_batchnorm # all batchnorm bits
    n_bits_total_real += n_classes * n_bits_per_bias # all bias bits
    n_bits_total_real += n_input_features * n_hidden_real * n_bits_per_weight # weight bits of first layer
    n_bits_total_real += n_classes * n_hidden_real * n_bits_per_weight # weight bits of last layer
    n_bits_total_real += n_hidden_real ** 2.0 *n_bits_per_weight * (n_layers - 2) # weight bits of intermediate layers

    n_bits_total_integer = (n_layers - 1) * n_hidden * n_bits_per_batchnorm # all batchnorm bits
    n_bits_total_integer += n_classes * n_bits_per_bias # all bias bits
    n_bits_total_integer += n_input_features * n_hidden * n_bits_per_weight # weight bits of first layer
    n_bits_total_integer += n_classes * n_hidden * n_bits_per_weight # weight bits of last layer
    n_bits_total_integer += n_hidden ** 2.0 *n_bits_per_weight * (n_layers - 2) # weight bits of intermediate layers

    print('Number of neurons: {} (real), {} (integer/rounded)'.format(n_hidden_real, n_hidden))
    print('Layout: {}'.format(layout))
    print('Number of target bits:                        {}'.format(n_target_bits))
    print('Number of total bits used [real #neurons]:    {}'.format(n_bits_total_real))
    print('Number of total bits used [integer #neurons]: {}'.format(n_bits_total_integer))

    if abs(n_bits_total_real - n_target_bits) > n_target_bits * 0.001:
        raise Exception('It seems we do something wrong')

    return model


def run(dataset,
        enable_one_hot_features,
        pgm_n_parameters,
        pgm_bits_per_parameter,
        n_bits_per_weight,
        n_layers,
        activation,
        enable_batchnorm,
        n_epochs,
        batch_size,
        learning_rate_start,
        fold_idx,
        dataset_dir,
        tensorboard_logdir):
    #-------------------------------------------------------------------------------------------------------------------
    # Data loader
    dataset_file = '{}/{}.npz'.format(dataset_dir, dataset)
    dataset_dict = dict(np.load(dataset_file))
    x_train = dataset_dict['x_tr_fold{}'.format(fold_idx + 1)].astype(np.int32) - 1 # loaded features start at 1
    y_train = dataset_dict['t_tr_fold{}'.format(fold_idx + 1)]
    x_test = dataset_dict['x_te_fold{}'.format(fold_idx + 1)].astype(np.int32) - 1
    y_test = dataset_dict['t_te_fold{}'.format(fold_idx + 1)]

    print('x_train.shape: {}'.format(x_train.shape))
    print('y_train.shape: {}'.format(y_train.shape))
    print('x_test.shape:  {}'.format(x_test.shape))
    print('y_test.shape:  {}'.format(y_test.shape))
    n_samples = x_train.shape[0]
    n_input_features = x_train.shape[1]
    n_classes = int(np.max(np.concatenate([y_train, y_test]) + 1))
    print('#samples:        {}'.format(n_samples))
    print('#input features: {}'.format(n_input_features))
    print('#classes:        {}'.format(n_classes))

    if enable_one_hot_features:
        def convert_to_onehot(indices, depth, dtype):
            assert indices.ndim == 1
            assert np.all(indices >= 0)
            assert np.all(indices < depth)
            onehot = np.zeros((indices.size, depth), dtype=dtype)
            onehot[np.arange(indices.size), indices] = 1
            return onehot

        print('Transforming discrete input features to one-hot encoding...')
        x_train_onehot = np.zeros((x_train.shape[0], 0), dtype=np.float32)
        x_test_onehot = np.zeros((x_test.shape[0], 0), dtype=np.float32)
        n_unique_features = []
        for feature_idx in range(n_input_features):
            unique_features = np.unique(x_train[:, feature_idx])
            assert unique_features.size == unique_features[-1] + 1
            n_unique_features.append(unique_features.size)

            x_train_onehot = np.concatenate(
                    [x_train_onehot, convert_to_onehot(x_train[:, feature_idx], unique_features.size, np.float32)], axis=1)
            x_test_onehot = np.concatenate(
                    [x_test_onehot, convert_to_onehot(x_test[:, feature_idx], unique_features.size, np.float32)], axis=1)

        print('Unique features:', n_unique_features)
        x_train = x_train_onehot
        x_test = x_test_onehot
        n_input_features = x_train.shape[1]
        print('#input features [new]: {}'.format(n_input_features))
    else:
        x_train = x_train.astype(np.float32)
        x_test = x_test.astype(np.float32)

        x_train_mean = np.mean(x_train, axis=0)
        x_train_std = np.std(x_train, axis=0)
        x_train_std[x_train_std == 0.0] = 1.0
        x_train = (x_train - x_train_mean) * (1.0 / x_train_std)
        x_test = (x_test - x_train_mean) * (1.0 / x_train_std)

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_ds = train_ds.shuffle(n_samples)
    train_ds = train_ds.batch(batch_size, drop_remainder=True)
    train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)

    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_ds = test_ds.batch(1000)
    test_ds = test_ds.prefetch(tf.data.experimental.AUTOTUNE)

    #-----------------------------------------------------------------------------------------------------------------------
    # Create the model
    model = createFCNN(n_target_bits=pgm_n_parameters * pgm_bits_per_parameter,
                       n_input_features=n_input_features,
                       n_classes=n_classes,
                       n_bits_per_weight=n_bits_per_weight,
                       n_layers=n_layers,
                       enable_batchnorm=enable_batchnorm,
                       activation=activation)

    model(np.ones((2,x_train.shape[1]), dtype=np.float32), True) # Build the model

    print('#' * 80)
    model.summary()
    print('#' * 80)

    #-----------------------------------------------------------------------------------------------------------------------
    # Create TF functions
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    test_loss = tf.keras.metrics.SparseCategoricalCrossentropy(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    learning_rate_end = learning_rate_start * 1e-4 # sweep learning rate over 4 orders of magnitude
    learning_rate_variable = tf.Variable(learning_rate_start, tf.float32) # suitable for real weights and relu activation
    learning_rate_schedule = np.logspace(np.log10(learning_rate_start), np.log10(learning_rate_end), n_epochs)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_variable)

    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            predictions = model(images, True)
            loss = loss_object(labels, predictions)
            if model.losses:
                loss += tf.add_n(model.losses)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_accuracy(labels, predictions)


    @tf.function
    def train_prediction_updates(images):
        model(images, False)


    @tf.function
    def test_step(images, labels):
        predictions = model(images, False)
        t_loss = loss_object(labels, predictions)

        test_loss(labels, predictions)
        test_accuracy(labels, predictions)

    #-----------------------------------------------------------------------------------------------------------------------
    # Optimization
    enable_tensorboard_logging = True
    if enable_tensorboard_logging:
        logwriter = tf.summary.create_file_writer(tensorboard_logdir)

    # Compute initial test errors
    t0_eval_test = time()
    for test_images, test_labels in test_ds:
        test_step(test_images, test_labels)
    t_elapsed_eval_test = time() - t0_eval_test

    # Convert tensors to python scalars
    to_pyscalar = lambda tensor : tensor.numpy().item()
    pyval_test_loss = to_pyscalar(test_loss.result())
    pyval_test_error = to_pyscalar(1.0 - test_accuracy.result())

    template = 'Epoch {:3d}/{:3d}, Loss: {:12e}, CE[TR]: {:8.5f}, Loss[TE]: {:e}, CE[TE]: {:8.5f}'
    print(template.format(0,
                          n_epochs,
                          float('nan'),
                          float('nan'),
                          pyval_test_loss,
                          pyval_test_error * 100.0))
    test_loss.reset_states()
    test_accuracy.reset_states()
    
    stats = {'train_loss'          : [],
             'train_error'         : [],
             'test_loss'           : [],
             'test_error'          : [],
             't_elapsed_train'     : [],
             't_elapsed_eval_test' : [],
             'learning_rate'       : []}

    for epoch in range(n_epochs):
        learning_rate_variable.assign(learning_rate_schedule[epoch])
        if epoch % 100 == 0:
            print('Current learning rate: {}'.format(learning_rate_variable))

        t0_train = time()
        for images, labels in train_ds:
            train_step(images, labels)
        t_elapsed_train = time() - t0_train

        t0_eval_test = time()
        for idx, (test_images, test_labels) in enumerate(test_ds):
            test_step(test_images, test_labels)
        t_elapsed_eval_test = time() - t0_eval_test

        # Convert tensors to python scalars
        pyval_train_loss = to_pyscalar(train_loss.result())
        pyval_train_error = to_pyscalar(1.0 - train_accuracy.result())
        pyval_test_loss = to_pyscalar(test_loss.result())
        pyval_test_error = to_pyscalar(1.0 - test_accuracy.result())
        pyval_learning_rate = to_pyscalar(learning_rate_variable)

        template = 'Epoch {:3d}/{:3d}, Loss: {:e}, CE[TR]: {:8.5f}, Loss[TE]: {:e}, CE[TE]: {:8.5f} [t_elapsed: {:6.2f} seconds]'
        print(template.format(epoch+1,
                              n_epochs,
                              pyval_train_loss,
                              pyval_train_error * 100.0,
                              pyval_test_loss,
                              pyval_test_error * 100.0,
                              t_elapsed_train + t_elapsed_eval_test))

        # Write interesting values to tensorboard
        if enable_tensorboard_logging:
            with logwriter.as_default():
                tf.summary.scalar("train_loss", pyval_train_loss, step=epoch+1)
                tf.summary.scalar("train_error", pyval_train_error * 100.0, step=epoch+1)
                tf.summary.scalar("test_loss", pyval_test_loss, step=epoch+1)
                tf.summary.scalar("test_error", pyval_test_error * 100.0, step=epoch+1)
                tf.summary.scalar("t_elapsed_train", t_elapsed_train, step=epoch+1)
                tf.summary.scalar("t_elapsed_eval_test", t_elapsed_eval_test, step=epoch+1)
                tf.summary.scalar("learning_rate", pyval_learning_rate, step=epoch+1)
        # Write interesting values to stats
        stats['train_loss'].append(pyval_train_loss)
        stats['train_error'].append(pyval_train_error)
        stats['test_loss'].append(pyval_test_loss)
        stats['test_error'].append(pyval_test_error)
        stats['t_elapsed_train'].append(t_elapsed_train)
        stats['t_elapsed_eval_test'].append(t_elapsed_eval_test)
        stats['learning_rate'].append(pyval_learning_rate)   

        # Reset the metrics for the next epoch
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()

    # Convert stats to numpy
    for key in stats:
        stats[key] = np.asarray(stats[key])
    return stats


def main():
    #-----------------------------------------------------------------------------------------------------------------------
    # Set up experiment
    parser = OptionParser()
    parser.add_option('--taskid', action='store', type='int', dest='taskid', default=1)
    parser.add_option('--experiment-dir', action='store', type='string', dest='experiment_dir', default='.')
    parser.add_option('--dataset-dir', action='store', type='string', dest='dataset_dir', default='../datasets')
    parser.add_option('--dataset', action='store', type='string', dest='dataset', default=None)
    parser.add_option('--one-hot', action='store_true', dest='enable_one_hot_features', default=False)
    parser.add_option('--n-folds', action='store', type='int', dest='n_folds', default=0)
    parser.add_option('--pgm-n-parameters', action='store', type='int', dest='pgm_n_parameters', default=0)
    parser.add_option('--activation', action='store', type='string', dest='activation', default='relu')
    parser.add_option('--n-epochs', action='store', type='int', dest='n_epochs', default=300)
    parser.add_option('--batch-size', action='store', type='int', dest='batch_size', default=100)
    
    options, args = parser.parse_args()
    taskid = options.taskid
    experiment_dir = options.experiment_dir
    dataset_dir = options.dataset_dir
    dataset = options.dataset
    enable_one_hot_features = options.enable_one_hot_features
    n_folds = options.n_folds
    pgm_n_parameters = options.pgm_n_parameters # Example: 25800 for mnist
    activation = options.activation
    n_epochs = options.n_epochs
    batch_size = options.batch_size

    assert taskid is not None and taskid >= 1
    assert n_epochs >= 1
    assert options.experiment_dir != ''
    assert activation in ['relu', 'sign']

    # We use different random seeds in our experiments
    if activation == 'relu':
        if enable_one_hot_features:
            rng = np.random.RandomState(seed=394128001)
        else:
            rng = np.random.RandomState(seed=278346541)
    elif activation == 'sign':
        if enable_one_hot_features:
            rng = np.random.RandomState(seed=825491023)
        else:
            rng = np.random.RandomState(seed=369658002)
    else:
        raise NotImplementedError('Activation \'{}\' not implemented'.format(activation))
    
    gridvals_pgm_bits_per_parameter = list(range(1, 9)) + [16,24,32,48,64]
    gridvals_n_bits_per_weight = list(range(1, 9))
    gridvals_n_layers = [2, 3, 4, 5]
    gridvals_enable_batchnorm = [False, True]
    gridvals_learning_rate_start = [3e-2, 3e-3, 3e-4]
    gridvals = [gridvals_pgm_bits_per_parameter,
                gridvals_n_bits_per_weight,
                gridvals_n_layers,
                gridvals_enable_batchnorm,
                gridvals_learning_rate_start]

    grid = list(product(*gridvals))
    n_grid = len(grid)
    n_seeds_per_parameter = 5
    n_jobs = n_grid * n_seeds_per_parameter

    rng_seeds = rng.randint(1, 1e9, size=(n_jobs,))

    if taskid > n_jobs:
        raise Exception('taskid {} too large (only {} defined)'.format(taskid, n_jobs))

    params = grid[(taskid - 1) // n_seeds_per_parameter]
    pgm_bits_per_parameter = params[0]
    n_bits_per_weight = params[1]
    n_layers = params[2]
    enable_batchnorm = params[3]
    learning_rate_start = params[4]

    rng_seed = rng_seeds[taskid - 1]
    tf.random.set_seed(rng_seed)

    print('-' * 80)
    print('taskid: {}/{}'.format(taskid, n_jobs))
    print('experiment_dir: \'{}\''.format(experiment_dir))
    print('dataset_dir: \'{}\''.format(dataset_dir))
    print('dataset \'{}\''.format(dataset))
    print('enable_one_hot_features: {}'.format(enable_one_hot_features))
    print('n_folds: {}'.format(n_folds))
    print('pgm_n_parameters: {}'.format(pgm_n_parameters))
    print('activation: \'{}\''.format(activation))
    print('n_epochs: {}'.format(n_epochs))
    print('batch_size: {}'.format(batch_size))
    print('-' * 80)
    print('Grid parameters:'.format(taskid))
    print('pgm_bits_per_parameter: {}'.format(pgm_bits_per_parameter))
    print('n_bits_per_weight: {}'.format(n_bits_per_weight))
    print('n_layers: {}'.format(n_layers))
    print('enable_batchnorm: {}'.format(enable_batchnorm))
    print('learning_rate_start: {}'.format(learning_rate_start))
    print('rng_seed: {}'.format(rng_seed))
    print('-' * 80)
    #-----------------------------------------------------------------------------------------------------------------------

    assert n_folds > 0
    stats_list = []
    for fold_idx in range(n_folds):
        print('Training fold {}/{}'.format(fold_idx + 1, n_folds))
        if n_folds > 1:
            tensorboard_logdir = '{}/tensorboard/experiment{:05d}/fold{:02d}'.format(experiment_dir, taskid, fold_idx + 1)
        else:
            tensorboard_logdir = '{}/tensorboard/experiment{:05d}'.format(experiment_dir, taskid)
        stats = run(dataset=dataset,
                    enable_one_hot_features=enable_one_hot_features,
                    pgm_n_parameters=pgm_n_parameters,
                    pgm_bits_per_parameter=pgm_bits_per_parameter,
                    n_bits_per_weight=n_bits_per_weight,
                    n_layers=n_layers,
                    activation=activation,
                    enable_batchnorm=enable_batchnorm,
                    n_epochs=n_epochs,
                    batch_size=batch_size,
                    learning_rate_start=learning_rate_start,
                    fold_idx=fold_idx,
                    dataset_dir=dataset_dir,
                    tensorboard_logdir=tensorboard_logdir)
        stats_list.append(stats)
    
    stats_stacked = {}
    for stat_entry in stats_list[0]:
        stats_stacked[stat_entry] = np.stack([stat[stat_entry] for stat in stats_list], axis=0)
    stats_stacked['experiment_parameters/taskid'] = taskid
    stats_stacked['experiment_parameters/pgm_bits_per_parameter'] = pgm_bits_per_parameter
    stats_stacked['experiment_parameters/n_bits_per_weight'] = n_bits_per_weight
    stats_stacked['experiment_parameters/enable_batchnorm'] = enable_batchnorm
    stats_stacked['experiment_parameters/learning_rate_start'] = learning_rate_start
    stats_stacked['experiment_parameters/rng_seed'] = rng_seed

    stats_stacked['call_arguments/taskid'] = taskid
    stats_stacked['call_arguments/experiment_dir'] = experiment_dir
    stats_stacked['call_arguments/dataset_dir'] = dataset_dir
    stats_stacked['call_arguments/dataset'] = dataset
    stats_stacked['call_arguments/enable_one_hot_features'] = enable_one_hot_features
    stats_stacked['call_arguments/n_folds'] = n_folds
    stats_stacked['call_arguments/pgm_n_parameters'] = pgm_n_parameters
    stats_stacked['call_arguments/activation'] = activation
    stats_stacked['call_arguments/n_epochs'] = n_epochs
    stats_stacked['call_arguments/batch_size'] = batch_size

    # Convert stats to numpy
    for key in stats_stacked:
        stats_stacked[key] = np.asarray(stats_stacked[key])
    np.savez_compressed('{}/stats/stats_{:05d}.npz'.format(experiment_dir, taskid), **stats_stacked)
    
    print('Job finished')


if __name__ == '__main__':
    main()
