import tensorflow as tf
import numpy as np

from time import time
from optparse import OptionParser
from itertools import product

from tensorflow.keras import Model
from tensorflow.keras.layers import Softmax

from layers.DistDropout import DistDropout
from layers.DistDense import DistDense
from layers.DistConv2D import DistConv2D
from layers.DistPool2D import DistPool2D
from layers.DistBatchNormalization import DistBatchNormalization
from layers.DistReLU import DistReLU
from layers.DistSign import DistSign
from layers.DistFlatten import DistFlatten

from layers.weights.RealWeights import RealWeights
from layers.weights.QuantizedWeightsStraightThrough import QuantizedWeightsStraightThrough

from layers.ste import sign_ste_id, sign_dorefa, linear_quantizer, linear_quantizer_dorefa, tanh_quantizer_dorefa


class CNN(Model):
    def __init__(self,
                 layout,
                 conv_kernel_size,
                 n_bits_per_weight=32,
                 activation='relu',
                 regularize_weights_l1=0.0,
                 regularize_weights_l2=0.0,
                 enable_batchnorm=True,
                 batchnorm_momentum=0.99,
                 dropout_rate=None):
        super(CNN, self).__init__()
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
        self.conv_kernel_size = conv_kernel_size
        self.dropout = []
        self.linear = []
        self.maxpool = []
        self.batchnorm = []
        self.act = []
        self.flatten = DistFlatten()
        self.softmax = Softmax()
        for layer_idx, n_neurons in enumerate(layout):
            is_last_layer = layer_idx == len(layout) - 1
            self.dropout.append(create_dropout(dropout_rate[layer_idx]))
            if is_last_layer:
                self.linear.append(DistDense(n_neurons, create_weights(), use_bias=is_last_layer))
            else:
                self.linear.append(DistConv2D(n_neurons, (conv_kernel_size, conv_kernel_size), create_weights(), use_bias=is_last_layer))
                self.maxpool.append(DistPool2D('max', (2,2)))
                self.batchnorm.append(create_batchnorm())
                self.act.append(create_activation())

    
    def call(self, x, training):
        for layer_idx in range(len(self.layout)):
            is_last_layer = layer_idx == len(self.layout) - 1
            if is_last_layer:
                x = self.flatten(x)
            if self.dropout[layer_idx] is not None:
                x = self.dropout[layer_idx](x, training)
            x = self.linear[layer_idx](x, training)
            if is_last_layer:
                x = self.softmax(x)
            else:
                if self.batchnorm[layer_idx] is not None:
                    x = self.batchnorm[layer_idx](x, training)
                x = self.act[layer_idx](x, training)
                x = self.maxpool[layer_idx](x, training)
        return x


def createCNN(n_target_operations,
              n_classes,
              n_bits_per_weight,
              n_layers,
              n_input_channels,
              input_image_size,
              conv_kernel_size,
              enable_batchnorm,
              activation,
              n_bits_per_batchnorm=64,
              n_bits_per_bias=32,
              dropout_rate=0.0):
    '''
    Calculates the number of hidden units to match the number of bits for the parameters to a given target number of bits.
    n_target_bits: The target number of operations
    n_classes: The number of classes
    n_bits_per_weight: The number of bits used for each weight
    n_layers: The number of layers, i.e., the number of weight matrices.
    n_input_channels: The number of input channels
    input_image_size : The height and the width of the input image (we assume a square image size)
    conv_kernel_size: The size of the convolutional kernel (we use a square k \times k kernel)
    enable_batchnorm: Determines whether batch normalization should be used
    activation: The activation function ('relu' or 'sign')
    n_bits_per_batchnorm: The number of bits consumed by each neuron for batch normalization
    n_bis_per_bias: The number of bits consumed by each bias. A bias is used in the last layer.
    dropout_rate: A dropout rate used for all layers.
    '''
    input_image_size_1 = float(np.floor(input_image_size * 0.5))
    input_image_size_2 = float(np.floor(input_image_size_1 * 0.5))
    assert n_layers > 1
    if not enable_batchnorm:
        n_bits_per_batchnorm = 0

    if n_layers == 2:
        n_ops_per_channel = input_image_size ** 2.0 * conv_kernel_size ** 2.0 * n_input_channels + input_image_size_1 ** 2.0 * n_classes
        if enable_batchnorm:
            n_ops_per_channel += input_image_size ** 2.0
        n_channels_real = (n_target_operations - n_classes) / n_ops_per_channel
        layout = [max(round(n_channels_real), 1), n_classes]
    elif n_layers == 3:
        a = 2.0 * conv_kernel_size ** 2.0 * input_image_size_1 ** 2.0
        b = input_image_size ** 2.0 * conv_kernel_size ** 2.0 * n_input_channels + input_image_size_2 ** 2.0 * 2.0 * n_classes
        if enable_batchnorm:
            b += input_image_size ** 2.0 + input_image_size_1 ** 2.0 * 2.0
        c = n_classes - n_target_operations
        n_channels1 = (-b - (b ** 2.0 - 4.0 * a * c) ** 0.5) / (2.0 * a)
        n_channels2 = (-b + (b ** 2.0 - 4.0 * a * c) ** 0.5) / (2.0 * a)
        n_channels_real = max(n_channels1, n_channels2)
        layout = [max(round(n_channels_real), 1), max(round(2.0 * n_channels_real), 1)] + [n_classes]
    else:
        raise NotImplementedError('createCNN is only implemented for \'n_layers\' in [2,3]') 
    model = CNN(layout=layout,
         conv_kernel_size=conv_kernel_size,
         n_bits_per_weight=n_bits_per_weight,
         activation=activation,
         enable_batchnorm=enable_batchnorm,
         batchnorm_momentum=0.9,
         regularize_weights_l2=0.0,
         dropout_rate=[dropout_rate] * n_layers)

    # This is to verify our implementation
    if n_layers == 2:
        n_ops_total_real = (input_image_size ** 2.0 * conv_kernel_size ** 2.0 * n_input_channels * n_channels_real) # conv ops
        if enable_batchnorm:
            n_ops_total_real += (input_image_size ** 2.0 * n_channels_real) # batchnorm ops
        n_ops_total_real += (input_image_size_1 ** 2.0 * n_channels_real * n_classes) # last layer (dense)
        n_ops_total_real += n_classes # bias ops

        n_ops_total_integer = (input_image_size ** 2.0 * conv_kernel_size ** 2.0 * n_input_channels * layout[0]) # conv ops
        if enable_batchnorm:
            n_ops_total_integer += (input_image_size ** 2.0 * layout[0]) # batchnorm ops
        n_ops_total_integer += (input_image_size_1 ** 2.0 * layout[0] * n_classes) # last layer (dense)
        n_ops_total_integer += n_classes # bias ops

        n_bits_total_real = (conv_kernel_size ** 2.0 * n_input_channels * n_channels_real) * n_bits_per_weight # weights first layer
        n_bits_total_real += (input_image_size_1 ** 2.0 * n_channels_real * n_classes) * n_bits_per_weight # weights last layer
        n_bits_total_real += n_channels_real * n_bits_per_batchnorm # all batchnorm bits
        n_bits_total_real += n_classes * n_bits_per_bias # all bias bits

        n_bits_total_integer = (conv_kernel_size ** 2.0 * n_input_channels * layout[0]) * n_bits_per_weight # weights first layer
        n_bits_total_integer += (input_image_size_1 ** 2.0 * layout[0] * n_classes) * n_bits_per_weight # weights last layer
        n_bits_total_integer += layout[0] * n_bits_per_batchnorm # all batchnorm bits
        n_bits_total_integer += n_classes * n_bits_per_bias # all bias bits
    elif n_layers == 3:
        n_ops_total_real = input_image_size ** 2.0 * conv_kernel_size ** 2.0 * n_input_channels * n_channels_real # conv1 ops
        n_ops_total_real += input_image_size_1 ** 2.0 * conv_kernel_size ** 2.0 * n_channels_real ** 2.0 * 2.0 # conv2 ops
        n_ops_total_real += input_image_size_2 ** 2.0 * 2.0 * n_channels_real * n_classes # last layer (dense) ops
        if enable_batchnorm:
            n_ops_total_real += input_image_size ** 2.0 * n_channels_real # batchnorm1 ops
            n_ops_total_real += input_image_size_1 ** 2.0 * 2.0 * n_channels_real # batchnorm2 ops
        n_ops_total_real += n_classes # bias ops

        n_ops_total_integer = input_image_size ** 2.0 * conv_kernel_size ** 2.0 * n_input_channels * layout[0] # conv1 ops
        n_ops_total_integer += input_image_size_1 ** 2.0 * conv_kernel_size ** 2.0 * layout[0] * layout[1] # conv2 ops
        n_ops_total_integer += input_image_size_2 ** 2.0 * layout[1] * n_classes # last layer (dense) ops
        if enable_batchnorm:
            n_ops_total_integer += input_image_size ** 2.0 * layout[0] # batchnorm1 ops
            n_ops_total_integer += input_image_size_1 ** 2.0 * layout[1] # batchnorm2 ops
        n_ops_total_integer += n_classes # bias ops

        n_bits_total_real = (conv_kernel_size ** 2.0 * n_input_channels * n_channels_real) * n_bits_per_weight # weights first layer
        n_bits_total_real += (2.0 * n_channels_real ** 2.0 * conv_kernel_size ** 2.0) * n_bits_per_weight # intermediate layer
        n_bits_total_real += (input_image_size_2 ** 2.0 * 2.0 * n_channels_real * n_classes) * n_bits_per_weight # last layer
        n_bits_total_real += (3.0 * n_channels_real) * n_bits_per_batchnorm # all batchnorm bits
        n_bits_total_real += n_classes * n_bits_per_bias # all bias bits

        n_bits_total_integer = (conv_kernel_size ** 2.0 * n_input_channels * layout[0]) * n_bits_per_weight # weights first layer
        n_bits_total_integer += (layout[0] * layout[1] * conv_kernel_size ** 2.0) * n_bits_per_weight # intermediate layer
        n_bits_total_integer += (input_image_size_2 ** 2.0 * layout[1] * n_classes) * n_bits_per_weight # last layer
        n_bits_total_integer += (layout[0] + layout[1]) * n_bits_per_batchnorm # all batchnorm bits
        n_bits_total_integer += n_classes * n_bits_per_bias # all bias bits          

    print('Number of channels: {} (real)'.format(n_channels_real))
    print('Layout: {} (rounded #channels)'.format(layout))
    print('Number of target operations:                    {}'.format(n_target_operations))
    print('Number of total operations [real #channels]:    {}'.format(n_ops_total_real))
    print('Number of total operations [integer #channels]: {}'.format(n_ops_total_integer))
    print('Number of total bits used [real #channels]:     {}'.format(n_bits_total_real))
    print('Number of total bits used [integer #channels]:  {}'.format(n_bits_total_integer))
    
    if abs(n_ops_total_real - n_target_operations) > n_target_operations * 0.001:
        raise Exception('It seems we do something wrong')
        
    return model


def run(dataset,
        n_input_channels,
        input_image_size,
        pgm_n_ops_factor,
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
    x_train = dataset_dict['x_tr_fold{}'.format(fold_idx + 1)].astype(np.float32)
    y_train = dataset_dict['t_tr_fold{}'.format(fold_idx + 1)]
    x_test = dataset_dict['x_te_fold{}'.format(fold_idx + 1)].astype(np.float32)
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
    print('#classes: {}'.format(n_classes))
    
    # Reshape to NHWC
    x_train = x_train.reshape(-1, input_image_size, input_image_size, n_input_channels).transpose(0, 2, 1, 3)
    x_test = x_test.reshape(-1, input_image_size, input_image_size, n_input_channels).transpose(0, 2, 1, 3)

    # Uncomment the following to display some images
    # import matplotlib.pyplot as plt
    # x_train = x_train - np.min(x_train)
    # x_train = x_train / np.max(x_train)
    # for sample_idx in range(1000, 1020):
    #     f = plt.figure()
    #     plt.imshow(x_train[sample_idx, ..., 0], cmap='gray')
    #     plt.title('target: {}'.format(y_train[sample_idx]))
    #     plt.show()

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
    model = createCNN(n_target_operations=n_classes * n_input_features * pgm_n_ops_factor,
                      n_classes=n_classes,
                      n_bits_per_weight=n_bits_per_weight,
                      n_layers=n_layers,
                      n_input_channels=n_input_channels,
                      input_image_size=input_image_size,
                      conv_kernel_size=3,
                      enable_batchnorm=enable_batchnorm,
                      activation=activation)

    model(np.ones((2,x_train.shape[1],x_train.shape[2],x_train.shape[3]), dtype=np.float32), True) # Build the model

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
    parser.add_option('--input-image-size', action='store', type='int', dest='input_image_size', default=0)
    parser.add_option('--n-input-channels', action='store', type='int', dest='n_input_channels', default=1)
    parser.add_option('--n-folds', action='store', type='int', dest='n_folds', default=0)
    parser.add_option('--activation', action='store', type='string', dest='activation', default='relu')
    parser.add_option('--n-epochs', action='store', type='int', dest='n_epochs', default=300)
    parser.add_option('--batch-size', action='store', type='int', dest='batch_size', default=100)
    
    options, args = parser.parse_args()
    taskid = options.taskid
    experiment_dir = options.experiment_dir
    dataset_dir = options.dataset_dir
    dataset = options.dataset
    input_image_size = options.input_image_size
    n_input_channels = options.n_input_channels
    n_folds = options.n_folds
    activation = options.activation
    n_epochs = options.n_epochs
    batch_size = options.batch_size

    assert taskid is not None and taskid >= 1
    assert n_epochs >= 1
    assert options.experiment_dir != ''
    assert activation in ['relu', 'sign']

    # We use different random seeds in our experiments
    if activation == 'relu':
        rng = np.random.RandomState(seed=380044197)
    elif activation == 'sign':
        rng = np.random.RandomState(seed=278415630)
    else:
        raise NotImplementedError('Activation \'{}\' not implemented'.format(activation))
    
    gridvals_pgm_n_ops_factor = [2 ** e for e in range(1, 9)]
    gridvals_n_bits_per_weight = list(range(1, 9))
    gridvals_n_layers = [2, 3]
    gridvals_enable_batchnorm = [False, True]
    gridvals_learning_rate_start = [3e-2, 3e-3, 3e-4]
    gridvals = [gridvals_pgm_n_ops_factor,
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
    pgm_n_ops_factor = params[0]
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
    print('dataset: \'{}\''.format(dataset))
    print('input_image_size: {}'.format(input_image_size))
    print('n_input_channels: {}'.format(n_input_channels))
    print('n_folds: {}'.format(n_folds))
    print('activation: \'{}\''.format(activation))
    print('n_epochs: {}'.format(n_epochs))
    print('batch_size: {}'.format(batch_size))
    print('-' * 80)
    print('Grid parameters:'.format(taskid))
    print('pgm_n_ops_factor: {}'.format(pgm_n_ops_factor))
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
                    n_input_channels=n_input_channels,
                    input_image_size=input_image_size,
                    pgm_n_ops_factor=pgm_n_ops_factor,
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
    stats_stacked['experiment_parameters/pgm_n_ops_factor'] = pgm_n_ops_factor
    stats_stacked['experiment_parameters/n_bits_per_weight'] = n_bits_per_weight
    stats_stacked['experiment_parameters/enable_batchnorm'] = enable_batchnorm
    stats_stacked['experiment_parameters/learning_rate_start'] = learning_rate_start
    stats_stacked['experiment_parameters/rng_seed'] = rng_seed

    stats_stacked['call_arguments/taskid'] = taskid
    stats_stacked['call_arguments/experiment_dir'] = experiment_dir
    stats_stacked['call_arguments/dataset_dir'] = dataset_dir
    stats_stacked['call_arguments/dataset'] = dataset
    stats_stacked['call_arguments/input_image_size'] = input_image_size
    stats_stacked['call_arguments/n_input_channels'] = n_input_channels
    stats_stacked['call_arguments/n_folds'] = n_folds
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
