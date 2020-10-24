import tensorflow as tf
import numpy as np

from time import time
from optparse import OptionParser
from itertools import product

from BNCModels import QuantizedTanBayesNetClassifierFixed


def train(model,
          n_classes,
          bnc_hybrid_tradeoff,
          bnc_hybrid_gamma,
          bnc_hybrid_eta,
          train_ds,
          test_ds,
          n_epochs,
          learning_rate_start,
          tensorboard_logdir):
    #-----------------------------------------------------------------------------------------------------------------------
    # Create TF functions

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    learning_rate_end = learning_rate_start * 1e-3 # sweep learning rate over 3 orders of magnitude
    learning_rate_variable = tf.Variable(learning_rate_start, tf.float32) # suitable for real weights and relu activation
    learning_rate_schedule = np.logspace(np.log10(learning_rate_start), np.log10(learning_rate_end), n_epochs)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_variable)


    def hybrid_loss(labels, predictions, tradeoff=1.0, gamma=1.0, eta=10.0):
        log_p_all = predictions # just renaming

        # Compute generative negative log-likelihood cost
        idx = tf.stack([tf.range(labels.shape[0]), labels], axis=-1)
        log_p_true = tf.gather_nd(log_p_all, idx)
        loss_nlogl = -log_p_true

        # Compute discriminative margin cost
        loss_margin = gamma - tf.reshape(log_p_true, (-1, 1)) + log_p_all
        mask_true_class = tf.one_hot(labels, n_classes, on_value=-np.inf, off_value=0.0)
        loss_margin = loss_margin + mask_true_class
        loss_margin = tf.reduce_logsumexp(loss_margin * eta, axis=1) * (1.0 / eta)
        loss_margin = tf.nn.relu(loss_margin)
        
        return loss_nlogl + tradeoff * loss_margin


    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            predictions = model(images, True, enable_quantization=True)
            loss = tf.reduce_mean(hybrid_loss(labels, predictions, bnc_hybrid_tradeoff, bnc_hybrid_gamma, bnc_hybrid_eta))
            if model.losses:
                loss += tf.add_n(model.losses)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_accuracy(labels, predictions)


    @tf.function
    def test_step(images, labels):
        predictions = model(images, False, enable_quantization=True)
        test_loss(hybrid_loss(labels, predictions))
        test_accuracy(labels, predictions) # for metrics mean


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


def run(dataset,
        feature_permutation,
        augmenting_features,
        n_bits_integer,
        n_bits_fractional,
        bnc_hybrid_tradeoff,
        bnc_hybrid_gamma,
        bnc_hybrid_eta,
        init_ml,
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
    x_train = dataset_dict['x_tr_fold{}'.format(fold_idx + 1)].astype(np.int32) - 1
    y_train = dataset_dict['t_tr_fold{}'.format(fold_idx + 1)].astype(np.int32)
    x_test = dataset_dict['x_te_fold{}'.format(fold_idx + 1)].astype(np.int32) - 1
    y_test = dataset_dict['t_te_fold{}'.format(fold_idx + 1)].astype(np.int32)

    # Permute features (required for TAN structure learning)
    x_train = x_train[:, feature_permutation]
    x_test = x_test[:, feature_permutation]

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

    n_unique_features = []
    for feature_idx in range(n_input_features):
        unique_features = np.unique(x_train[:, feature_idx])
        assert unique_features.size == unique_features[-1] + 1
        n_unique_features.append(unique_features.size)
    print('Unique features:', n_unique_features)

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_ds = train_ds.shuffle(n_samples)
    train_ds = train_ds.batch(batch_size, drop_remainder=True)
    train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)

    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_ds = test_ds.batch(1000)
    test_ds = test_ds.prefetch(tf.data.experimental.AUTOTUNE)

    #-----------------------------------------------------------------------------------------------------------------------
    # Create the model
    model =  QuantizedTanBayesNetClassifierFixed(
            n_classes,
            n_unique_features,
            augmenting_features,
            n_bits_integer,
            n_bits_fractional,
            init_data=(x_train, y_train) if init_ml else None)
    out = model(np.zeros((2,x_train.shape[1]), dtype=np.int32), True) # Build the model
    
    print('#' * 80)
    model.summary()
    print('#' * 80)

    stats = train(model=model,
                  n_classes=n_classes,
                  bnc_hybrid_tradeoff=bnc_hybrid_tradeoff,
                  bnc_hybrid_gamma=bnc_hybrid_gamma,
                  bnc_hybrid_eta=bnc_hybrid_eta,
                  train_ds=train_ds,
                  test_ds=test_ds,
                  n_epochs=n_epochs,
                  learning_rate_start=learning_rate_start,
                  tensorboard_logdir=tensorboard_logdir)
    return stats


def main():
    #-----------------------------------------------------------------------------------------------------------------------
    # Set up experiment
    parser = OptionParser()
    parser.add_option('--taskid', action='store', type='int', dest='taskid', default=1)
    parser.add_option('--experiment-dir', action='store', type='string', dest='experiment_dir', default='.')
    parser.add_option('--dataset-dir', action='store', type='string', dest='dataset_dir', default='../datasets')
    parser.add_option('--dataset', action='store', type='string', dest='dataset', default=None)
    parser.add_option('--n-folds', action='store', type='int', dest='n_folds', default=0)
    parser.add_option('--n-epochs', action='store', type='int', dest='n_epochs', default=300)
    parser.add_option('--batch-size', action='store', type='int', dest='batch_size', default=100)
    parser.add_option('--taskid-other', action='store', type='int', dest='taskid_other', default=-1)
    parser.add_option('--init-ml', action='store_true', dest='init_ml', default=False)

    options, args = parser.parse_args()
    taskid = options.taskid
    experiment_dir = options.experiment_dir
    dataset_dir = options.dataset_dir
    dataset = options.dataset
    n_folds = options.n_folds
    n_epochs = options.n_epochs
    batch_size = options.batch_size
    taskid_other = options.taskid_other
    init_ml = options.init_ml

    assert taskid is not None and taskid >= 1
    assert n_epochs >= 1
    assert options.experiment_dir != ''

    rng = np.random.RandomState(seed=361273804)
    
    gridvals_n_bits_total = list(range(1, 9))
    gridvals_n_bits_integer = [1,2,3,4,5,6]
    gridvals_random_params = list(range(100))
    gridvals_learning_rate_start = [3e-2, 3e-3]
    gridvals = [gridvals_n_bits_total,
                gridvals_n_bits_integer,
                gridvals_random_params,
                gridvals_learning_rate_start]

    grid = list(product(*gridvals))
    n_grid = len(grid)
    n_seeds_per_parameter = 1
    n_jobs = n_grid * n_seeds_per_parameter

    rng_seeds = rng.randint(1, 1e9, size=(n_jobs,))
    random_bnc_hybrid_tradeoff = 10.0 ** rng.uniform(low=1.0, high=3.0, size=(len(gridvals_random_params),))
    random_bnc_hybrid_gamma = 10.0 ** rng.uniform(low=-1.0, high=2.0, size=(len(gridvals_random_params),))

    if taskid > n_jobs:
        raise Exception('taskid {} too large (only {} defined)'.format(taskid, n_jobs))

    params = grid[(taskid - 1) // n_seeds_per_parameter]
    n_bits_total = params[0]
    n_bits_integer = params[1]
    random_param_idx = params[2]
    learning_rate_start = params[3]
    n_bits_fractional = n_bits_total - n_bits_integer # can also be negative
    bnc_hybrid_tradeoff = random_bnc_hybrid_tradeoff[random_param_idx]
    bnc_hybrid_gamma = random_bnc_hybrid_gamma[random_param_idx]
    bnc_hybrid_eta = 10.0

    # Take best TAN structure from PGM-2020 "TAN Subset" experiment
    # feature_permutation:
    #   specifies the feature ordering
    # augmenting_features_per_fold:
    #   specifies the additional augmenting parent of each feature in each fold (no additional parent is encoded as
    #   structure[idx] == idx)
    #   augmenting_features_per_fold is with respect to the already permuted features according to feature_permutation
    if dataset == 'letter':
        feature_permutation = [2, 13, 9, 11, 5, 8, 10, 4, 7, 3, 14, 15, 0, 6, 1, 12]
        augmenting_features_per_fold = [
                [0, 0, 1, 2, 3, 2, 5, 5, 5, 8, 8, 8, 10, 6, 2, 10]]
    elif dataset == 'satimage':
        feature_permutation = [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
                               17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
                               34, 35]
        augmenting_features_per_fold = [
                [ 0,  0,  1,  1,  1,  4,  4,  5,  1,  8,  4,  9,  5,  3, 13, 12,  5,
                 15, 16, 17, 14, 11, 21,  4, 21, 11, 23, 25, 25, 28, 17, 29, 21,  0,
                 33, 33],
                [ 0,  0,  1,  1,  1,  3,  5,  5,  5,  4,  9,  9,  5,  5,  1, 12, 13,
                 15, 16, 17, 14, 11, 21,  1, 21, 11, 23, 13, 25, 28, 21, 29, 21,  0,
                 33, 33],
                [ 0,  0,  1,  1,  3,  3,  4,  5,  1,  8,  8,  9,  1, 12, 13,  5, 13,
                  7, 16, 17, 14, 11, 21,  1, 21, 23,  8, 12, 25, 28, 17, 16, 21,  0,
                 29, 33],
                [ 0,  0,  1,  1,  3,  4,  4,  5,  1,  8,  9,  9,  5, 12,  9,  5, 13,
                  4, 16, 17, 14, 11, 21, 13, 13, 11,  8, 13, 25, 28, 17, 29, 22,  0,
                 33, 20],
                [ 0,  0,  1,  1,  1,  3,  5,  4,  5,  8,  8,  9,  5, 11,  1, 12,  5,
                  4, 16, 17, 15, 11, 20,  1, 21, 23, 23, 25, 25, 20, 17, 29, 21, 19,
                 33, 24]]
    elif dataset == 'usps':
        feature_permutation = [130, 180, 132, 190, 199,  11,  28,  39, 107,  79,  47,  55,   8,
                               136,   2, 228, 235,   4,  84,  33,  94, 145,  56, 246, 177,  91,
                                82, 238,  46, 223, 157, 248, 147, 220, 129, 116, 217,  27, 163,
                               187, 200,  12,   1,  18, 214,  98, 158, 245, 171,  40,  85, 110,
                               164, 117, 185, 229, 224,  90, 251, 156,   7, 197, 153, 118, 114,
                               109,  80, 198,  63, 137, 189, 176,  21,  76,  53, 207, 120, 150,
                                20, 188, 166,  13, 139, 154,  72, 232, 178, 254, 121, 182, 249,
                               243, 125, 186, 173,   6,  30, 212,  51,  14,  78, 106,  32,  37,
                                48,  57, 144,  17, 143, 208, 222,  42, 103, 244, 191, 209, 195,
                                81, 113,  77, 210, 100,  58, 105, 123,  65, 192,  52, 252, 204,
                                19,  73, 242, 253, 213, 126, 119, 111, 101,  22, 181, 255, 142,
                               219,  92,  41,  23, 102,  89,  99, 112,   0,  87, 233, 215,  25,
                               172, 149, 221, 183, 241, 218,  38, 161, 127, 162,  93, 205,  61,
                                 9, 170,  29, 160, 135,  69, 227, 179,  34, 165, 151,  75, 206,
                                95, 237, 167, 128,  71, 141,  15, 146,   5,  96,  64, 169, 225,
                                60,  16, 104, 140, 201, 193, 175,  35, 152,  97, 184, 231,  26,
                                10,  54, 133, 134,  88,   3, 115, 236,  86, 168,  74,  43,  68,
                               131, 122, 124,  83, 240,  44,  70,  50, 194, 203, 239,  45, 196,
                                49, 211,  66, 226,  31, 155,  67, 108, 159, 250,  59, 148, 174,
                               216, 247, 202,  24,  36, 138, 230,  62, 234]
        augmenting_features_per_fold = [
                [  0,   0,   1,   2,   2,   1,   3,   2,   6,   6,   6,   5,   7,
                   6,   7,   6,   3,  14,   2,  15,  16,   2,  20,  16,  19,  24,
                   3,  16,  25,  19,   6,  16,  21,   3,  18,  22,  16,  25,   2,
                  37,  30,  20,  37,  24,  37,   7,  10,  41,  13,  41,  17,  34,
                  44,  22,  30,  24,  24,  50,  21,  46,  21,  53,   4,  37,  49,
                  30,  20,  24,  53,  55,  68,  24,  18,   5,  11,  37,  25,  76,
                  47,  71,  69,  74,  40,  82,  52,  80,  67,  42,  49,  20,  54,
                  38,  91,  28,  81,  16,  68,  24,  46,  15,  72,  20,  25,  99,
                  95,  84,  34,  17,  59,  97, 100, 101,   7,  67,  18,  97,  38,
                  50,  99,  46,  45,  66,  27,  52,  69,  91,  55,  72, 114, 109,
                  19,  53,  55,  20,  35,  73,  64,  85,  32, 125,  44, 109,  28,
                 125,  37, 136, 112,  74, 145,  96, 149,  91, 145, 147, 100,  80,
                  99, 106, 140, 157,  52, 158, 127, 121, 129, 105, 131,  85,  25,
                  45,  94, 147,  87,  24,  26, 115, 116, 175,  97, 153,  88, 131,
                  50, 163,  15,  57, 124,  67,  81,  20,  22, 138, 165,  85, 191,
                 123, 138, 147,  49, 154, 183,  39, 194, 140, 126,  23, 159, 136,
                   7,  11,  89, 155, 180, 206,  34,   6,   7, 134, 197, 178,  49,
                 189,  50, 216, 145,  56,  10,  49, 225, 132,  83, 126,  20, 163,
                 109, 109,  15, 126, 134, 216, 130, 200,  72, 110, 218,  21,  93,
                 233, 104, 247, 186, 167,  62, 100, 244, 122]]
    elif dataset == 'mnist':
        feature_permutation = [158,  87, 125,  78, 118, 121,  12,   4,  85,  91, 120, 184, 107,
                                30,  11,   2, 174, 156, 136,  81,  20, 144,  56, 190, 186,  46,
                               172, 182, 159, 135, 113,  51, 100,  48, 164, 111,  28,  40, 177,
                                98,  17, 116,  90,  63,  72, 151, 106, 128, 101,  89, 179,  19,
                               163,  94, 147,  73,  82,   8, 152, 132,  52,  55, 117, 140, 160,
                               126, 178,  22,  79, 153,  14, 180, 146, 102,  93, 114, 127,  84,
                               194,  25, 142,  21, 181, 187, 188,  38,  65, 143, 129, 166,  77,
                               103, 110,  61,   9, 167,  42,  29,  32,   7,  69, 150, 105,  34,
                                37,  23, 192,  27,  95, 149,  39,  92, 109,  57,   0,  71, 176,
                                15, 173,   5,  96, 119, 145,  64,  41, 161,  60,  16, 189,  35,
                                97,  26,  10, 170, 133, 134,  88, 185, 115, 168,  86,  74,  43,
                               191, 195, 122, 124,  83, 157, 171,  70,  50,  45,  49,  66,  31,
                               155,  67, 108, 162,  59, 148,  24,  36, 138,  62,  33,  53, 154,
                                44,  54, 169,   6,  18, 130, 104, 193, 123,  76,   1,  80, 137,
                               183, 165,  47,   3,  99, 112, 141, 175,  58,  68,  75,  13, 139, 131]
        augmenting_features_per_fold = [
                [  0,   0,   0,   0,   3,   2,   3,   4,   1,   3,   9,   8,   9,
                   3,   9,  10,   4,   8,  10,   8,   9,   0,   4,   3,   2,  13,
                  16,   4,   9,  18,  24,   9,  30,  25,  29,   3,  35,  17,  32,
                  28,  30,  26,  31,   9,  32,  42,  18,  21,  47,  48,  43,  49,
                  41,  35,   4,  48,  19,  43,  30,  16,   9,  32,  21,  43,  54,
                  30,  25,  42,  20,  56,  68,  42,  62,  41,  66,  58,  73,  60,
                  46,  48,   8,  31,  62,  52,  52,  74,  85,  76,  48,  80,  43,
                  21,  56,  25,  54,  74,   9,  30,  42,  33,  37,  29,  46,  31,
                  93,  68,  59,  59,  74,   9,  73,  84, 109,  44, 102, 103,  72,
                  90,  62, 103, 100,  97,  28,  85,  91, 109,   8, 125,  43, 104,
                  34,  90,  86,  45, 109,  64,  98,  26,  73,  46,  42,  17, 141,
                  60,  92,  74, 120,   2,  76,  91, 123,  46,  98,  21, 100,  54,
                 111, 147,  44,  30, 155,   4,  98, 113, 100,  38, 109,  32, 115,
                  76, 151,  86,  38, 102,  41,  42,  54, 113,  78, 153,  60, 159,
                  73, 151,  93, 122,   1, 186,  41,  93,  33,   3, 141, 102, 109,  11]]
    else:
        raise NotImplementedError('Dataset \'{}\' not implemented'.format(dataset))
    feature_permutation = np.asarray(feature_permutation, np.int32)

    rng_seed = rng_seeds[taskid - 1]
    tf.random.set_seed(rng_seed)

    print('-' * 80)
    print('taskid: {}/{}'.format(taskid, n_jobs))
    print('experiment_dir: \'{}\''.format(experiment_dir))
    print('dataset_dir: \'{}\''.format(dataset_dir))
    print('dataset \'{}\''.format(dataset))
    print('n_folds: {}'.format(n_folds))
    print('n_epochs: {}'.format(n_epochs))
    print('batch_size: {}'.format(batch_size))
    print('init_ml: {}'.format(init_ml))
    print('-' * 80)
    print('TAN structure:')
    print('feature_permutation: {}'.format(feature_permutation))
    print('augmenting features:')
    for fold_idx in range(n_folds):
        print('  Fold {:2d}: {}'.format(fold_idx + 1, augmenting_features_per_fold[fold_idx]))
    print('-' * 80)
    print('Grid parameters:'.format(taskid))
    print('n_bits_total: {}'.format(n_bits_total))
    print('n_bits_integer: {} [n_bits_fractional: {}]'.format(n_bits_integer, n_bits_fractional))
    print('random_param_idx: {} [tradeoff: {}, gamma: {}, eta: {}]'.format(
            random_param_idx, bnc_hybrid_tradeoff, bnc_hybrid_gamma, bnc_hybrid_eta))
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
                    feature_permutation=feature_permutation,
                    augmenting_features=augmenting_features_per_fold[fold_idx],
                    n_bits_integer=n_bits_integer,
                    n_bits_fractional=n_bits_fractional,
                    bnc_hybrid_tradeoff=bnc_hybrid_tradeoff,
                    bnc_hybrid_gamma=bnc_hybrid_gamma,
                    bnc_hybrid_eta=bnc_hybrid_eta,
                    init_ml=init_ml,
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
    stats_stacked['experiment_parameters/n_bits_total'] = n_bits_total
    stats_stacked['experiment_parameters/n_bits_integer'] = n_bits_integer
    stats_stacked['experiment_parameters/n_bits_fractional'] = n_bits_fractional
    stats_stacked['experiment_parameters/random_param_idx'] = random_param_idx
    stats_stacked['experiment_parameters/bnc_hybrid_tradeoff'] = bnc_hybrid_tradeoff
    stats_stacked['experiment_parameters/bnc_hybrid_gamma'] = bnc_hybrid_gamma
    stats_stacked['experiment_parameters/bnc_hybrid_eta'] = bnc_hybrid_eta
    stats_stacked['experiment_parameters/learning_rate_start'] = learning_rate_start
    stats_stacked['experiment_parameters/rng_seed'] = rng_seed

    stats_stacked['experiment_parameters/feature_permutation'] = feature_permutation
    stats_stacked['experiment_parameters/augmenting_features_per_fold'] = np.asarray(augmenting_features_per_fold)

    stats_stacked['call_arguments/taskid'] = taskid
    stats_stacked['call_arguments/experiment_dir'] = experiment_dir
    stats_stacked['call_arguments/dataset_dir'] = dataset_dir
    stats_stacked['call_arguments/dataset'] = dataset
    stats_stacked['call_arguments/n_folds'] = n_folds
    stats_stacked['call_arguments/n_epochs'] = n_epochs
    stats_stacked['call_arguments/batch_size'] = batch_size
    stats_stacked['call_arguments/init_ml'] = init_ml

    # Convert stats to numpy
    for key in stats_stacked:
        stats_stacked[key] = np.asarray(stats_stacked[key])
    np.savez_compressed('{}/stats/stats_{:05d}.npz'.format(experiment_dir, taskid), **stats_stacked)
    
    print('Job finished')


if __name__ == '__main__':
    main()
