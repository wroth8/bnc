import tensorflow as tf
import numpy as np

from time import time
from optparse import OptionParser
from itertools import product

from BNCModels import TanBayesNetClassifierFixed


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
            predictions = model(images, True)
            loss = tf.reduce_mean(hybrid_loss(labels, predictions, bnc_hybrid_tradeoff, bnc_hybrid_gamma, bnc_hybrid_eta))
            if model.losses:
                loss += tf.add_n(model.losses)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_accuracy(labels, predictions)


    @tf.function
    def test_step(images, labels):
        predictions = model(images, False)
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
    model =  TanBayesNetClassifierFixed(
            n_classes,
            n_unique_features,
            augmenting_features,
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
    parser.add_option('--init-ml', action='store_true', dest='init_ml', default=False)
    
    options, args = parser.parse_args()
    taskid = options.taskid
    experiment_dir = options.experiment_dir
    dataset_dir = options.dataset_dir
    dataset = options.dataset
    n_folds = options.n_folds
    n_epochs = options.n_epochs
    batch_size = options.batch_size
    init_ml = options.init_ml

    assert taskid is not None and taskid >= 1
    assert n_epochs >= 1
    assert options.experiment_dir != ''

    rng = np.random.RandomState(seed=265304870)

    gridvals_random_feature_permutation = list(range(5))
    gridvals_random_augmenting_features = list(range(10))
    gridvals_random_params = list(range(100))
    gridvals_learning_rate_start = [3e-2, 3e-3]
    gridvals = [gridvals_random_feature_permutation,
                gridvals_random_augmenting_features,
                gridvals_random_params,
                gridvals_learning_rate_start]

    grid = list(product(*gridvals))
    n_grid = len(grid)
    n_seeds_per_parameter = 1
    n_jobs = n_grid * n_seeds_per_parameter

    # The following lines of code might appear awkward. The purpose of these lines of code is to generate the same
    # random hyperparameters for different experiments to improve fairness.

    # The following calls are to evolve the rng state to get the same random_feature_permutation as in experiment020A
    rng_seeds = rng.randint(1, 1e9, size=(1000,))
    rng.uniform(low=0.0, high=3.0, size=(500,))
    rng.uniform(low=-1.0, high=2.0, size=(500,))
    rng.uniform(low=1.0, high=20.0, size=(500,))
    rng_seeds = np.concatenate([rng_seeds, rng.randint(1, 1e9, size=(4000,))])

    # Now the rng state should be where it is in experiment020A
    n_features = {
            'letter' : 16,
            'mnist' : 196,
            'satimage' : 36,
            'usps' : 256 }
    random_feature_permutation = [list(range(n_features[dataset]))]
    for _ in range(len(gridvals_random_feature_permutation) - 1):
        random_feature_permutation.append(rng.permutation(n_features[dataset]).tolist())

    # Sample random TAN structures. The following implementations ensures that for the first 5 setups a possible
    # structure from experiment 021A is sampled. The following structures are new.
    random_augmenting_features = []
    for _ in range(len(gridvals_random_augmenting_features)):
        feature_permutations = np.full((n_features[dataset], n_features[dataset]), -1, np.int32)
        for feature_idx in range(n_features[dataset]):
            feature_permutations[feature_idx, :feature_idx] = rng.permutation(feature_idx)
        random_augmenting_features.append(feature_permutations[:, 0].tolist())
    
    rng_seeds = np.concatenate([rng_seeds, rng.randint(1, 1e9, size=(n_jobs - rng_seeds.size,))])

    # Generate the same random hyperparameters as in experiment015A/B
    rng_experiment015A = np.random.RandomState(seed=361273804)
    rng_experiment015A.randint(1, 1e9, size=(9600,))
    random_bnc_hybrid_tradeoff = 10.0 ** rng_experiment015A.uniform(low=1.0, high=3.0, size=(len(gridvals_random_params),))
    random_bnc_hybrid_gamma = 10.0 ** rng_experiment015A.uniform(low=-1.0, high=2.0, size=(len(gridvals_random_params),))

    if taskid > n_jobs:
        raise Exception('taskid {} too large (only {} defined)'.format(taskid, n_jobs))

    params = grid[(taskid - 1) // n_seeds_per_parameter]
    random_feature_permutation_idx = params[0]
    random_augmenting_features_idx = params[1]
    random_param_idx = params[2]
    learning_rate_start = params[3]
    bnc_hybrid_tradeoff = random_bnc_hybrid_tradeoff[random_param_idx]
    bnc_hybrid_gamma = random_bnc_hybrid_gamma[random_param_idx]
    bnc_hybrid_eta = 10.0
    feature_permutation = random_feature_permutation[random_feature_permutation_idx]

    # Create random augmenting features. Note that the non-augmented connection depending only on the class is always
    # present. This one is encoded as the last feature, i.e., using feature_idx.
    augmenting_features_onehot = np.zeros((n_features[dataset], n_features[dataset]), np.int32)
    augmenting_features = []
    for feature_idx in range(n_features[dataset]):
        if feature_idx == 0:
            cond_feature_idx = 0
        else:
            cond_feature_idx = random_augmenting_features[random_augmenting_features_idx][feature_idx]
        augmenting_features_onehot[feature_idx, cond_feature_idx] = 1
        augmenting_features.append(cond_feature_idx)

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
    print('Grid parameters:'.format(taskid))
    print('random_feature_permutation_idx: {} [feature_permutation: {}]'.format(random_feature_permutation_idx, feature_permutation))
    print('random_augmenting_features_idx: {} [augmenting features: {}]'.format(random_augmenting_features_idx, augmenting_features))
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
                    augmenting_features=augmenting_features,
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
    stats_stacked['experiment_parameters/random_feature_permutation_idx'] = random_feature_permutation_idx
    stats_stacked['experiment_parameters/feature_permutation'] = feature_permutation
    stats_stacked['experiment_parameters/random_augmenting_features_idx'] = random_augmenting_features_idx
    stats_stacked['experiment_parameters/augmenting_features_onehot'] = augmenting_features_onehot
    stats_stacked['experiment_parameters/augmenting_features'] = augmenting_features
    stats_stacked['experiment_parameters/random_param_idx'] = random_param_idx
    stats_stacked['experiment_parameters/bnc_hybrid_tradeoff'] = bnc_hybrid_tradeoff
    stats_stacked['experiment_parameters/bnc_hybrid_gamma'] = bnc_hybrid_gamma
    stats_stacked['experiment_parameters/bnc_hybrid_eta'] = bnc_hybrid_eta
    stats_stacked['experiment_parameters/learning_rate_start'] = learning_rate_start
    stats_stacked['experiment_parameters/rng_seed'] = rng_seed

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
