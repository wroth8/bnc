import tensorflow as tf
import numpy as np

from time import time
from optparse import OptionParser
from itertools import product

from BNCModels import TanBayesNetClassifierStructurePenalty


def train(model,
          n_classes,
          bnc_hybrid_tradeoff,
          bnc_hybrid_gamma,
          bnc_hybrid_eta,
          structure_penalty,
          train_ds,
          test_ds,
          n_epochs,
          learning_rate_start,
          learning_rate_structure,
          gumbel_softmax_temperature_start,
          gumbel_softmax_temperature_end,
          tensorboard_logdir):
    #-----------------------------------------------------------------------------------------------------------------------
    # Create TF functions

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_loss_hybrid = tf.keras.metrics.Mean(name='train_loss_hybrid')
    train_loss_structure = tf.keras.metrics.Mean(name='train_loss_structure')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_loss_hybrid = tf.keras.metrics.Mean(name='test_loss_hybrid')
    test_loss_structure = tf.keras.metrics.Mean(name='test_loss_structure')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    learning_rate_end = learning_rate_start * 1e-3 # sweep learning rate over 3 orders of magnitude
    learning_rate_variable = tf.Variable(learning_rate_start, tf.float32) # suitable for real weights and relu activation
    learning_rate_schedule = np.logspace(np.log10(learning_rate_start), np.log10(learning_rate_end), n_epochs)

    gumbel_softmax_temperature_schedule = np.logspace(np.log10(gumbel_softmax_temperature_start), np.log10(gumbel_softmax_temperature_end), n_epochs)
    model.gumbel_softmax_temperature.assign(gumbel_softmax_temperature_start)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_variable)
    optimizer_structure = tf.keras.optimizers.Adam(learning_rate=learning_rate_structure)


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
            loss_hybrid = tf.reduce_mean(hybrid_loss(labels, predictions, bnc_hybrid_tradeoff, bnc_hybrid_gamma, bnc_hybrid_eta))
            loss_structure = tf.add_n(model.losses)
            loss = loss_hybrid + loss_structure * structure_penalty
        gradients = tape.gradient(loss, model.trainable_variables)

        updates_params = []
        updates_structure_params = []
        for gradient, trainable_variable in zip(gradients, model.trainable_variables):
            if 'Structure' not in trainable_variable.name:
                updates_params.append((gradient, trainable_variable))
            else:
                updates_structure_params.append((gradient, trainable_variable))
        optimizer.apply_gradients(updates_params)
        optimizer_structure.apply_gradients(updates_structure_params)

        train_loss(loss)
        train_loss_hybrid(loss_hybrid)
        train_loss_structure(loss_structure)
        train_accuracy(labels, predictions)

    @tf.function
    def test_step(images, labels):
        predictions = model(images, False)
        loss_hybrid = hybrid_loss(labels, predictions)
        loss_structure = tf.add_n(model.losses)
        loss = loss_hybrid + loss_structure * structure_penalty
        test_loss(loss)
        test_loss_hybrid(loss_hybrid)
        test_loss_structure(loss_structure)
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
    pyval_test_loss_structure = to_pyscalar(test_loss_structure.result())
    pyval_test_error = to_pyscalar(1.0 - test_accuracy.result())

    template = 'Epoch {:3d}/{:3d}, Loss: {:12e}, CE[TR]: {:8.5f}, Loss[TE]: {:e}, CE[TE]: {:8.5f}, Structure-Loss (most probable): {:8.1f}'
    print(template.format(0,
                          n_epochs,
                          float('nan'),
                          float('nan'),
                          pyval_test_loss,
                          pyval_test_error * 100.0,
                          pyval_test_loss_structure))
    test_loss.reset_states()
    test_loss_hybrid.reset_states()
    test_loss_structure.reset_states()
    test_accuracy.reset_states()
    
    stats = {'train_loss'           : [],
             'train_loss_hybrid'    : [],
             'train_loss_structure' : [],
             'train_error'          : [],
             'test_loss'            : [],
             'test_loss_hybrid'     : [],
             'test_loss_structure'  : [],
             'test_error'           : [],
             't_elapsed_train'      : [],
             't_elapsed_eval_test'  : [],
             'learning_rate'        : []}

    for epoch in range(n_epochs):
        learning_rate_variable.assign(learning_rate_schedule[epoch])
        model.gumbel_softmax_temperature.assign(gumbel_softmax_temperature_schedule[epoch])
        if epoch % 100 == 0:
            print('Current learning rate: {}'.format(learning_rate_variable))
            print('Current Gumbel softmax temperature: {}'.format(model.gumbel_softmax_temperature))

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
        pyval_train_loss_hybrid = to_pyscalar(train_loss_hybrid.result())
        pyval_train_loss_structure = to_pyscalar(train_loss_structure.result())
        pyval_train_error = to_pyscalar(1.0 - train_accuracy.result())
        pyval_test_loss = to_pyscalar(test_loss.result())
        pyval_test_loss_hybrid = to_pyscalar(test_loss_hybrid.result())
        pyval_test_loss_structure = to_pyscalar(test_loss_structure.result())
        pyval_test_error = to_pyscalar(1.0 - test_accuracy.result())
        pyval_learning_rate = to_pyscalar(learning_rate_variable)

        template = 'Epoch {:3d}/{:3d}, Loss: {:e}, CE[TR]: {:8.5f}, Loss[TE]: {:e}, CE[TE]: {:8.5f}, Structure-Loss: {:8.1f} (most probable: {:8.1f}) [t_elapsed: {:6.2f} seconds]'
        print(template.format(epoch+1,
                              n_epochs,
                              pyval_train_loss,
                              pyval_train_error * 100.0,
                              pyval_test_loss,
                              pyval_test_error * 100.0,
                              pyval_train_loss_structure,
                              pyval_test_loss_structure,
                              t_elapsed_train + t_elapsed_eval_test))

        # Write interesting values to tensorboard
        if enable_tensorboard_logging:
            with logwriter.as_default():
                tf.summary.scalar("train_loss", pyval_train_loss, step=epoch+1)
                tf.summary.scalar("train_loss_hybrid", pyval_train_loss_hybrid, step=epoch+1)
                tf.summary.scalar("train_loss_structure", pyval_train_loss_structure, step=epoch+1)
                tf.summary.scalar("train_error", pyval_train_error * 100.0, step=epoch+1)
                tf.summary.scalar("test_loss", pyval_test_loss, step=epoch+1)
                tf.summary.scalar("test_loss_hybrid", pyval_test_loss_hybrid, step=epoch+1)
                tf.summary.scalar("test_loss_structure", pyval_test_loss_structure, step=epoch+1)
                tf.summary.scalar("test_error", pyval_test_error * 100.0, step=epoch+1)
                tf.summary.scalar("t_elapsed_train", t_elapsed_train, step=epoch+1)
                tf.summary.scalar("t_elapsed_eval_test", t_elapsed_eval_test, step=epoch+1)
                tf.summary.scalar("learning_rate", pyval_learning_rate, step=epoch+1)
        # Write interesting values to stats
        stats['train_loss'].append(pyval_train_loss)
        stats['train_loss_hybrid'].append(pyval_train_loss_hybrid)
        stats['train_loss_structure'].append(pyval_train_loss_structure)
        stats['train_error'].append(pyval_train_error)
        stats['test_loss'].append(pyval_test_loss)
        stats['test_loss_hybrid'].append(pyval_test_loss_hybrid)
        stats['test_loss_structure'].append(pyval_test_loss_structure)
        stats['test_error'].append(pyval_test_error)
        stats['t_elapsed_train'].append(t_elapsed_train)
        stats['t_elapsed_eval_test'].append(t_elapsed_eval_test)
        stats['learning_rate'].append(pyval_learning_rate)   

        # Reset the metrics for the next epoch
        train_loss.reset_states()
        train_loss_hybrid.reset_states()
        train_loss_structure.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_loss_hybrid.reset_states()
        test_loss_structure.reset_states()
        test_accuracy.reset_states()

    # Convert stats to numpy
    for key in stats:
        stats[key] = np.asarray(stats[key])
    return stats


def run(dataset,
        feature_permutation,
        bnc_hybrid_tradeoff,
        bnc_hybrid_gamma,
        bnc_hybrid_eta,
        init_ml,
        structure_penalty,
        n_epochs,
        batch_size,
        learning_rate_start,
        learning_rate_structure,
        gumbel_softmax_temperature_start,
        gumbel_softmax_temperature_end,
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
    model =  TanBayesNetClassifierStructurePenalty(
            n_classes,
            n_unique_features,
            use_gumbel_straight_through=True,
            gumbel_softmax_temperature=gumbel_softmax_temperature_start,
            normalize_structure_loss=True,
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
                  structure_penalty=structure_penalty,
                  train_ds=train_ds,
                  test_ds=test_ds,
                  n_epochs=n_epochs,
                  learning_rate_start=learning_rate_start,
                  learning_rate_structure=learning_rate_structure,
                  gumbel_softmax_temperature_start=gumbel_softmax_temperature_start,
                  gumbel_softmax_temperature_end=gumbel_softmax_temperature_end,
                  tensorboard_logdir=tensorboard_logdir)

    structure_logits = np.zeros((n_input_features, n_input_features), np.float32)
    structure = np.zeros((n_input_features, n_input_features), np.int32)
    structure[0, 0] = 1
    for feature_idx in range(1, n_input_features):
        feature_structure_logits = model.structure_logits[feature_idx].numpy()
        structure_logits[feature_idx, :(feature_idx+1)] = feature_structure_logits
        structure[feature_idx, np.argmax(feature_structure_logits)] = 1
    stats['structure_logits'] = structure_logits
    stats['structure'] = structure

    print('Final Structure:')
    print(structure)

    return stats


def getStructureLearningSettingPGM2020(n_features, feature_permutation_idx):
    '''
    Generates the same random feature orderings as the TAN All experiment from the PGM-2020 paper.
    '''
    assert feature_permutation_idx < 5
    rng = np.random.RandomState(seed=265304870)

    # The following dummy calls are to evolve the rng state to get the same random_feature_permutations as used in the PGM-2020 experiment
    rng_seeds = rng.randint(1, 1e9, size=(1000,))
    rng.uniform(low=0.0, high=3.0, size=(500,))
    rng.uniform(low=-1.0, high=2.0, size=(500,))
    rng.uniform(low=1.0, high=20.0, size=(500,))
    rng_seeds = np.concatenate([rng_seeds, rng.randint(1, 1e9, size=(4000,))])

    # Generate random orderings
    random_feature_permutation = [list(range(n_features))] # first ordering corresponds to "keep the order as is"
    for _ in range(4):
        random_feature_permutation.append(rng.permutation(n_features).tolist())

    return random_feature_permutation[feature_permutation_idx]


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
    parser.add_option('--learning-rate-structure', action='store', type='float', dest='learning_rate_structure', default=1e-3)
    parser.add_option('--gumbel-softmax-temperature-start', action='store', type='float', dest='gumbel_softmax_temperature_start', default=1e1)
    parser.add_option('--gumbel-softmax-temperature-end', action='store', type='float', dest='gumbel_softmax_temperature_end', default=1e-2)
    parser.add_option('--init-ml', action='store_true', dest='init_ml', default=False)

    options, args = parser.parse_args()
    taskid = options.taskid
    experiment_dir = options.experiment_dir
    dataset_dir = options.dataset_dir
    dataset = options.dataset
    n_folds = options.n_folds
    n_epochs = options.n_epochs
    batch_size = options.batch_size
    learning_rate_structure = options.learning_rate_structure
    gumbel_softmax_temperature_start = options.gumbel_softmax_temperature_start
    gumbel_softmax_temperature_end = options.gumbel_softmax_temperature_end
    init_ml = options.init_ml

    assert taskid is not None and taskid >= 1
    assert n_epochs >= 1
    assert options.experiment_dir != ''

    rng = np.random.RandomState(seed=579102839)

    gridvals_structure_penalty = np.logspace(-7, 1, 41)
    gridvals_learning_rate_start = [3e-2, 3e-3]
    gridvals = [gridvals_structure_penalty,
                gridvals_learning_rate_start]

    grid = list(product(*gridvals))
    n_grid = len(grid)
    n_seeds_per_parameter = 5
    n_jobs = n_grid * n_seeds_per_parameter

    rng_seeds = rng.randint(1, 1e9, size=(n_jobs,))

    if taskid > n_jobs:
        raise Exception('taskid {} too large (only {} defined)'.format(taskid, n_jobs))

    params = grid[(taskid - 1) // n_seeds_per_parameter]
    structure_penalty = params[0]
    learning_rate_start = params[1]

    # Take best TAN structure learning setting from PGM-2020 "TAN All" experiment
    # feature_permutation:
    #   specifies the feature ordering
    if dataset == 'letter':
        # feature_permutation = [2, 13, 9, 11, 5, 8, 10, 4, 7, 3, 14, 15, 0, 6, 1, 12]
        feature_permutation = getStructureLearningSettingPGM2020(16, 3)
        bnc_hybrid_tradeoff = 301.1658679473318
        bnc_hybrid_gamma = 2.5668022666840273
        bnc_hybrid_eta = 1.9555211927748575
    elif dataset == 'satimage':
       # feature_permutation = [ 0,  1,  4, 19, 30, 25,  8, 24, 29, 18, 10, 21, 35, 15, 26,  2, 12,
       #                        28, 31, 33, 27, 22, 16, 17, 20, 14, 32,  5,  6, 23, 34, 13, 11,  9,
       #                         7,  3]
       feature_permutation = getStructureLearningSettingPGM2020(36, 2)
       bnc_hybrid_tradeoff = 5.475221171581703
       bnc_hybrid_gamma = 90.02910576471406
       bnc_hybrid_eta = 18.3157151058286
    else:
        # usps and mnist have too many input features and the tensorflow model would not compile
        raise NotImplementedError('Dataset \'{}\' not implemented'.format(dataset))

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
    print('learning_rate_structure: {}'.format(learning_rate_structure))
    print('gumbel_softmax_temperature_start: {}'.format(gumbel_softmax_temperature_start))
    print('gumbel_softmax_temperature_end: {}'.format(gumbel_softmax_temperature_end))
    print('init_ml: {}'.format(init_ml))
    print('-' * 80)
    print('Parameters from PGM-2020 experiment:')
    print('feature_permutation: {}'.format(feature_permutation))
    print('hybrid objective parameters: tradeoff: {}, gamma: {}, eta: {}]'.format(
                bnc_hybrid_tradeoff, bnc_hybrid_gamma, bnc_hybrid_eta))
    print('-' * 80)
    print('Grid parameters:'.format(taskid))
    print('structure_penalty: {}'.format(structure_penalty))
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
                    bnc_hybrid_tradeoff=bnc_hybrid_tradeoff,
                    bnc_hybrid_gamma=bnc_hybrid_gamma,
                    bnc_hybrid_eta=bnc_hybrid_eta,
                    init_ml=init_ml,
                    structure_penalty=structure_penalty,
                    n_epochs=n_epochs,
                    batch_size=batch_size,
                    learning_rate_start=learning_rate_start,
                    learning_rate_structure=learning_rate_structure,
                    gumbel_softmax_temperature_start=gumbel_softmax_temperature_start,
                    gumbel_softmax_temperature_end=gumbel_softmax_temperature_end,
                    fold_idx=fold_idx,
                    dataset_dir=dataset_dir,
                    tensorboard_logdir=tensorboard_logdir)
        stats_list.append(stats)
    
    stats_stacked = {}
    for stat_entry in stats_list[0]:
        stats_stacked[stat_entry] = np.stack([stat[stat_entry] for stat in stats_list], axis=0)
    stats_stacked['experiment_parameters/taskid'] = taskid
    stats_stacked['experiment_parameters/structure_penalty'] = structure_penalty
    stats_stacked['experiment_parameters/learning_rate_structure'] = learning_rate_structure

    stats_stacked['experiment_parameters_pgm2020/feature_permutation'] = feature_permutation
    stats_stacked['experiment_parameters_pgm2020/bnc_hybrid_tradeoff'] = bnc_hybrid_tradeoff
    stats_stacked['experiment_parameters_pgm2020/bnc_hybrid_gamma'] = bnc_hybrid_gamma
    stats_stacked['experiment_parameters_pgm2020/bnc_hybrid_eta'] = bnc_hybrid_eta

    stats_stacked['call_arguments/taskid'] = taskid
    stats_stacked['call_arguments/experiment_dir'] = experiment_dir
    stats_stacked['call_arguments/dataset_dir'] = dataset_dir
    stats_stacked['call_arguments/dataset'] = dataset
    stats_stacked['call_arguments/n_folds'] = n_folds
    stats_stacked['call_arguments/n_epochs'] = n_epochs
    stats_stacked['call_arguments/batch_size'] = batch_size
    stats_stacked['call_arguments/learning_rate_structure'] = learning_rate_structure
    stats_stacked['call_arguments/gumbel_softmax_temperature_start'] = gumbel_softmax_temperature_start
    stats_stacked['call_arguments/gumbel_softmax_temperature_end'] = gumbel_softmax_temperature_end
    stats_stacked['call_arguments/init_ml'] = init_ml

    # Convert stats to numpy
    for key in stats_stacked:
        stats_stacked[key] = np.asarray(stats_stacked[key])
    np.savez_compressed('{}/stats/stats_{:05d}.npz'.format(experiment_dir, taskid), **stats_stacked)
    
    print('Job finished')


if __name__ == '__main__':
    main()
