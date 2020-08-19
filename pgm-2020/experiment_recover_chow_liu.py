import tensorflow as tf
import numpy as np

from time import time
from optparse import OptionParser
from itertools import product

from BNCModels import TanBayesNetClassifier
from BNCModelsCommons import compute_chow_liu_structure_with_permutation


def train(model,
          n_classes,
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
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    learning_rate_end = learning_rate_start * 1e-3 # sweep learning rate over 3 orders of magnitude
    learning_rate_variable = tf.Variable(learning_rate_start, tf.float32) # suitable for real weights and relu activation
    learning_rate_schedule = np.logspace(np.log10(learning_rate_start), np.log10(learning_rate_end), n_epochs)

    gumbel_softmax_temperature_schedule = np.logspace(np.log10(gumbel_softmax_temperature_start), np.log10(gumbel_softmax_temperature_end), n_epochs)
    model.gumbel_softmax_temperature.assign(gumbel_softmax_temperature_start)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_variable)
    optimizer_structure = tf.keras.optimizers.Adam(learning_rate=learning_rate_structure)


    def nlogl_loss(labels, predictions):
        log_p_all = predictions # just renaming
        idx = tf.stack([tf.range(labels.shape[0]), labels], axis=-1)
        log_p_true = tf.gather_nd(log_p_all, idx)
        return -log_p_true


    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            predictions = model(images, True)
            loss = tf.reduce_mean(nlogl_loss(labels, predictions))
            if model.losses:
                loss += tf.add_n(model.losses)
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
        train_accuracy(labels, predictions)


    @tf.function
    def test_step(images, labels):
        predictions = model(images, False)
        test_loss(nlogl_loss(labels, predictions))
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

    #-----------------------------------------------------------------------------------------------------------------------
    # Remove singleton features and compute Chow Liu TAN Structure
    root_feature_idx = 0 # TODO: make this a parameter
    mi_graph_file = '{}/{}_mi.npz'.format(dataset_dir, dataset)
    mi_graph_dict = dict(np.load(mi_graph_file))
    mi_graph = mi_graph_dict['mi_graph_{}'.format(fold_idx + 1)]

    # Remove singleton features because they make the Chow-Liu structure ambiguous
    n_features_removed = 0
    for feature_idx in range(n_input_features - 1, -1, -1):
        if n_unique_features[feature_idx] == 1:
            print('Removing singleton feature {}'.format(feature_idx))
            x_train = np.delete(x_train, feature_idx, axis=1)
            x_test = np.delete(x_test, feature_idx, axis=1)
            mi_graph = np.delete(mi_graph, feature_idx, axis=0)
            mi_graph = np.delete(mi_graph, feature_idx, axis=1)
            del n_unique_features[feature_idx]
            n_features_removed += 1
    if n_features_removed > 0:
        n_input_features -= n_features_removed
        print('#input features: {} [after singleton removal]'.format(n_input_features))

    augmenting_features, feature_permutation = compute_chow_liu_structure_with_permutation(mi_graph, root_feature_idx)
    print('augmenting features: {}'.format(augmenting_features))

    # Permute features so that the Chow-Liu structure can be found within the left-to-right augmenting connections
    n_unique_features = np.asarray(n_unique_features)
    n_unique_features = n_unique_features[feature_permutation].tolist()
    x_train = x_train[:, feature_permutation]
    x_test = x_test[:, feature_permutation]
    
    # Permute the Chow-Liu augmenting features accordingly (for later comparison with the discovered structure)
    augmenting_features_new = []
    for feature_idx in range(n_input_features):
        feature_idx_new = feature_permutation[feature_idx]
        cond_idx_old = augmenting_features[feature_idx_new]
        cond_idx_new = feature_permutation.index(cond_idx_old)
        augmenting_features_new.append(cond_idx_new)
    augmenting_features = augmenting_features_new
    print('Augmenting features: {}'.format(augmenting_features))

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_ds = train_ds.shuffle(n_samples)
    train_ds = train_ds.batch(batch_size, drop_remainder=True)
    train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)

    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_ds = test_ds.batch(1000)
    test_ds = test_ds.prefetch(tf.data.experimental.AUTOTUNE)

    #-----------------------------------------------------------------------------------------------------------------------
    # Create the model
    model =  TanBayesNetClassifier(
            n_classes,
            n_unique_features,
            use_gumbel_straight_through=True,
            gumbel_softmax_temperature=gumbel_softmax_temperature_start)
    out = model(np.zeros((2,x_train.shape[1]), dtype=np.int32), True) # Build the model

    print('#' * 80)
    model.summary()
    print('#' * 80)

    stats = train(model=model,
                  n_classes=n_classes,
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

    structure_chow_liu = np.zeros((n_input_features, n_input_features), np.int32)
    for feature_idx in range(1, n_input_features):
        structure_chow_liu[feature_idx, augmenting_features[feature_idx]] = 1
    
    print('Chow-Liu Structure:')
    for feature_idx in range(n_input_features):
        print('Row {:3d}: {}'.format(feature_idx, structure_chow_liu[feature_idx].tolist()))

    print('Final Structure:')
    for feature_idx in range(n_input_features):
        print('Row {:3d}: {}'.format(feature_idx, structure[feature_idx].tolist()))

    print('Differences to Chow-Liu structure: [note: row 0 is always different]')
    for feature_idx in range(n_input_features):
        delta_row = structure[feature_idx] - structure_chow_liu[feature_idx]
        print('Row {:3d}: Delta: {:d} {}'.format(feature_idx, np.any(delta_row != 0), delta_row.tolist())) 
    
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
    parser.add_option('--learning-rate-structure', action='store', type='float', dest='learning_rate_structure', default=1e-3)
    parser.add_option('--gumbel-softmax-temperature-start', action='store', type='float', dest='gumbel_softmax_temperature_start', default=1e1)
    parser.add_option('--gumbel-softmax-temperature-end', action='store', type='float', dest='gumbel_softmax_temperature_end', default=1e-1)
    
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

    assert taskid is not None and taskid >= 1
    assert n_epochs >= 1
    assert options.experiment_dir != ''

    gridvals_learning_rate_start = [3e-2, 3e-3]
    gridvals = [gridvals_learning_rate_start]
    # TODO: Make root feature idx a random parameter

    grid = list(product(*gridvals))
    n_grid = len(grid)
    n_seeds_per_parameter = 1
    n_jobs = n_grid * n_seeds_per_parameter

    rng = np.random.RandomState(seed=265304870)
    rng_seeds = rng.randint(1, 1e9, size=(n_jobs,))

    if taskid > n_jobs:
        raise Exception('taskid {} too large (only {} defined)'.format(taskid, n_jobs))

    params = grid[(taskid - 1) // n_seeds_per_parameter]
    learning_rate_start = params[0]

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
    print('-' * 80)
    print('Grid parameters:'.format(taskid))
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
    stats_stacked['experiment_parameters/learning_rate_start'] = learning_rate_start
    stats_stacked['experiment_parameters/rng_seed'] = rng_seed

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

    # Convert stats to numpy
    for key in stats_stacked:
        stats_stacked[key] = np.asarray(stats_stacked[key])
    np.savez_compressed('{}/stats/stats_{:05d}.npz'.format(experiment_dir, taskid), **stats_stacked)
    
    print('Job finished')


if __name__ == '__main__':
    main()
