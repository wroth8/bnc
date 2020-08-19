import numpy as np


def build_mutual_information_graph(x_train, y_train, n_classes, n_unique_features, vectorize_classes=False, symmetric_graph=False, compute_diagonal=True):
    n_features = len(n_unique_features)
    _, prob_c = np.unique(y_train, return_counts=True)
    prob_c = prob_c / y_train.size
    
    graph = np.zeros((n_features, n_features), np.float64)
    if vectorize_classes:
        # We have tested the vectorize_classes implementation on several datasets and it turned out that it took longer on all of them.
        xy_stacked = np.concatenate([x_train, y_train[:, None]], axis=1)
        for feat_idx_1 in range(n_features):
            print('Processing feature {}/{}'.format(feature_idx_1 + 1, n_features))
            for feat_idx_2 in range(feat_idx_1):
                probs = np.zeros((n_unique_features[feat_idx_1], n_unique_features[feat_idx_2], n_classes), np.float64)
                unique_feat, counts = np.unique(xy_stacked[:, [feat_idx_1, feat_idx_2, -1]], axis=0, return_counts=True)
                probs[unique_feat[:, 0], unique_feat[:, 1], unique_feat[:, 2]] = counts
                assert xy_stacked.shape[0] == np.sum(probs)
                probs = probs / xy_stacked.shape[0] # p(f1, f2, c)
                probs_c = np.sum(probs, axis=(0, 1), keepdims=True)
                probs_feat_1_2 = probs / probs_c # p(f1, f2 | c)
                probs_feat_1 = np.sum(probs_feat_1_2, axis=1, keepdims=True) # p(f1 | c)
                probs_feat_2 = np.sum(probs_feat_1_2, axis=0, keepdims=True) # p(f2 | c)
                mutual_inf = probs * (np.log(probs_feat_1_2) - np.log(probs_feat_1) - np.log(probs_feat_2))
                mutual_inf[np.isnan(mutual_inf)] = 0.0
                mutual_inf_inc = np.sum(mutual_inf)
                assert mutual_inf_inc >= 0.0 or np.allclose(mutual_inf_inc, 0.0, atol=1e-15)
                mutual_inf_inc = max(mutual_inf_inc, 0.0)
                graph[feat_idx_1, feat_idx_2] = mutual_inf_inc
                if symmetric_graph:
                    graph[feat_idx_2, feat_idx_1] = mutual_inf_inc
            if compute_diagonal:
                probs = np.zeros((n_unique_features[feat_idx_1], n_classes), np.float64)
                unique_feat, counts = np.unique(xy_stacked[:, [feat_idx_1, -1]], axis=0, return_counts=True)
                probs[unique_feat[:, 0], unique_feat[:, 1]] = counts
                assert xy_stacked.shape[0] == np.sum(probs)
                probs = probs / xy_stacked.shape[0] # p(f1, c)
                probs_c = np.sum(probs, axis=0, keepdims=True)
                cond_entropy = -probs * np.log(probs / probs_c)
                cond_entropy[np.isnan(cond_entropy)] = 0.0
                cond_entropy_inc = np.sum(cond_entropy)
                assert cond_entropy_inc >= 0.0 or np.allclose(cond_entropy_inc, 0.0, atol=1e-15)
                cond_entropy_inc = max(cond_entropy_inc, 0.0)
                graph[feat_idx_1, feat_idx_1] = cond_entropy_inc
    else:
        for c in range(n_classes):
            print('Processing class {}/{}'.format(c + 1, n_classes))
            x_c = x_train[y_train == c]
            for feat_idx_1 in range(n_features):
                for feat_idx_2 in range(feat_idx_1):
                    probs = np.zeros((n_unique_features[feat_idx_1], n_unique_features[feat_idx_2]), np.float64)
                    unique_feat, counts = np.unique(x_c[:, [feat_idx_1, feat_idx_2]], axis=0, return_counts=True)
                    probs[unique_feat[:, 0], unique_feat[:, 1]] = counts
                    assert x_c.shape[0] == np.sum(probs)
                    probs = probs / x_c.shape[0] # p(f1, f2 | c)
                    probs_feat_1 = np.sum(probs, axis=1, keepdims=True) # p(f1 | c)
                    probs_feat_2 = np.sum(probs, axis=0) # p(f2 | c)
                    mutual_inf = probs * prob_c[c] * (np.log(probs) - np.log(probs_feat_1) - np.log(probs_feat_2))
                    mutual_inf[np.isnan(mutual_inf)] = 0.0
                    mutual_inf_inc = np.sum(mutual_inf)
                    assert mutual_inf_inc >= 0.0 or np.allclose(mutual_inf_inc, 0.0, atol=1e-15)
                    mutual_inf_inc = max(mutual_inf_inc, 0.0)
                    graph[feat_idx_1, feat_idx_2] += mutual_inf_inc
                    if symmetric_graph:
                        graph[feat_idx_2, feat_idx_1] = mutual_inf_inc
                if compute_diagonal:
                    probs = np.zeros((n_unique_features[feat_idx_1],), np.float64)
                    unique_feat, counts = np.unique(x_c[:, feat_idx_1], return_counts=True)
                    probs[unique_feat] = counts
                    assert x_c.shape[0] == np.sum(probs)
                    probs = probs / x_c.shape[0] # p(f1 | c)
                    cond_entropy = -probs * prob_c[c] * np.log(probs)
                    cond_entropy[np.isnan(cond_entropy)] = 0.0
                    cond_entropy_inc = np.sum(cond_entropy)
                    assert cond_entropy_inc >= 0.0 or np.allclose(cond_entropy_inc, 0.0, atol=1e-15)
                    cond_entropy_inc = max(cond_entropy_inc, 0.0)
                    graph[feat_idx_1, feat_idx_1] += cond_entropy_inc
    return graph


def main():
    configs = [{'name': 'letter', 'n_folds': 1},
               {'name': 'mnist', 'n_folds': 1},
               {'name': 'satimage', 'n_folds': 5},
               {'name': 'usps', 'n_folds': 1},
               ]
    
    for config in configs:
        dataset = config['name']
        n_folds = config['n_folds']
        print('Processing dataset \'{}\''.format(dataset))
        dataset_file = '{}.npz'.format(dataset)
        mi_graph_file = '{}_mi.npz'.format(dataset)
        dataset_dict = dict(np.load(dataset_file))
        
        mi_graphs = {}
        for fold_idx in range(n_folds):
            x_train = dataset_dict['x_tr_fold{}'.format(fold_idx + 1)].astype(np.int32) - 1
            y_train = dataset_dict['t_tr_fold{}'.format(fold_idx + 1)].astype(np.int32)
            y_test = dataset_dict['t_te_fold{}'.format(fold_idx + 1)].astype(np.int32)

            n_input_features = x_train.shape[1]
            n_classes = int(np.max(np.concatenate([y_train, y_test]) + 1))
            n_unique_features = []
            for feature_idx in range(n_input_features):
                unique_features = np.unique(x_train[:, feature_idx])
                assert unique_features.size == unique_features[-1] + 1
                n_unique_features.append(unique_features.size)

            mi_graph = build_mutual_information_graph(
                    x_train,
                    y_train,
                    n_classes,
                    n_unique_features,
                    vectorize_classes=False,
                    symmetric_graph=False,
                    compute_diagonal=True)
            mi_graphs['mi_graph_{}'.format(fold_idx + 1)] = mi_graph
        np.savez_compressed(mi_graph_file, **mi_graphs)


if __name__ == '__main__':
    main()
