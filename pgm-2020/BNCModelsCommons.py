import numpy as np

def init_ml_logits(init_data, shape, smoothing=1.0):
    assert len(init_data) == len(shape)
    assert len(init_data) in [1, 2, 3]

    init_data_stacked = np.stack(init_data, axis=1)
    vals, counts = np.unique(init_data_stacked, return_counts=True, axis=0)

    init_vals = np.zeros(shape, np.float32)
    if len(init_data) == 1:
        init_vals[vals[:, 0]] = counts
    elif len(init_data) == 2:
        init_vals[vals[:, 0], vals[:, 1]] = counts
    elif len(init_data) == 3:
        init_vals[vals[:, 0], vals[:, 1], vals[:, 2]] = counts
    else:
        assert False
    init_vals = init_vals + float(smoothing)
    # init_vals = init_vals / np.sum(init_vals) # division is not necessary
    return np.log(init_vals)


def compute_chow_liu_structure(mi_graph, root_feature_idx):
    n_features = mi_graph.shape[0]
    # Compute maximum spanning tree
    # Sort edges in descending order of their mutual information
    connected_components = np.arange(n_features) # stores for each feature to which component (partial tree) it is connected (initially each feature in separate component)
    n_edges = (n_features * (n_features - 1)) // 2
    edges = np.zeros((n_edges, 2), np.int32)
    weights = np.zeros((n_edges,), np.float64)
    edge_idx = 0
    for feat_idx_1 in range(n_features):
        for feat_idx_2 in range(feat_idx_1):
            edges[edge_idx, 0] = feat_idx_1
            edges[edge_idx, 1] = feat_idx_2
            weights[edge_idx] = mi_graph[feat_idx_1, feat_idx_2]
            edge_idx = edge_idx + 1
    assert edge_idx == n_edges
    sort_permutation = np.argsort(-weights)
    edges = edges[sort_permutation]
    weights = weights[sort_permutation]

    # Incrementally add largest edges that are not in the same connected component
    chow_liu_onehot = np.zeros((n_features, n_features), np.int32)
    for edge_idx in range(n_edges):
        edge = edges[edge_idx]
        component_1 = connected_components[edge[0]]
        component_2 = connected_components[edge[1]]
        if component_1 != component_2:
            chow_liu_onehot[edge[0], edge[1]] = 1
            connected_components[connected_components == component_2] = component_1 # merge connected components
    
    # Pick an element as root and make edges directed, i.e., do a breadth first search
    feature_permutation = np.zeros((n_features,), np.int32)
    augmenting_features_onehot = np.zeros((n_features, n_features), np.int32)
    feature_queue = [root_feature_idx]
    feature_permutation = []
    augmenting_features = np.zeros((n_features, ), np.int32)
    augmenting_features[root_feature_idx] = root_feature_idx
    while len(feature_queue) > 0:
        # Pick first element of the queue and process it
        cond_feature = feature_queue[0]
        feature_queue = feature_queue[1:]
        feature_permutation.append(cond_feature)

        # Fetch edges and remove them so that we do not go back again
        edges = chow_liu_onehot[cond_feature, :] + chow_liu_onehot[:, cond_feature] # edges are NOT stored symmetrically
        chow_liu_onehot[cond_feature, :] = 0
        chow_liu_onehot[:, cond_feature] = 0

        features = np.nonzero(edges)[0].tolist()
        feature_queue = feature_queue + features
        for feature_idx in features:
            augmenting_features_onehot[feature_idx, cond_feature] = 1
            augmenting_features[feature_idx] = cond_feature

    # augmenting_features_onehot = augmenting_features_onehot[feature_permutation, :]
    # augmenting_features_onehot = augmenting_features_onehot[:, feature_permutation]
    # print(augmenting_features_onehot)

    # feature_permutation:
    #   - This feature permutation can be used so that the optimal structure is within the restricted set of left-to-right-only connections
    # augmenting_features:
    #  - augmenting_features[idx] contains the conditional feature index of feature idx
    return augmenting_features.tolist()


def compute_chow_liu_structure_with_permutation(mi_graph, root_feature_idx):
    n_features = mi_graph.shape[0]
    # Compute maximum spanning tree
    # Sort edges in descending order of their mutual information
    connected_components = np.arange(n_features) # stores for each feature to which component (partial tree) it is connected (initially each feature in separate component)
    n_edges = (n_features * (n_features - 1)) // 2
    edges = np.zeros((n_edges, 2), np.int32)
    weights = np.zeros((n_edges,), np.float64)
    edge_idx = 0
    for feat_idx_1 in range(n_features):
        for feat_idx_2 in range(feat_idx_1):
            edges[edge_idx, 0] = feat_idx_1
            edges[edge_idx, 1] = feat_idx_2
            weights[edge_idx] = mi_graph[feat_idx_1, feat_idx_2]
            edge_idx = edge_idx + 1
    assert edge_idx == n_edges
    sort_permutation = np.argsort(-weights)
    edges = edges[sort_permutation]
    weights = weights[sort_permutation]

    # Incrementally add largest edges that are not in the same connected component
    chow_liu_onehot = np.zeros((n_features, n_features), np.int32)
    for edge_idx in range(n_edges):
        edge = edges[edge_idx]
        component_1 = connected_components[edge[0]]
        component_2 = connected_components[edge[1]]
        if component_1 != component_2:
            chow_liu_onehot[edge[0], edge[1]] = 1
            connected_components[connected_components == component_2] = component_1 # merge connected components
    
    # Pick an element as root and make edges directed, i.e., do a breadth first search
    feature_permutation = np.zeros((n_features,), np.int32)
    augmenting_features_onehot = np.zeros((n_features, n_features), np.int32)
    feature_queue = [root_feature_idx]
    feature_permutation = []
    augmenting_features = np.zeros((n_features, ), np.int32)
    augmenting_features[root_feature_idx] = root_feature_idx
    while len(feature_queue) > 0:
        # Pick first element of the queue and process it
        cond_feature = feature_queue[0]
        feature_queue = feature_queue[1:]
        feature_permutation.append(cond_feature)

        # Fetch edges and remove them so that we do not go back again
        edges = chow_liu_onehot[cond_feature, :] + chow_liu_onehot[:, cond_feature] # edges are NOT stored symmetrically
        chow_liu_onehot[cond_feature, :] = 0
        chow_liu_onehot[:, cond_feature] = 0

        features = np.nonzero(edges)[0].tolist()
        feature_queue = feature_queue + features
        for feature_idx in features:
            augmenting_features_onehot[feature_idx, cond_feature] = 1
            augmenting_features[feature_idx] = cond_feature

    # augmenting_features_onehot = augmenting_features_onehot[feature_permutation, :]
    # augmenting_features_onehot = augmenting_features_onehot[:, feature_permutation]
    # print(augmenting_features_onehot)

    # feature_permutation:
    #   - This feature permutation can be used so that the optimal structure is within the restricted set of left-to-right-only connections
    # augmenting_features:
    #  - augmenting_features[idx] contains the conditional feature index of feature idx
    return augmenting_features.tolist(), feature_permutation
