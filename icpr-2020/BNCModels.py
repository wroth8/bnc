import tensorflow as tf

from tensorflow.keras import Model
from BNCModelsCommons import init_ml_logits, round_positive_fixed_point, max_ste


class QuantizedBayesNetClassifier(Model):

    def __init__(self,
                 n_classes,
                 n_unique_features,
                 n_bits_integer,
                 n_bits_fractional,
                 init_data=None,
                 init_smoothing=1.0):
        super(QuantizedBayesNetClassifier, self).__init__()

        self.n_classes = n_classes
        self.n_unique_features = n_unique_features
        self.n_features = len(n_unique_features)

        self.n_bits_integer = float(n_bits_integer)
        self.n_bits_fractional = float(n_bits_fractional)

        if init_data is not None:
            assert isinstance(init_data, tuple)
            assert len(init_data) == 2
            if isinstance(init_data[0], tf.Tensor):
                assert isinstance(init_data[1], tf.Tensor)
                init_data = (init_data[0].numpy(), init_data[1].numpy())

        if init_data is None:
            logit_init = tf.random.uniform((n_classes,), minval=-0.1, maxval=0.1)
        else:
            logit_init = tf.convert_to_tensor(init_ml_logits([init_data[1]], (self.n_classes,), init_smoothing), dtype=tf.float32)
        self.class_logits = tf.Variable(
                logit_init,
                trainable=True,
                name='ClassLogits',
                constraint=None)
        
        self.feature_logits = []
        for feature_idx in range(self.n_features):
            if init_data is None:
                logit_init = tf.random.uniform((n_unique_features[feature_idx], n_classes), minval=-0.1, maxval=0.1)
            else:
                logit_init = tf.convert_to_tensor(
                        init_ml_logits([init_data[0][:, feature_idx], init_data[1]],
                        (n_unique_features[feature_idx], n_classes),
                        init_smoothing), dtype=tf.float32)
            self.feature_logits.append(tf.Variable(
                    logit_init,
                    trainable=True,
                    name='FeatureLogits{}'.format(feature_idx + 1),
                    constraint=None))


    def call(self, x, training, enable_quantization=True):
        class_logits_normalized = self.class_logits - tf.reduce_logsumexp(self.class_logits)
        if enable_quantization:
            class_logits_normalized = -round_positive_fixed_point(-class_logits_normalized, self.n_bits_integer, self.n_bits_fractional)
        out = class_logits_normalized
        for feature_idx in range(self.n_features):
            feature_logits_normalized = self.feature_logits[feature_idx] - tf.reduce_logsumexp(self.feature_logits[feature_idx], axis=0)
            if enable_quantization:
                feature_logits_normalized = -round_positive_fixed_point(-feature_logits_normalized, self.n_bits_integer, self.n_bits_fractional)
            out = out + tf.gather(feature_logits_normalized, x[:, feature_idx])
        return out


class QuantizedTanBayesNetClassifierFixed(Model):

    def __init__(self,
                 n_classes,
                 n_unique_features,
                 augmenting_features,
                 n_bits_integer=8,
                 n_bits_fractional=8,
                 init_data=None,
                 init_smoothing=1.0):
        super(QuantizedTanBayesNetClassifierFixed, self).__init__()

        self.n_classes = n_classes
        self.n_unique_features = n_unique_features
        self.n_features = len(n_unique_features)
        self.augmenting_features = augmenting_features

        self.n_bits_integer = float(n_bits_integer)
        self.n_bits_fractional = float(n_bits_fractional)

        if init_data is not None:
            assert isinstance(init_data, tuple)
            assert len(init_data) == 2
            if isinstance(init_data[0], tf.Tensor):
                assert isinstance(init_data[1], tf.Tensor)
                init_data = (init_data[0].numpy(), init_data[1].numpy())

        if init_data is None:
            logit_init = tf.random.uniform((n_classes,), minval=-0.1, maxval=0.1)
        else:
            logit_init = tf.convert_to_tensor(init_ml_logits([init_data[1]], (self.n_classes,), init_smoothing), dtype=tf.float32)
        self.class_logits = tf.Variable(
                logit_init,
                trainable=True,
                name='ClassLogits')

        self.feature_logits = []
        for feature_idx in range(self.n_features):
            cond_idx = augmenting_features[feature_idx]
            if cond_idx == feature_idx:
                if init_data is None:
                    logit_init = tf.random.uniform((n_unique_features[feature_idx], n_classes), minval=-0.1, maxval=0.1)
                else:
                    logit_init = tf.convert_to_tensor(
                            init_ml_logits([init_data[0][:, feature_idx], init_data[1]],
                            (n_unique_features[feature_idx], n_classes),
                            init_smoothing), dtype=tf.float32)
                self.feature_logits.append(tf.Variable(
                        logit_init,
                        trainable=True,
                        name='FeatureLogits{}'.format(feature_idx)))
            else:
                if init_data is None:
                    logit_init = tf.random.uniform((n_unique_features[feature_idx], n_unique_features[cond_idx], n_classes), minval=-0.1, maxval=0.1)
                else:
                    logit_init = tf.convert_to_tensor(
                            init_ml_logits([init_data[0][:, feature_idx], init_data[0][:, cond_idx], init_data[1]],
                            (n_unique_features[feature_idx], n_unique_features[cond_idx], n_classes),
                            init_smoothing), dtype=tf.float32)
                    
                self.feature_logits.append(tf.Variable(
                        logit_init,
                        trainable=True,
                        name='FeatureLogits{}_{}'.format(feature_idx, cond_idx)))


    def call(self, x, training, enable_quantization=False):
        class_logits_normalized = self.class_logits - tf.reduce_logsumexp(self.class_logits)
        if enable_quantization:
            class_logits_normalized = -round_positive_fixed_point(-class_logits_normalized, self.n_bits_integer, self.n_bits_fractional)
        out = class_logits_normalized
        
        for feature_idx in range(self.n_features):
            cond_idx = self.augmenting_features[feature_idx]
            feature_logits_normalized = self.feature_logits[feature_idx] - tf.reduce_logsumexp(self.feature_logits[feature_idx], axis=0)
            if enable_quantization:
                feature_logits_normalized = -round_positive_fixed_point(-feature_logits_normalized, self.n_bits_integer, self.n_bits_fractional)
            if cond_idx == feature_idx:
                out = out + tf.gather(feature_logits_normalized, x[:, feature_idx])
            else:
                out = out + tf.gather_nd(feature_logits_normalized, tf.stack([x[:, feature_idx], x[:, cond_idx]], axis=-1))

        return out # log-joint probability: log p(c,x1,...,xD) for all c \in C


class TanBayesNetClassifierStructurePenalty(Model):

    def __init__(self,
                 n_classes,
                 n_unique_features,
                 use_gumbel_straight_through=False,
                 gumbel_softmax_temperature=1.0,
                 gumbel_softmax_epsilon=1e-6,
                 normalize_structure_loss=False, # if true, subtracts the number of parameters of the naive Bayes model
                 init_data=None,
                 init_smoothing=1.0):
        super(TanBayesNetClassifierStructurePenalty, self).__init__()

        self.n_classes = n_classes
        self.n_unique_features = n_unique_features
        self.n_features = len(n_unique_features)

        self.use_gumbel_straight_through = use_gumbel_straight_through
        self.gumbel_softmax_epsilon = gumbel_softmax_epsilon
        self.gumbel_softmax_temperature = tf.Variable(
                gumbel_softmax_temperature,
                trainable=False,
                name='GumbelSoftmaxTemperature',
                dtype=tf.float32)

        if init_data is not None:
            assert isinstance(init_data, tuple)
            assert len(init_data) == 2
            if isinstance(init_data[0], tf.Tensor):
                assert isinstance(init_data[1], tf.Tensor)
                init_data = (init_data[0].numpy(), init_data[1].numpy())

        if init_data is None:
            logit_init = tf.random.uniform((n_classes,), minval=-0.1, maxval=0.1)
        else:
            logit_init = tf.convert_to_tensor(init_ml_logits([init_data[1]], (self.n_classes,), init_smoothing), dtype=tf.float32)
        self.class_logits = tf.Variable(
                logit_init,
                trainable=True,
                name='ClassLogits')

        self.feature_logits = []
        self.structure_logits = []
        self.augmented_feature_logits = []
        self.n_augmented_feature_params = []
        for feature_idx in range(self.n_features):
            # Logits if no augmenting edge is added, i.e., we only depend on the class.
            if init_data is None:
                logit_init = tf.random.uniform((n_unique_features[feature_idx], n_classes), minval=-0.1, maxval=0.1)
            else:
                logit_init = tf.convert_to_tensor(
                        init_ml_logits([init_data[0][:, feature_idx], init_data[1]],
                        (n_unique_features[feature_idx], n_classes),
                        init_smoothing), dtype=tf.float32)
            self.feature_logits.append(tf.Variable(
                    logit_init,
                    trainable=True,
                    name='FeatureLogits{}'.format(feature_idx)))

            # Structure parameters that determine the TAN structure (log-probabilities)
            if feature_idx == 0:
                self.structure_logits.append(None)
                self.augmented_feature_logits.append(None)
                if normalize_structure_loss:
                    self.n_augmented_feature_params.append(tf.constant([0], dtype=tf.float32))
                else:
                    self.n_augmented_feature_params.append(tf.constant([n_unique_features[feature_idx] * n_classes], dtype=tf.float32))
            else:
                structure_logit_init = tf.zeros((feature_idx + 1,), tf.float32)
                self.structure_logits.append(tf.Variable(
                        structure_logit_init,
                        trainable=True,
                        name='StructureLogits{}'.format(feature_idx)))
                self.augmented_feature_logits.append([])

                # Logits if augmening edges are used, i.e., we depend on the class and another previous feature.
                tmp_n_augmented_feature_params = []
                for cond_idx in range(feature_idx):
                    if init_data is None:
                        logit_init = tf.random.uniform((n_unique_features[feature_idx], n_unique_features[cond_idx], n_classes), minval=-0.1, maxval=0.1)
                    else:
                        logit_init = tf.convert_to_tensor(
                                init_ml_logits([init_data[0][:, feature_idx], init_data[0][:, cond_idx], init_data[1]],
                                (n_unique_features[feature_idx], n_unique_features[cond_idx], n_classes),
                                init_smoothing), dtype=tf.float32)
                    self.augmented_feature_logits[feature_idx].append(tf.Variable(
                            logit_init,
                            trainable=True,
                            name='AugmentedFeatureLogits{}_'.format(feature_idx, cond_idx)))
                    if normalize_structure_loss:
                        tmp_n_augmented_feature_params.append(n_unique_features[feature_idx] * n_unique_features[cond_idx] * n_classes - n_unique_features[feature_idx] * n_classes)
                    else:
                        tmp_n_augmented_feature_params.append(n_unique_features[feature_idx] * n_unique_features[cond_idx] * n_classes)
                if normalize_structure_loss:
                    tmp_n_augmented_feature_params.append(0)
                else:
                    tmp_n_augmented_feature_params.append(n_unique_features[feature_idx] * n_classes)
                self.n_augmented_feature_params.append(tf.constant(tmp_n_augmented_feature_params, dtype=tf.float32))


    def call(self, x, training):
        class_logits_normalized = self.class_logits - tf.reduce_logsumexp(self.class_logits)
        out = class_logits_normalized
        
        feature_logits_normalized = self.feature_logits[0] - tf.reduce_logsumexp(self.feature_logits[0], axis=0)
        out = out + tf.gather(feature_logits_normalized, x[:, 0])
        self.add_loss(tf.reduce_sum(self.n_augmented_feature_params[0]))
        
        for feature_idx in range(1, self.n_features):
            # Sample TAN connection using Gumbel softmax
            if training:
                noise_uniform = tf.random.uniform(self.structure_logits[feature_idx].shape)
                noise_gumbel = -tf.math.log(-tf.math.log(noise_uniform + self.gumbel_softmax_epsilon) + self.gumbel_softmax_epsilon) # Gumbel(0, 1) noise
                logits_sample = (self.structure_logits[feature_idx] + noise_gumbel) * (1.0 / self.gumbel_softmax_temperature)
                softmax_sample = tf.nn.softmax(logits_sample)
                if self.use_gumbel_straight_through:
                    softmax_sample = max_ste(softmax_sample)
                structure_probabilities = tf.nn.softmax(self.structure_logits[feature_idx])
                self.add_loss(tf.reduce_sum(structure_probabilities * self.n_augmented_feature_params[feature_idx]))
            else:
                softmax_sample = max_ste(self.structure_logits[feature_idx]) # this is actually a one-hot sample for testing
                self.add_loss(tf.reduce_sum(softmax_sample * self.n_augmented_feature_params[feature_idx]))

            feature_logits_normalized = self.feature_logits[feature_idx] - tf.reduce_logsumexp(self.feature_logits[feature_idx], axis=0)
            out = out + tf.gather(feature_logits_normalized, x[:, feature_idx]) * softmax_sample[-1]

            for cond_idx in range(feature_idx):
                feature_logits_normalized = self.augmented_feature_logits[feature_idx][cond_idx] - tf.reduce_logsumexp(self.augmented_feature_logits[feature_idx][cond_idx], axis=0)
                out = out + tf.gather_nd(feature_logits_normalized, tf.stack([x[:, feature_idx], x[:, cond_idx]], axis=-1)) * softmax_sample[cond_idx]

        return out # log-joint probability: log p(c,x1,...,xD) for all c \in C


class TanBayesNetClassifierSubsetStructurePenalty(Model):

    def __init__(self,
                 n_classes,
                 n_unique_features,
                 augmenting_features,
                 use_gumbel_straight_through=False,
                 gumbel_softmax_temperature=1.0,
                 gumbel_softmax_epsilon=1e-6,
                 normalize_structure_loss=False, # if true, subtracts the number of parameters of the naive Bayes model
                 init_data=None,
                 init_smoothing=1.0):
        super(TanBayesNetClassifierSubsetStructurePenalty, self).__init__()

        self.n_classes = n_classes
        self.n_unique_features = n_unique_features
        self.n_features = len(n_unique_features)
        self.augmenting_features = augmenting_features

        self.use_gumbel_straight_through = use_gumbel_straight_through
        self.gumbel_softmax_epsilon = gumbel_softmax_epsilon
        self.gumbel_softmax_temperature = tf.Variable(
                gumbel_softmax_temperature,
                trainable=False,
                name='GumbelSoftmaxTemperature',
                dtype=tf.float32)

        if init_data is not None:
            assert isinstance(init_data, tuple)
            assert len(init_data) == 2
            if isinstance(init_data[0], tf.Tensor):
                assert isinstance(init_data[1], tf.Tensor)
                init_data = (init_data[0].numpy(), init_data[1].numpy())

        if init_data is None:
            logit_init = tf.random.uniform((n_classes,), minval=-0.1, maxval=0.1)
        else:
            logit_init = tf.convert_to_tensor(init_ml_logits([init_data[1]], (self.n_classes,), init_smoothing), dtype=tf.float32)
        self.class_logits = tf.Variable(
                logit_init,
                trainable=True,
                name='ClassLogits')

        self.feature_logits = []
        self.structure_logits = []
        self.n_augmented_feature_params = []
        for feature_idx in range(self.n_features):
            self.feature_logits.append([])
            if feature_idx == 0:
                if init_data is None:
                    logit_init = tf.random.uniform((n_unique_features[0], n_classes), minval=-0.1, maxval=0.1)
                else:
                    logit_init = tf.convert_to_tensor(
                            init_ml_logits([init_data[0][:, 0], init_data[1]],
                            (n_unique_features[0], n_classes),
                            init_smoothing), dtype=tf.float32)
                self.feature_logits[0].append(tf.Variable(
                        logit_init,
                        trainable=True,
                        name='FeatureLogits0'))
                self.structure_logits.append(None)
                if normalize_structure_loss:
                    self.n_augmented_feature_params.append(tf.constant([0], dtype=tf.float32))
                else:
                    self.n_augmented_feature_params.append(tf.constant([n_unique_features[feature_idx] * n_classes], dtype=tf.float32))
            else:
                n_augmenting_features = len(self.augmenting_features[feature_idx])
                structure_logit_init = tf.zeros((n_augmenting_features,), tf.float32)
                self.structure_logits.append(tf.Variable(
                        structure_logit_init,
                        trainable=True,
                        name='StructureLogits{}'.format(feature_idx)))
                tmp_n_augmented_feature_params = []
                for cond_idx in self.augmenting_features[feature_idx]:
                    if cond_idx == feature_idx:
                        if init_data is None:
                            logit_init = tf.random.uniform((n_unique_features[feature_idx], n_classes), minval=-0.1, maxval=0.1)
                        else:
                            logit_init = tf.convert_to_tensor(
                                    init_ml_logits([init_data[0][:, feature_idx], init_data[1]],
                                    (n_unique_features[feature_idx], n_classes),
                                    init_smoothing), dtype=tf.float32)
                        var_name = 'FeatureLogits{}'.format(feature_idx)
                        if normalize_structure_loss:
                            tmp_n_augmented_feature_params.append(0)
                        else:
                            tmp_n_augmented_feature_params.append(n_unique_features[feature_idx] * n_classes)
                    else:
                        if init_data is None:
                            logit_init = tf.random.uniform((n_unique_features[feature_idx], n_unique_features[cond_idx], n_classes), minval=-0.1, maxval=0.1)
                        else:
                            logit_init = tf.convert_to_tensor(
                                    init_ml_logits([init_data[0][:, feature_idx], init_data[0][:, cond_idx], init_data[1]],
                                    (n_unique_features[feature_idx], n_unique_features[cond_idx], n_classes),
                                    init_smoothing), dtype=tf.float32)
                        var_name = 'FeatureLogits{}_{}'.format(feature_idx, cond_idx)
                        if normalize_structure_loss:
                            tmp_n_augmented_feature_params.append(n_unique_features[feature_idx] * n_unique_features[cond_idx] * n_classes - n_unique_features[feature_idx] * n_classes)
                        else:
                            tmp_n_augmented_feature_params.append(n_unique_features[feature_idx] * n_unique_features[cond_idx] * n_classes)
                    self.feature_logits[feature_idx].append(tf.Variable(
                            logit_init,
                            trainable=True,
                            name=var_name))
                self.n_augmented_feature_params.append(tf.constant(tmp_n_augmented_feature_params, dtype=tf.float32))


    def call(self, x, training):
        class_logits_normalized = self.class_logits - tf.reduce_logsumexp(self.class_logits)
        out = class_logits_normalized
        
        feature_logits_normalized = self.feature_logits[0][0] - tf.reduce_logsumexp(self.feature_logits[0][0], axis=0)
        out = out + tf.gather(feature_logits_normalized, x[:, 0])
        self.add_loss(tf.reduce_sum(self.n_augmented_feature_params[0]))
        
        for feature_idx in range(1, self.n_features):
            # Sample TAN connection using Gumbel softmax
            if training:
                noise_uniform = tf.random.uniform(self.structure_logits[feature_idx].shape)
                noise_gumbel = -tf.math.log(-tf.math.log(noise_uniform + self.gumbel_softmax_epsilon) + self.gumbel_softmax_epsilon) # Gumbel(0, 1) noise
                logits_sample = (self.structure_logits[feature_idx] + noise_gumbel) * (1.0 / self.gumbel_softmax_temperature)
                softmax_sample = tf.nn.softmax(logits_sample)
                if self.use_gumbel_straight_through:
                    softmax_sample = max_ste(softmax_sample)
                structure_probabilities = tf.nn.softmax(self.structure_logits[feature_idx])
                self.add_loss(tf.reduce_sum(structure_probabilities * self.n_augmented_feature_params[feature_idx]))
            else:
                softmax_sample = max_ste(self.structure_logits[feature_idx]) # this is actually a one-hot sample for testing
                self.add_loss(tf.reduce_sum(softmax_sample * self.n_augmented_feature_params[feature_idx]))

            for subset_idx, cond_idx in enumerate(self.augmenting_features[feature_idx]):
                feature_logits_normalized = self.feature_logits[feature_idx][subset_idx] - tf.reduce_logsumexp(self.feature_logits[feature_idx][subset_idx], axis=0)
                if cond_idx == feature_idx:
                    out = out + tf.gather(feature_logits_normalized, x[:, feature_idx]) * softmax_sample[subset_idx]
                else:
                    out = out + tf.gather_nd(feature_logits_normalized, tf.stack([x[:, feature_idx], x[:, cond_idx]], axis=-1)) * softmax_sample[subset_idx]
                    
        return out # log-joint probability: log p(c,x1,...,xD) for all c \in C
