import tensorflow as tf
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


@tf.custom_gradient
def round_positive_fixed_point(x, n_bits_integer=4, n_bits_fractional=4):
    '''
    Rounds values to the next fixed point value with the given number of integer and fractional bits.
    The smallest value is 0.0. The largest value is 2**n_bits_integer - 2*-n_bits_fractional.
    This function only supports positive values, i.e., no sign bit is involved. Note that n_bits_fractional
    can also be a negative value, e.g., a value of -1 would mean that only every second integer is used.
    '''
    min_value = 0.0
    max_value = 2.0 ** (n_bits_integer) - 2.0 ** (-n_bits_fractional)
    y = tf.math.round(x * (2.0 ** n_bits_fractional)) * (2.0 ** (-n_bits_fractional))
    y = tf.clip_by_value(y, min_value, max_value)
    def grad(dy):
        return dy, None, None
    return y, grad


@tf.custom_gradient
def max_ste(x):
    max_idx = tf.math.argmax(x, axis=0)
    max_one_hot = tf.one_hot(max_idx, x.shape[0], on_value=1.0, off_value=0.0, dtype=x.dtype)
    def grad(dy):
        return dy
    return max_one_hot, grad
