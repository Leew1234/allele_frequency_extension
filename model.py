"""Allele-frequency-aware 1D U-Net derived from the project's cnn_batchnorm.py.

Two output heads are available:

  softmax     3 channels: P(no variant), P(rare), P(common). Trained with
              sparse categorical cross entropy; ignored sites get weight 0.
  factorized  2 channels: P(variant) and P(common | variant). The first
              channel is the same target as the original binary model, so it
              can be compared with it directly. The second is trained only at
              variant sites.

  spectrum    1 + K channels: P(variant) and a K-way softmax
              P(class | variant) over the frequency-spectrum classes of
              spectrum.py (singleton, ultra-rare, rare, common for K = 4).
              The first channel is again the original model's target; the
              softmax is trained only at variant sites. Analysis plan v2.

softmax and factorized are converted to the same 3-class probabilities for
prediction/scoring, spectrum to 1 + K class probabilities (P(none) first).
The NumPy helpers at the bottom do that conversion and undo class weighting.
"""
import numpy as np

HEADS = ('softmax', 'factorized', 'spectrum')
_EPS = 1e-7


def factorized_loss(variant_pos_weight=1.0, common_pos_weight=1.0):
    """Masked binary cross entropies for the factorized head.

    y_true holds integer labels: -1 ignore, 0 none, 1 rare, 2 common, and
    3 = variant whose frequency class should not be learned here (used by
    train.py --mask-scope common outside the neutral mask). -1 sites contribute
    nothing; the common/rare term is evaluated only at labels 1 and 2.
    """
    import tensorflow as tf

    def loss(y_true, y_pred):
        y = tf.reshape(tf.cast(y_true, tf.float32), tf.shape(y_pred)[:2])
        p = tf.clip_by_value(tf.cast(y_pred, tf.float32), _EPS, 1.0 - _EPS)
        pv, pc = p[..., 0], p[..., 1]
        valid = tf.cast(y >= 0, tf.float32)
        variant = tf.cast(y >= 1, tf.float32)
        freq_known = variant * tf.cast(y <= 2.5, tf.float32)
        common = tf.cast(tf.abs(y - 2.0) < 0.5, tf.float32)
        l_var = -(variant_pos_weight * variant * tf.math.log(pv)
                  + (1.0 - variant) * tf.math.log(1.0 - pv)) * valid
        l_com = -(common_pos_weight * common * tf.math.log(pc)
                  + (1.0 - common) * tf.math.log(1.0 - pc)) * freq_known
        return (tf.reduce_sum(l_var) + tf.reduce_sum(l_com)) / tf.maximum(tf.reduce_sum(valid), 1.0)

    return loss


def spectrum_loss(n_freq=4, variant_pos_weight=1.0):
    """Masked losses for the spectrum head.

    y_true holds integer labels: -1 ignore, 0 none, 1..n_freq the frequency
    class of a variant, and any larger value (spectrum.UNKNOWN) = a variant
    whose class should not be learned here (train.py --mask-scope common
    outside the neutral mask). The variant term is a binary cross entropy
    over all valid sites; the class term is a categorical cross entropy
    evaluated only at variants with a known class. Normalized by the number
    of valid sites, like factorized_loss.
    """
    import tensorflow as tf

    def loss(y_true, y_pred):
        y = tf.reshape(tf.cast(y_true, tf.float32), tf.shape(y_pred)[:2])
        p = tf.clip_by_value(tf.cast(y_pred, tf.float32), _EPS, 1.0 - _EPS)
        pv, pc = p[..., 0], p[..., 1:]
        valid = tf.cast(y >= 0, tf.float32)
        variant = tf.cast(y >= 1, tf.float32)
        known = variant * tf.cast(y <= n_freq + 0.5, tf.float32)
        index = tf.clip_by_value(tf.cast(tf.round(y), tf.int32) - 1, 0, n_freq - 1)
        l_var = -(variant_pos_weight * variant * tf.math.log(pv)
                  + (1.0 - variant) * tf.math.log(1.0 - pv)) * valid
        l_cls = -tf.reduce_sum(tf.one_hot(index, n_freq) * tf.math.log(pc), axis=-1) * known
        return (tf.reduce_sum(l_var) + tf.reduce_sum(l_cls)) / tf.maximum(tf.reduce_sum(valid), 1.0)

    return loss


def build_model(window=2048, channels=4, lr=1e-3, head='softmax',
                variant_pos_weight=1.0, common_pos_weight=1.0, n_freq=4):
    import tensorflow as tf
    from tensorflow.keras import layers as L

    if head not in HEADS:
        raise ValueError(f'head must be one of {HEADS}')
    if window % 16:
        raise ValueError('window length must be divisible by 16')
    inputs = L.Input((window, channels), name='dna')

    def block(x, n):
        for _ in range(2):
            x = L.Conv1D(n, 3, padding='same', activation='relu')(x)
            x = L.BatchNormalization()(x)
        return x

    s1 = block(inputs, 16)
    s2 = block(L.MaxPool1D(2)(s1), 32)
    s3 = block(L.MaxPool1D(2)(s2), 64)
    s4 = block(L.MaxPool1D(2)(s3), 128)
    x = block(L.MaxPool1D(2)(s4), 256)
    for skip, filters in [(s4, 256), (s3, 128), (s2, 64), (s1, 32)]:
        x = L.Conv1DTranspose(filters, 2, strides=2, padding='same')(x)
        x = L.Concatenate()([x, skip])
        x = block(x, filters)

    optimizer = tf.keras.optimizers.Adam(lr)
    if head == 'softmax':
        out = L.Conv1D(3, 3, padding='same', activation='softmax', name='frequency_class')(x)
        model = tf.keras.Model(inputs, out, name='allele_frequency_unet')
        # weighted_metrics honour sample_weight, so ignored (-1) sites are excluded.
        model.compile(optimizer=optimizer,
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                      weighted_metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')])
    elif head == 'spectrum':
        if n_freq < 2:
            raise ValueError('the spectrum head needs at least 2 frequency classes')
        variant = L.Conv1D(1, 3, padding='same', activation='sigmoid', name='variant')(x)
        classes = L.Conv1D(n_freq, 3, padding='same', activation='softmax',
                           name='class_given_variant')(x)
        out = L.Concatenate(name='variant_and_spectrum')([variant, classes])
        model = tf.keras.Model(inputs, out, name='allele_frequency_unet_spectrum')
        model.compile(optimizer=optimizer, loss=spectrum_loss(n_freq, variant_pos_weight))
    else:
        out = L.Conv1D(2, 3, padding='same', activation='sigmoid', name='variant_and_common')(x)
        model = tf.keras.Model(inputs, out, name='allele_frequency_unet_factorized')
        model.compile(optimizer=optimizer,
                      loss=factorized_loss(variant_pos_weight, common_pos_weight))
    return model


# ---------------------------------------------------------------- NumPy helpers

def _unweight_binary(p, weight):
    """Undo a positive-class weight w in BCE: the model learns w*p/(w*p + 1-p)."""
    if weight == 1:
        return p
    a = p / weight
    return a / (a + (1.0 - p))


def factorized_to_classes(raw, variant_pos_weight=1.0, common_pos_weight=1.0):
    """(..., 2) [P(variant), P(common|variant)] -> (..., 3) class probabilities."""
    raw = np.asarray(raw, dtype=np.float64)
    pv = _unweight_binary(raw[..., 0], variant_pos_weight)
    pc = _unweight_binary(raw[..., 1], common_pos_weight)
    return np.stack([1.0 - pv, pv * (1.0 - pc), pv * pc], axis=-1).astype(np.float32)


def spectrum_to_classes(raw, variant_pos_weight=1.0):
    """(..., 1+K) [P(variant), P(class | variant) x K] -> (..., 1+K) class
    probabilities [P(none), P(class 1), ..., P(class K)]."""
    raw = np.asarray(raw, dtype=np.float64)
    pv = _unweight_binary(raw[..., 0], variant_pos_weight)
    cond = raw[..., 1:]
    cond = cond / np.maximum(cond.sum(axis=-1, keepdims=True), _EPS)  # averaged windows: renormalize
    return np.concatenate([(1.0 - pv)[..., None], pv[..., None] * cond], axis=-1).astype(np.float32)


def n_output_classes(meta):
    """Number of class-probability columns predict.py writes for this model."""
    if meta.get('head', 'softmax') == 'spectrum':
        return 1 + int(meta.get('n_freq_classes', 4))
    return 3


def unweight_softmax(probs, class_weights=(1, 1, 1)):
    """Undo class weighting: weighted CE learns q_k proportional to w_k * p_k."""
    probs = np.asarray(probs, dtype=np.float64)
    w = np.asarray(class_weights, dtype=np.float64)
    if np.all(w == w[0]):
        return probs.astype(np.float32)
    q = probs / w
    return (q / q.sum(axis=-1, keepdims=True)).astype(np.float32)


def to_class_probabilities(raw, meta):
    """Convert raw network output to unweighted class probabilities
    (3 columns, or 1 + K for the spectrum head)."""
    if meta.get('head', 'softmax') == 'spectrum':
        return spectrum_to_classes(raw, meta.get('variant_pos_weight', 1.0))
    if meta.get('head', 'softmax') == 'factorized':
        return factorized_to_classes(raw, meta.get('variant_pos_weight', 1.0),
                                     meta.get('common_pos_weight', 1.0))
    return unweight_softmax(raw, meta.get('class_weights', (1, 1, 1)))
