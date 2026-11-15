"""The original OVEP model, ported unchanged to TensorFlow 2.x / Keras 3.

Architecture and loss are copied from the original repository's
code_halftrainmaskmissing/base_excludeN_final_extrapolate/cnn_batchnorm.py:
encoder blocks are Conv1D(relu) followed by BatchNormalization, the decoder
has no batch normalization, the output is one sigmoid per base, and the loss is
plain (unmasked) binary cross-entropy with Adam at its default settings.
"""
import numpy as np

WINDOW = 2048


def self_crossentropy(y_true, y_pred):
    """Same as the original: mean binary cross-entropy over every base."""
    import tensorflow as tf
    y_true = tf.reshape(tf.cast(y_true, tf.float32), [-1])
    y_pred = tf.clip_by_value(tf.reshape(tf.cast(y_pred, tf.float32), [-1]), 1e-7, 1.0 - 1e-7)
    out = -(y_true * tf.math.log(y_pred) + (1.0 - y_true) * tf.math.log(1.0 - y_pred))
    return tf.reduce_mean(out)


def cnn1d(size=WINDOW, channel=4):
    from tensorflow.keras.layers import (Input, Conv1D, MaxPooling1D, BatchNormalization,
                                         Conv1DTranspose, concatenate)
    from tensorflow.keras.models import Model

    def enc(x, n):
        x = BatchNormalization()(Conv1D(n, kernel_size=3, activation='relu', padding='same')(x))
        return BatchNormalization()(Conv1D(n, kernel_size=3, activation='relu', padding='same')(x))

    def dec(x, skip, n):
        x = concatenate([Conv1DTranspose(n, 2, strides=2, padding='same')(x), skip], axis=2)
        x = Conv1D(n, 3, activation='relu', padding='same')(x)
        return Conv1D(n, 3, activation='relu', padding='same')(x)

    the_input = Input((size, channel))
    conv1 = enc(the_input, 16)
    conv2 = enc(MaxPooling1D(pool_size=2)(conv1), 32)
    conv3 = enc(MaxPooling1D(pool_size=2)(conv2), 64)
    conv4 = enc(MaxPooling1D(pool_size=2)(conv3), 128)
    conv5 = enc(MaxPooling1D(pool_size=2)(conv4), 256)
    conv6 = dec(conv5, conv4, 256)
    conv7 = dec(conv6, conv3, 128)
    conv8 = dec(conv7, conv2, 64)
    up9 = concatenate([Conv1DTranspose(32, 2, strides=2, padding='same')(conv8), conv1], axis=2)
    conv9 = Conv1D(32, 3, activation='relu', padding='same')(up9)
    conv9 = Conv1D(1, 3, activation='sigmoid', padding='same')(conv9)
    model = Model(inputs=[the_input], outputs=[conv9])
    model.compile(loss=self_crossentropy, optimizer='adam', metrics=['mean_absolute_error'], jit_compile=False)
    return model


def original_labels(sites_vcf, chromosome, length):
    """Binary labels exactly as the original data/pickle_label_chr.py made them.

    Every VCF record marks 1 (no FILTER, AC or callability check). A record
    whose ALT field is as long as REF marks POS only; any other record marks
    POS plus len(REF) further bases (1000genome/select_indel.py). Everything
    else, including N, is 0.
    """
    import gzip
    label = np.zeros(length, dtype=np.uint8)
    prefix = chromosome + '\t'
    opener = gzip.open if str(sites_vcf).endswith('.gz') else open
    with opener(sites_vcf, 'rt') as fh:
        for line in fh:
            if not line.startswith(prefix):
                continue
            f = line.split('\t', 5)
            pos, ref, alt = int(f[1]), f[3], f[4]
            end = pos + 1 if len(alt) == len(ref) else pos + len(ref) + 1
            label[max(pos - 1, 0):min(end - 1, length)] = 1
    return label


def calibration_sites(labels, dna, window_size=512, step_size=256):
    """Boolean mask of sites the original calibration.py fits on: positions in
    any 512-bp window (step 256) that has at least one label and no N base."""
    labels = np.asarray(labels)
    n = len(labels)
    keep = np.zeros(n, dtype=bool)
    starts = np.arange(0, n - window_size + 1, step_size)
    if starts.size == 0:
        return keep
    lab = np.concatenate([[0], np.cumsum(labels != 0)])
    nbase = np.concatenate([[0], np.cumsum(np.asarray(dna).max(axis=1) == 0)])
    ok = ((lab[starts + window_size] - lab[starts]) > 0) & ((nbase[starts + window_size] - nbase[starts]) == 0)
    for s in starts[ok]:
        keep[s:s + window_size] = True
    return keep
