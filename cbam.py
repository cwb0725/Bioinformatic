import tensorflow as tf
from tensorflow.keras import layers


def channel_attention(x, reduction=16):
    channels = x.shape[-1]

    avg_pool = layers.GlobalAveragePooling2D()(x)
    max_pool = layers.GlobalMaxPooling2D()(x)

    shared_dense_1 = layers.Dense(channels // reduction, activation='relu')
    shared_dense_2 = layers.Dense(channels)

    avg_out = shared_dense_2(shared_dense_1(avg_pool))
    max_out = shared_dense_2(shared_dense_1(max_pool))

    scale = tf.nn.sigmoid(avg_out + max_out)
    scale = layers.Reshape((1, 1, channels))(scale)

    return x * scale


def spatial_attention(inputs):
    x_max = tf.reduce_max(inputs, axis=-1, keepdims=True)
    x_avg = tf.reduce_mean(inputs, axis=-1, keepdims=True)

    x = layers.Concatenate(axis=-1)([x_max, x_avg])
    x = layers.Conv2D(1, kernel_size=7, padding='same')(x)
    x = layers.Activation('sigmoid')(x)

    return layers.Multiply()([inputs, x])


def cbam(x):
    x = channel_attention(x)
    x = spatial_attention(x)
    return x
