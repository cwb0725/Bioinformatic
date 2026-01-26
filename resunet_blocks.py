import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Activation, MaxPooling2D, Add
from tensorflow.keras.activations import relu


# =========================================================
# ResUNet Downsampling Block
# 完全等价于原 conv_block
# =========================================================
def resunet_down_block(input_x, kn1, kn2, kn3, side_kn):
    """
    Parameters
    ----------
    input_x : Tensor
        Input feature map
    kn1, kn2, kn3 : int
        Filters for main branch (1x1 -> 3x3 -> 1x1)
    side_kn : int
        Filters for shortcut branch
    """

    # ----- Main branch -----
    x = Conv2D(filters=kn1, kernel_size=(1, 1))(input_x)
    x = Activation(relu)(x)

    x = Conv2D(filters=kn2, kernel_size=(3, 3), padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Activation(relu)(x)

    x = Conv2D(filters=kn3, kernel_size=(1, 1))(x)
    x = Activation(relu)(x)

    # ----- Shortcut branch -----
    y = Conv2D(filters=side_kn, kernel_size=(1, 1))(input_x)
    y = MaxPooling2D(pool_size=(2, 2))(y)
    y = Activation(relu)(y)

    # ----- Merge -----
    out = Add()([x, y])
    out = Activation(relu)(out)

    return out


# =========================================================
# ResUNet Upsampling Block
# 完全等价于原 conv_block1
# =========================================================
def resunet_up_block(input_x, kn1, kn2, kn3, side_kn):
    """
    No pooling inside this block
    """

    # ----- Main branch -----
    x = Conv2D(filters=kn1, kernel_size=(1, 1))(input_x)
    x = Activation(relu)(x)

    x = Conv2D(filters=kn2, kernel_size=(3, 3), padding='same')(x)
    x = Activation(relu)(x)

    x = Conv2D(filters=kn3, kernel_size=(1, 1))(x)
    x = Activation(relu)(x)

    # ----- Shortcut branch -----
    y = Conv2D(filters=side_kn, kernel_size=(1, 1))(input_x)
    y = Activation(relu)(y)

    # ----- Merge -----
    out = Add()([x, y])
    out = Activation(relu)(out)

    return out


# =========================================================
# ResUNet Identity Block
# 完全等价于原 identity_block
# =========================================================
def resunet_identity_block(input_x, kn1, kn2, kn3):
    """
    Input and output channel dimensions must match
    """

    # ----- Main branch -----
    x = Conv2D(filters=kn1, kernel_size=(1, 1))(input_x)
    x = Activation(relu)(x)

    x = Conv2D(filters=kn2, kernel_size=(3, 3), padding='same')(x)
    x = Activation(relu)(x)

    x = Conv2D(filters=kn3, kernel_size=(1, 1))(x)
    x = Activation(relu)(x)

    # ----- Identity merge -----
    out = Add()([x, input_x])
    out = Activation(relu)(out)

    return out
