import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from .config import BASE_CH, INPUT_MODE, OUT_SCALE, RESIDUAL_ALPHA_INIT, USE_RESIDUAL_ON_COARSE


def conv_block(x, f, k=3):
    x = layers.Conv2D(f, k, padding="same")(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(f, k, padding="same")(x)
    x = layers.Activation("relu")(x)
    return x


class ScalarMultiply(layers.Layer):
    def __init__(self, init_value=1e-3, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.init_value = float(init_value)

    def build(self, input_shape):
        self.alpha = self.add_weight(
            name="alpha",
            shape=(),
            initializer=tf.keras.initializers.Constant(self.init_value),
            trainable=True,
        )

    def call(self, x):
        return self.alpha * x


def build_unet(input_shape, base_ch=BASE_CH, out_scale=OUT_SCALE):
    inp = keras.Input(shape=input_shape)

    mask0 = layers.Lambda(lambda t: t[..., 2:3], name="mask_l0")(inp)
    mask1 = layers.AveragePooling2D(pool_size=2, strides=2, padding="same", name="mask_l1")(mask0)
    mask2 = layers.AveragePooling2D(pool_size=2, strides=2, padding="same", name="mask_l2")(mask1)
    mask3 = layers.AveragePooling2D(pool_size=2, strides=2, padding="same", name="mask_l3")(mask2)
    mask4 = layers.AveragePooling2D(pool_size=2, strides=2, padding="same", name="mask_l4")(mask3)

    x0 = layers.Concatenate()([inp, mask0])
    c1 = conv_block(x0, base_ch)
    p1 = layers.MaxPool2D()(c1)

    x1 = layers.Concatenate()([p1, mask1])
    c2 = conv_block(x1, base_ch * 2)
    p2 = layers.MaxPool2D()(c2)

    x2 = layers.Concatenate()([p2, mask2])
    c3 = conv_block(x2, base_ch * 4)
    p3 = layers.MaxPool2D()(c3)

    x3 = layers.Concatenate()([p3, mask3])
    c4 = conv_block(x3, base_ch * 8)
    p4 = layers.MaxPool2D()(c4)

    x4 = layers.Concatenate()([p4, mask4])
    bn = conv_block(x4, base_ch * 16)

    u4 = layers.UpSampling2D(interpolation="bilinear")(bn)
    u4 = layers.Concatenate()([u4, c4, mask3])
    c5 = conv_block(u4, base_ch * 8)

    u3 = layers.UpSampling2D(interpolation="bilinear")(c5)
    u3 = layers.Concatenate()([u3, c3, mask2])
    c6 = conv_block(u3, base_ch * 4)

    u2 = layers.UpSampling2D(interpolation="bilinear")(c6)
    u2 = layers.Concatenate()([u2, c2, mask1])
    c7 = conv_block(u2, base_ch * 2)

    u1 = layers.UpSampling2D(interpolation="bilinear")(c7)
    u1 = layers.Concatenate()([u1, c1, mask0])
    c8 = conv_block(u1, base_ch)

    if USE_RESIDUAL_ON_COARSE:
        coarse_uv = layers.Lambda(lambda t: t[..., 3:5], name="coarse_uv")(inp)
        residual_raw = layers.Conv2D(
            2,
            1,
            padding="same",
            kernel_initializer="zeros",
            bias_initializer="zeros",
            name="residual_raw_uv",
        )(c8)
        residual_bounded = layers.Lambda(lambda t: out_scale * tf.tanh(t), name="residual_bounded_uv")(residual_raw)
        residual_scaled = ScalarMultiply(init_value=RESIDUAL_ALPHA_INIT, name="residual_alpha")(residual_bounded)
        out = layers.Add(name="pred_uv")([coarse_uv, residual_scaled])
    else:
        raw = layers.Conv2D(2, 1, padding="same", name="raw_uv")(c8)
        out = layers.Lambda(lambda t: out_scale * t, name="pred_uv")(raw)

    return keras.Model(inp, out, name=f"unet_sparse2dense_{input_shape[-1]}ch")


def input_channels():
    return 3 if INPUT_MODE == "3ch" else 5
