import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from keras.layers import *
from layers import *

# Here we will be using "same" padding  rather than reflected padding
# proposed in the paper

def conv_block(inp):
    kernel_init = tf.keras.initializers.TruncatedNormal(stddev=0.02)
    out = Conv2D(64, kernel_size=(7, 7), strides=(1, 1), padding="same", kernel_initializer=kernel_init)(inp)
    out = InstanceNormalization()(out)
    out = ReLU()(out)
    out = Conv2D(128, kernel_size=(3, 3), strides=(2, 2), padding="same", kernel_initializer=kernel_init)(out)
    out = InstanceNormalization()(out)
    out = ReLU()(out)
    out = Conv2D(256, kernel_size=(3, 3), strides=(2, 2), padding="same", kernel_initializer=kernel_init)(out)
    out = InstanceNormalization()(out)
    out = ReLU()(out)
    return out

def resnet_block(inp, n_filters):
    kernel_init = tf.keras.initializers.TruncatedNormal(stddev=0.02)
    g = Conv2D(n_filters, kernel_size=(3, 3), padding="same", kernel_initializer=kernel_init)(inp)
    g = InstanceNormalization()(g)
    g = ReLU()(g)
    g = Conv2D(n_filters, kernel_size=(3, 3), padding="same", kernel_initializer=kernel_init)(g)
    g = InstanceNormalization()(g)
    g = ReLU()(g)
    return g + inp



def deconv_block(inp):
    kernel_init = tf.keras.initializers.TruncatedNormal(stddev=0.02)
    out = Conv2DTranspose(128, kernel_size=(3, 3), strides=(2, 2), padding="same", kernel_initializer=kernel_init)(inp)
    out = InstanceNormalization()(out)
    out = ReLU()(out)
    out = Conv2DTranspose(64, kernel_size=(3, 3), strides=(2, 2), padding="same", kernel_initializer=kernel_init)(out)
    out = InstanceNormalization()(out)
    out = ReLU()(out)
    out = Conv2D(3, kernel_size=(7, 7), strides=(1, 1), padding="same", kernel_initializer=kernel_init)(out)
    out = Activation("tanh")(out)
    return out



def define_generator(inp_shape, num_resnet_blocks, summary=True):
    inp = Input(shape=inp_shape)
    out = conv_block(inp)
    for _ in range(num_resnet_blocks):
        out = resnet_block(out, 256)
    out = deconv_block(out)
    model = Model([inp], [out])
    if summary:
        model.summary()
    return model


def define_discriminator(inp_shape, summary=False):
    inp = Input(shape=inp_shape)
    kernel_init = tf.keras.initializers.TruncatedNormal(stddev=0.02)
    h0 = Conv2D(32, (4, 4), bias_initializer=None, kernel_initializer=kernel_init, strides=2)(inp)
    h0 = LeakyReLU()(h0)
    h1 = Conv2D(64, (4, 4), bias_initializer=None, kernel_initializer=kernel_init, strides=2)(h0)
    h1 = BatchNormalization()(h1)
    h1 = LeakyReLU()(h1)
    h2 = Conv2D(128, (4, 4), bias_initializer=None, kernel_initializer=kernel_init, strides=2)(h1)
    h2 = BatchNormalization()(h2)
    h2 = LeakyReLU()(h2)
    h3 = Conv2D(256, (4, 4), bias_initializer=None, kernel_initializer=kernel_init, strides=2)(h2)
    h3 = BatchNormalization()(h3)
    h3 = LeakyReLU()(h3)
    h4 = Conv2D(1, (3, 3), bias_initializer=None, kernel_initializer=kernel_init)(h3)
    model = Model(inputs=[inp], outputs=[h4])
    if summary:
        model.summary()
    return model


if __name__ == "__main__":
    g = define_generator((256, 256, 3), 6, summary=True)
    print("\n\n\n\n")
    d = define_discriminator((256, 256, 3), summary=True)



