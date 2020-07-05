import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from keras.layers import *
from keras.optimizers import Adam
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
    out = InstanceNormalization()(out)
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
    d = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=kernel_init)(inp)
    d = LeakyReLU(alpha=0.2)(d)
    d = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=kernel_init)(d)
    d = InstanceNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    d = Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=kernel_init)(d)
    d = InstanceNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    d = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=kernel_init)(d)
    d = InstanceNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    d = Conv2D(512, (4,4), padding='same', kernel_initializer=kernel_init)(d)
    d = InstanceNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    patch_out = Conv2D(1, (4,4), padding='same', kernel_initializer=kernel_init)(d)

    model = Model(inp, patch_out)

    model.compile(loss='mse', optimizer=Adam(lr=0.0002, beta_1=0.5), loss_weights=[0.5])
    if summary:
        model.summary()
    return model


if __name__ == "__main__":
    g = define_generator((256, 256, 3), 9, summary=True)
    print("\n\n\n\n")
    d = define_discriminator((256, 256, 3), summary=True)



