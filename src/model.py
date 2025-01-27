import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Layer, Conv2D, BatchNormalization, Dense, Dropout, MaxPool2D, GlobalMaxPool2D


class TNet(Layer):
    def __init__(self, add_regularization=False, bn_momentum=0.99, **kwargs):
        super(TNet, self).__init__(**kwargs)
        self.add_regularization = add_regularization
        self.bn_momentum = bn_momentum
        self.conv0 = CustomConv(64, (1, 1), strides=(1, 1), bn_momentum=bn_momentum)
        self.conv1 = CustomConv(128, (1, 1), strides=(1, 1), bn_momentum=bn_momentum)
        self.conv2 = CustomConv(1024, (1, 1), strides=(1, 1), bn_momentum=bn_momentum)
        self.fc0 = CustomDense(512, activation=tf.nn.relu, apply_bn=True, bn_momentum=bn_momentum)
        self.fc1 = CustomDense(256, activation=tf.nn.relu, apply_bn=True, bn_momentum=bn_momentum)

    def build(self, input_shape):
        self.K = input_shape[-1]

        self.w = self.add_weight(shape=(256, self.K**2), initializer=tf.zeros_initializer,
                                 trainable=True, name='w')
        self.b = self.add_weight(shape=(self.K, self.K), initializer=tf.zeros_initializer,
                                 trainable=True, name='b')

        # Initialize bias with identity
        I = tf.constant(np.eye(self.K), dtype=tf.float32)
        self.b = tf.math.add(self.b, I)

    def call(self, x, training=None):
        input_x = x                                                     # BxNxK

        # Embed to higher dim
        x = tf.expand_dims(input_x, axis=2)                             # BxNx1xK
        x = self.conv0(x, training=training)
        x = self.conv1(x, training=training)
        x = self.conv2(x, training=training)
        x = tf.squeeze(x, axis=2)                                       # BxNx1024

        # Global features
        x = tf.reduce_max(x, axis=1)                                    # Bx1024

        # Fully-connected layers
        x = self.fc0(x, training=training)                              # Bx512
        x = self.fc1(x, training=training)                              # Bx256

        # Convert to KxK matrix to matmul with input
        x = tf.expand_dims(x, axis=1)                                   # Bx1x256
        x = tf.matmul(x, self.w)                                        # Bx1xK^2
        x = tf.squeeze(x, axis=1)
        x = tf.reshape(x, (-1, self.K, self.K))

        # Add bias term (initialized to identity matrix)
        x += self.b

        # Add regularization
        if self.add_regularization:
            eye = tf.constant(np.eye(self.K), dtype=tf.float32)
            x_xT = tf.matmul(x, tf.transpose(x, perm=[0, 2, 1]))
            reg_loss = tf.nn.l2_loss(eye - x_xT)
            self.add_loss(1e-3 * reg_loss)

        return tf.matmul(input_x, x)

    def get_config(self):
        config = super(TNet, self).get_config()
        config.update({
            'add_regularization': self.add_regularization,
            'bn_momentum': self.bn_momentum})

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class CustomConv(Layer):
    def __init__(self, filters, kernel_size, strides, padding='valid', activation=None,
                 apply_bn=False, bn_momentum=0.99, **kwargs):
        super(CustomConv, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.apply_bn = apply_bn
        self.bn_momentum = bn_momentum
        self.conv = Conv2D(filters, kernel_size, strides=strides, padding=padding,
                           activation=activation, use_bias=not apply_bn)
        if apply_bn:
            self.bn = BatchNormalization(momentum=bn_momentum, fused=False)

    def call(self, x, training=None):
        x = self.conv(x)
        if self.apply_bn:
            x = self.bn(x, training=training)
        if self.activation:
            x = self.activation(x)
        return x

    def get_config(self):
        config = super(CustomConv, self).get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'activation': self.activation,
            'apply_bn': self.apply_bn,
            'bn_momentum': self.bn_momentum})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class CustomDense(Layer):
    def __init__(self, units, activation=None, apply_bn=False, bn_momentum=0.99, **kwargs):
        super(CustomDense, self).__init__(**kwargs)
        self.units = units
        self.activation = activation
        self.apply_bn = apply_bn
        self.bn_momentum = bn_momentum
        self.dense = Dense(units, activation=activation, use_bias=not apply_bn)
        if apply_bn:
            self.bn = BatchNormalization(momentum=bn_momentum, fused=False)

    def call(self, x, training=None):
        x = self.dense(x)
        if self.apply_bn:
            x = self.bn(x, training=training)
        if self.activation:
            x = self.activation(x)
        return x

    def get_config(self):
        config = super(CustomDense, self).get_config()
        config.update({
            'units': self.units,
            'activation': self.activation,
            'apply_bn': self.apply_bn,
            'bn_momentum': self.bn_momentum})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class ResidualIdentityBlock(Layer):
    def __init__(self, filters, kernel_size):
        super(ResidualIdentityBlock, self).__init__(name='')
        filters1, filters2, filters3 = filters

        self.conv2a = Conv2D(filters1, (1, 1))
        self.bn2a = BatchNormalization()

        self.conv2b = Conv2D(filters2, kernel_size, padding='same')
        self.bn2b = BatchNormalization()

        self.conv2c = Conv2D(filters3, (1, 1))
        self.bn2c = BatchNormalization()

    def call(self, input_tensor, training=False):
        x = self.conv2a(input_tensor)
        x = self.bn2a(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2b(x)
        x = self.bn2b(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2c(x)
        x = self.bn2c(x, training=training)

        x += input_tensor
        return tf.nn.relu(x)

class ResidualConvBlock(Layer):
    def __init__(self, filters, kernel_size):
        super(ResidualConvBlock, self).__init__(name='')
        filters1, filters2, filters3 = filters

        self.conv2a = Conv2D(filters1, (1, 1))
        self.bn2a = BatchNormalization()

        self.conv2b = Conv2D(filters2, kernel_size, padding='same')
        self.bn2b = BatchNormalization()

        self.conv2c = Conv2D(filters3, (1, 1))
        self.bn2c = BatchNormalization()

        self.conv2d = Conv2D(filters3, (1, 1))
        self.bn2d = BatchNormalization()

    def call(self, input_tensor, training=False):
        x = self.conv2a(input_tensor)
        x = self.bn2a(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2b(x)
        x = self.bn2b(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2c(x)
        x = self.bn2c(x, training=training)

        # Skip connection path
        xs = self.conv2d(input_tensor)
        xs = self.bn2d(xs)

        x += xs
        return tf.nn.relu(x)


def get_model(bn_momentum):
    pt_cloud = Input(shape=(None, 5), dtype=tf.float32, name='pt_cloud')    # BxNx3

    # Input transformer (B x N x 3 -> B x N x 3)
    #pt_cloud_transform = TNet(bn_momentum=bn_momentum)(pt_cloud)

    # Embed to 64-dim space (B x N x 3 -> B x N x 64)
    pt_cloud_transform = tf.expand_dims(pt_cloud, axis=2)         # for weight-sharing of conv

    # Stage1
    x = CustomConv(64, (1, 1), strides=(1, 1), activation=tf.nn.relu, apply_bn=True,
                            bn_momentum=bn_momentum)(pt_cloud_transform)
    x = BatchNormalization()(x)
    x = tf.nn.relu(x)
    x = MaxPool2D(pool_size=(2,1))(x)

    #Stage2
    f = [64, 64, 256]
    x = ResidualConvBlock(filters=f, kernel_size=2)(x)
    x = ResidualIdentityBlock(filters=f, kernel_size=2)(x)
    #x = ResidualIdentityBlock(filters=f, kernel_size=2)(x)

    #Stage3
    f = [128, 128, 512]
    x = ResidualConvBlock(filters=f, kernel_size=2)(x)
    x = ResidualIdentityBlock(filters=f, kernel_size=2)(x)
    #x = ResidualIdentityBlock(filters=f, kernel_size=2)(x)

    #Stage4
    f = [256, 256, 1024]
    x = ResidualConvBlock(filters=f, kernel_size=2)(x)
    x = ResidualIdentityBlock(filters=f, kernel_size=2)(x)
    #x = ResidualIdentityBlock(filters=f, kernel_size=2)(x)

    #Stage5
    #f = [512, 512, 2048]
    f = [512, 512, 1024]
    x = ResidualConvBlock(filters=f, kernel_size=2)(x)
    x = ResidualIdentityBlock(filters=f, kernel_size=2)(x)
    x = ResidualIdentityBlock(filters=f, kernel_size=2)(x)

    # hidden_64 = CustomConv(64, (1, 1), strides=(1, 1), activation=tf.nn.relu, apply_bn=True,
    #                        bn_momentum=bn_momentum)(pt_cloud_transform)
    # embed_64 = CustomConv(64, (1, 1), strides=(1, 1), activation=tf.nn.relu, apply_bn=True,
    #                       bn_momentum=bn_momentum)(hidden_64)
    #embed_64 = tf.squeeze(embed_64, axis=2)

    # Feature transformer (B x N x 64 -> B x N x 64)
    #embed_64_transform = TNet(bn_momentum=bn_momentum, add_regularization=True)(embed_64)

    # Embed to 1024-dim space (B x N x 64 -> B x N x 1024)
    #embed_64_transform = tf.expand_dims(embed_64, axis=2)
    # hidden_64 = CustomConv(64, (1, 1), strides=(1, 1), activation=tf.nn.relu, apply_bn=True,
    #                        bn_momentum=bn_momentum)(embed_64)
    # hidden_128 = CustomConv(128, (1, 1), strides=(1, 1), activation=tf.nn.relu, apply_bn=True,
    #                         bn_momentum=bn_momentum)(res2)
    # embed_1024 = CustomConv(1024, (1, 1), strides=(1, 1), activation=tf.nn.relu, apply_bn=True,
    #                         bn_momentum=bn_momentum)(hidden_128)
    # x = tf.squeeze(x, axis=2)

    # Global feature vector (B x N x 1024 -> B x 1024)
    # global_descriptor = tf.reduce_max(x, axis=1)
    global_descriptor = GlobalMaxPool2D()(x)

    # FC layers to output k scores (B x 1024 -> B x 5)
    # hidden_512 = CustomDense(512, activation=tf.nn.relu, apply_bn=True,
    #                          bn_momentum=bn_momentum)(global_descriptor)
    # hidden_512 = Dropout(rate=0.2)(hidden_512)

    # hidden_256 = CustomDense(256, activation=tf.nn.relu, apply_bn=True,
    #                          bn_momentum=bn_momentum)(hidden_512)
    # hidden_256 = Dropout(rate=0.2)(hidden_256)

    # logits = CustomDense(5, apply_bn=False)(hidden_256)

    logits = Dense(units=5)(global_descriptor)

    return Model(inputs=pt_cloud, outputs=logits)
