try:
    import tensorflow.compat.v1 as tf
except ImportError:
    import tensorflow as tf
from .utils.ops import *

def descriptor(inputs, attr, batch_layer=batch_norm, net={}, reuse=False):
    with tf.variable_scope('des', reuse=tf.AUTO_REUSE):

        # input ConvNet structure
        net['conv1'] = conv2d(inputs, 64, kernal=(5, 5), strides=(2, 2), padding="SAME", name="conv1")
        # conv1 = batch_layer(conv1, name="conv1_bn")
        net['conv1'] = leaky_relu(net['conv1'])

        net['conv2'] = conv2d(net['conv1'], 128, kernal=(3, 3), strides=(2, 2), padding="SAME", name="conv2")
        # conv2 = batch_layer(conv2, name="conv2_bn")
        net['conv2'] = leaky_relu(net['conv2'])

        net['conv3'] = conv2d(net['conv2'], 256, kernal=(3, 3), strides=(1, 1), padding="SAME", name="conv3")
        # conv3 = batch_layer(conv3, name="conv3_bn")
        net['conv3'] = leaky_relu(net['conv3'])

        #net['conv4'] = conv2d(net['conv3'], 512, kernal=(3, 3), strides=(1, 1), padding="SAME", name="conv4")
        # conv4 = batch_layer(conv4, name="conv4_bn")
        #net['conv4'] = leaky_relu(net['conv4'])

        net['fc'] = fully_connected(net['conv3'] , 256, name="fc")
        net['fc'] = tf.reshape(net['fc'], [-1, 256])

        #net['fc'] = leaky_relu(net['fc'])
        net['enc'] = tf.concat([attr, net['fc']], axis=1)

        net['fc1'] = tf.layers.dense(net['enc'], 128, kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        net['fc1'] = leaky_relu(net['fc1'])

        net['fc2'] = tf.layers.dense(net['fc1'], 64, kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        net['fc2'] = leaky_relu(net['fc2'])

        net['fc3'] = tf.layers.dense(net['fc2'], 1, kernel_initializer=tf.random_normal_initializer(stddev=0.02))

        return net['fc3']

def descriptor_attr2img_linear(inputs, attr, batch_layer=batch_norm, net={}, reuse=False):
    with tf.variable_scope('des', reuse=tf.AUTO_REUSE):
        df_dim = 64
        # input ConvNet structure
        net['conv1'] = conv2d(inputs, df_dim, kernal=(5, 5), strides=(2, 2), padding="SAME", name="conv1")
        net['conv1'] = leaky_relu(net['conv1'])

        net['conv2'] = conv2d(net['conv1'], df_dim*2, kernal=(3, 3), strides=(2, 2), padding="SAME", name="conv2")
        net['conv2'] = leaky_relu(net['conv2'])

        # net['conv3'] = conv2d(net['conv2'], df_dim*4, kernal=(4, 4), strides=(2, 2), padding="SAME", name="conv3")
        # net['conv3'] = leaky_relu(net['conv3'])


        # net['conv4'] = conv2d(net['conv3'], df_dim*8, kernal=(4, 4), strides=(2, 2), padding="SAME", name="conv4")
       #  net['conv4'] = leaky_relu(net['conv4'])
        # (bn) x 4 x 4 x 512
        # (bn) x 312
        attr_size = int(attr.get_shape()[1])
        net['attr'] = linear(attr, 16 * 16 * attr_size, name='linear')

        net['attr'] = tf.reshape(net['attr'], [-1, 16, 16, attr_size])
        net['enc'] = tf.concat([net['attr'], net['conv2']], axis=3, name='concat')


        net['conv5'] = conv2d(net['enc'], df_dim*8, kernal=(3, 3), strides=(1, 1), padding="SAME", name="conv5")
        net['conv5'] = leaky_relu(net['conv5'])

        net['fc'] = fully_connected(net['conv5'], 100, name="fc")

        return net['fc']

def descriptor_attr2img(inputs, attr, batch_layer=batch_norm, net={}, reuse=False):
    with tf.variable_scope('des', reuse=tf.AUTO_REUSE):
        df_dim = 64
        # input ConvNet structure
        net['conv1'] = conv2d(inputs, df_dim, kernal=(5, 5), strides=(2, 2), padding="SAME", name="conv1")
        net['conv1'] = leaky_relu(net['conv1'])

        net['conv2'] = conv2d(net['conv1'], df_dim*2, kernal=(3, 3), strides=(2, 2), padding="SAME", name="conv2")
        net['conv2'] = leaky_relu(net['conv2'])

        # net['conv3'] = conv2d(net['conv2'], df_dim*4, kernal=(4, 4), strides=(2, 2), padding="SAME", name="conv3")
        # net['conv3'] = leaky_relu(net['conv3'])


        # net['conv4'] = conv2d(net['conv3'], df_dim*8, kernal=(4, 4), strides=(2, 2), padding="SAME", name="conv4")
       #  net['conv4'] = leaky_relu(net['conv4'])
        # (bn) x 4 x 4 x 512
        # (bn) x 312

        # net['attr'] = linear(attr, 128, name='linear')
        net['attr'] = tf.reshape(attr, [-1, 1, 1, attr.get_shape()[1]])
        net['attr'] = tf.tile(net['attr'], [1, 32, 32, 1], name='tile')
        net['enc'] = tf.concat([net['attr'], net['conv2']], axis=3, name='concat')

        net['conv5'] = conv2d(net['enc'], df_dim*4, kernal=(3, 3), strides=(1, 1), padding="SAME", name="conv5")
        net['conv5'] = leaky_relu(net['conv5'])

        net['fc'] = fully_connected(net['conv5'], 100, name="fc")

        return net['fc']

def descriptor_label2img(inputs, attr, batch_layer=batch_norm, net={}, reuse=False):
    with tf.variable_scope('des', reuse=tf.AUTO_REUSE):
        ndf = 64
        ngf  = 64
        # input ConvNet structure
        net['attr'] = tf.reshape(attr, [-1, 1, 1, attr.get_shape()[1]])
        net['convt1'] = convt2d(net['attr'], (None, 4, 4, ngf*8), kernal=(4, 4), strides=(1, 1), padding="VALID", name="convt1")
        net['convt1'] = batch_layer(net['convt1'], name="convt1_bn")
        # convt1 = tf.nn.dropout(convt1, 0.5)
        net['convt1'] = tf.nn.relu(net['convt1'])

        # (2 x 2 x 512)
        net['convt2'] = convt2d(net['convt1'], (None, 8, 8, ngf*4), kernal=(5, 5), strides=(2, 2), padding="SAME", name="convt2")
        net['convt2'] = batch_layer(net['convt2'], name="convt2_bn")
        # convt2 = tf.nn.dropout(convt2, 0.5)
        net['convt2'] = tf.nn.relu(net['convt2'])

        # (4 x 4 x 512)
        net['convt3'] = convt2d(net['convt2'], (None, 16, 16, ngf*2), kernal=(5, 5), strides=(2, 2), padding="SAME", name="convt3")
        net['convt3'] = batch_layer(net['convt3'], name="convt3_bn")
        # convt3 = tf.nn.dropout(convt3, 0.5)
        net['convt3'] = tf.nn.relu(net['convt3'])

        # (8 x 8 x 512)
        net['convt4'] = convt2d(net['convt3'], (None, 32, 32, ngf), kernal=(5, 5), strides=(2, 2), padding="SAME", name="convt4")
        net['convt4'] = batch_layer(net['convt4'], name="convt4_bn")
        net['convt4'] = tf.nn.relu(net['convt4'])

        # (16 x 16 x 512)
        net['convt5'] = convt2d(net['convt4'], (None, 64, 64, 25), kernal=(5, 5), strides=(2, 2), padding="SAME", name="convt5")
        net['convt5'] = tf.nn.tanh(net['convt5'])

        net['enc'] = tf.concat([net['convt5'], inputs], axis=3)

        net['conv1'] = conv2d(net['enc'], ndf, kernal=(5, 5), strides=(2, 2), padding="SAME", name="conv1")
        net['conv1'] = leaky_relu(net['conv1'])

        net['conv2'] = conv2d(net['conv1'], ndf*2, kernal=(5, 5), strides=(2, 2), padding="SAME", name="conv2")
        net['conv2'] = leaky_relu(net['conv2'])

        net['conv3'] = conv2d(net['conv2'], ndf*4, kernal=(5, 5), strides=(2, 2), padding="SAME", name="conv3")
        net['conv3'] = leaky_relu(net['conv3'])

        net['conv4'] = conv2d(net['conv3'], ndf*8, kernal=(4, 4), strides=(2, 2), padding="SAME", name="conv4")
        net['conv4'] = leaky_relu(net['conv4'])

        net['fc'] = fully_connected(net['conv4'], 100, name="fc")

        return net['fc']


def descriptor_mnist(inputs, attr, batch_layer=batch_norm, net={}, reuse=False):
    with tf.variable_scope('des', reuse=tf.AUTO_REUSE):
        df_dim = 64
        gf_dim  = 64
        # input ConvNet structure
        net['attr'] = tf.reshape(attr, [-1, 1, 1, attr.get_shape()[1]])

        convt1 = convt2d(net['attr'], (None, 4, 4, gf_dim*4), kernal=(4, 4), strides=(1, 1), padding="VALID", name="convt1")
        convt1 = batch_layer(convt1, name="convt1_bn")
        # convt1 = tf.nn.dropout(convt1, 0.5)
        convt1 = tf.nn.relu(convt1)

        # (2 x 2 x 512)
        convt2 = convt2d(convt1, (None, 7, 7, gf_dim*2), kernal=(4, 4), strides=(2, 2), padding="SAME", name="convt2")
        convt2 = batch_layer(convt2, name="convt2_bn")
        # convt2 = tf.nn.dropout(convt2, 0.5)
        convt2 = tf.nn.relu(convt2)

        # (4 x 4 x 512)
        convt3 = convt2d(convt2, (None, 14, 14, gf_dim), kernal=(4, 4), strides=(2, 2), padding="SAME", name="convt3")
        convt3 = batch_layer(convt3, name="convt3_bn")
        # convt3 = tf.nn.dropout(convt3, 0.5)
        convt3 = tf.nn.relu(convt3)

        # (8 x 8 x 512)
        convt4 = convt2d(convt3, (None, 28, 28, 1), kernal=(4, 4), strides=(2, 2), padding="SAME", name="convt4")
        convt4 = tf.nn.tanh(convt4)


        net['enc'] = tf.concat([convt4, inputs], axis=3)

        net['conv1'] = conv2d(net['enc'], df_dim, kernal=(4, 4), strides=(2, 2), padding="SAME", name="conv1")
        net['conv1'] = leaky_relu(net['conv1'])

        net['conv2'] = conv2d(net['conv1'], df_dim*2, kernal=(4, 4), strides=(2, 2), padding="SAME", name="conv2")
        net['conv2'] = leaky_relu(net['conv2'])

        net['conv3'] = conv2d(net['conv2'], df_dim*4, kernal=(4, 4), strides=(2, 2), padding="SAME", name="conv3")
        net['conv3'] = leaky_relu(net['conv3'])

        # net['conv4'] = conv2d(net['conv3'], 1, kernal=(4, 4), strides=(1, 1), padding="VALID", name="conv4")
        # net['conv4'] = leaky_relu(net['conv4'])


        # (bn) x 4 x 4 x 512
        # (bn) x 312

        # net['attr'] = tf.reshape(attr, [-1, 1, 1, attr.get_shape()[1]])
        # net['attr'] = tf.tile(net['attr'], [1, 4, 4, 1], name='tile')
        # net['enc'] = tf.concat([net['attr'], net['conv3']], axis=3, name='concat')
        #
        #
        # net['conv5'] = conv2d(net['enc'], df_dim*8, kernal=(4, 4), strides=(1, 1), padding="SAME", name="conv5")
        # net['conv5'] = leaky_relu(net['conv5'])
        # net['img'] = tf.reshape(inputs, [-1, 1, 1, 784])
        # net['enc'] = tf.concat([net['attr'], net['conv4']], axis=3, name='concat')
        # net['enc'] = tf.concat([net['attr'], net['img']], axis=3)



        # net['conv4'] = fully_connected(net['enc'], df_dim*2, name="conv4")
        # net['conv4'] = leaky_relu(net['conv4'])
        #
        # net['conv5'] = fully_connected(net['conv4'], df_dim, name="conv5")
        # net['conv5'] = leaky_relu(net['conv5'])
        #
        net['fc'] = fully_connected(net['conv3'], 100, name="fc")

        #print(net['fc'])

        return net['fc']


def descriptor_onehotmap(inputs, attr, batch_layer=batch_norm, net={}, reuse=False):
    with tf.variable_scope('des', reuse=tf.AUTO_REUSE):
        df_dim = 64

        onehotattr = None
        for value in attr:
            if value == 1:
                if onehotattr is None:
                    onehotattr = tf.ones([inputs.shape[0].value, inputs.shape[1].value, inputs.shape[2].value, 1])
                else:
                    onehotattr = tf.concat(
                        [onehotattr, tf.ones([inputs.shape[0].value, inputs.shape[1].value, inputs.shape[2].value, 1])],
                        axis=3)
            else:
                if onehotattr is None:
                    onehotattr = tf.zeros([inputs.shape[0].value, inputs.shape[1].value, inputs.shape[2].value, 1])
                else:
                    onehotattr = tf.concat(
                        [onehotattr, tf.zeros([inputs.shape[0].value, inputs.shape[1].value, inputs.shape[2].value, 1])],
                        axis=3)

        net['enc'] = tf.concat([onehotattr, inputs], axis=3)

        net['conv1'] = conv2d(net['enc'], df_dim, kernal=(4, 4), strides=(2, 2), padding="SAME", name="conv1")
        net['conv1'] = leaky_relu(net['conv1'])

        net['conv2'] = conv2d(net['conv1'], df_dim * 2, kernal=(4, 4), strides=(2, 2), padding="SAME", name="conv2")
        net['conv2'] = leaky_relu(net['conv2'])

        net['conv3'] = conv2d(net['conv2'], df_dim * 4, kernal=(4, 4), strides=(2, 2), padding="SAME", name="conv3")
        net['conv3'] = leaky_relu(net['conv3'])

        net['fc'] = fully_connected(net['conv3'], 100, name="fc")

        return net['fc']


def generator_onehotmap(attr, z, batch_layer=batch_norm, reuse=False):
    gf_dim = 64

    with tf.variable_scope('gen', reuse=tf.AUTO_REUSE):
        # attr = tf.layers.dense(attr, units=256, activation=leaky_relu,
        #                               kernel_initializer=tf.random_normal_initializer(stddev=0.01))

        input_ = tf.concat([attr, z], axis=1)
        input_ = tf.reshape(input_, [-1, 1, 1, input_.get_shape()[1]])

        # (1 x 1 x 512)
        convt1 = convt2d(input_, (None, 4, 4, gf_dim * 4), kernal=(4, 4), strides=(1, 1), padding="VALID",
                         name="convt1")
        convt1 = batch_layer(convt1, name="convt1_bn")
        # convt1 = tf.nn.dropout(convt1, 0.5)
        convt1 = tf.nn.relu(convt1)

        # (2 x 2 x 512)
        convt2 = convt2d(convt1, (None, 7, 7, gf_dim * 2), kernal=(4, 4), strides=(2, 2), padding="SAME", name="convt2")
        convt2 = batch_layer(convt2, name="convt2_bn")
        # convt2 = tf.nn.dropout(convt2, 0.5)
        convt2 = tf.nn.relu(convt2)

        # (4 x 4 x 512)
        convt3 = convt2d(convt2, (None, 14, 14, gf_dim), kernal=(4, 4), strides=(2, 2), padding="SAME", name="convt3")
        convt3 = batch_layer(convt3, name="convt3_bn")
        # convt3 = tf.nn.dropout(convt3, 0.5)
        convt3 = tf.nn.relu(convt3)

        # (8 x 8 x 512)
        convt4 = convt2d(convt3, (None, 28, 28, 1), kernal=(4, 4), strides=(2, 2), padding="SAME", name="convt4")
        convt4 = tf.nn.tanh(convt4)

        return convt4




def generator64(attr, z, batch_layer=batch_norm, reuse=False):
    with tf.variable_scope('gen', reuse=tf.AUTO_REUSE):
        # attr = tf.layers.dense(attr, units=256, activation=leaky_relu,
        #                               kernel_initializer=tf.random_normal_initializer(stddev=0.01))

        input_ = tf.concat([attr, z], axis=1)
        input_ = tf.reshape(input_, [-1, 1, 1, input_.get_shape()[1]])

        # (1 x 1 x 512)
        convt1 = convt2d(input_, (None, 4, 4, 512), kernal=(4, 4), strides=(1, 1), padding="VALID", name="convt1")
        convt1 = batch_layer(convt1, name="convt1_bn")
        # convt1 = tf.nn.dropout(convt1, 0.5)
        convt1 = tf.nn.relu(convt1)

        # (2 x 2 x 512)
        convt2 = convt2d(convt1, (None, 8, 8, 256), kernal=(5, 5), strides=(2, 2), padding="SAME", name="convt2")
        convt2 = batch_layer(convt2, name="convt2_bn")
        # convt2 = tf.nn.dropout(convt2, 0.5)
        convt2 = tf.nn.relu(convt2)

        # (4 x 4 x 512)
        convt3 = convt2d(convt2, (None, 16, 16, 128), kernal=(5, 5), strides=(2, 2), padding="SAME", name="convt3")
        convt3 = batch_layer(convt3, name="convt3_bn")
        # convt3 = tf.nn.dropout(convt3, 0.5)
        convt3 = tf.nn.relu(convt3)

        # (8 x 8 x 512)
        convt4 = convt2d(convt3, (None, 32, 32, 64), kernal=(5, 5), strides=(2, 2), padding="SAME", name="convt4")
        convt4 = batch_layer(convt4, name="convt4_bn")
        convt4 = tf.nn.relu(convt4)

        # (16 x 16 x 512)
        convt5 = convt2d(convt4, (None, 64, 64, 3), kernal=(5, 5), strides=(2, 2), padding="SAME", name="convt5")
        convt5 = tf.nn.tanh(convt5)

        return convt5

def generator128(attr, z, batch_layer=batch_norm, reuse=False):
    with tf.variable_scope('gen', reuse=tf.AUTO_REUSE):
        # attr = tf.layers.dense(attr, units=256, activation=leaky_relu,
        #                               kernel_initializer=tf.random_normal_initializer(stddev=0.01))

        input_ = tf.concat([attr, z], axis=1)
        input_ = tf.reshape(input_, [-1, 1, 1, input_.get_shape()[1]])

        # (1 x 1 x 512)
        convt1 = convt2d(input_, (None, 4, 4, 512), kernal=(4, 4), strides=(1, 1), padding="VALID", name="convt1")
        convt1 = batch_layer(convt1, name="convt1_bn")
        # convt1 = tf.nn.dropout(convt1, 0.5)
        convt1 = tf.nn.relu(convt1)

        # (2 x 2 x 512)
        convt2 = convt2d(convt1, (None, 8, 8, 256), kernal=(5, 5), strides=(2, 2), padding="SAME", name="convt2")
        convt2 = batch_layer(convt2, name="convt2_bn")
        # convt2 = tf.nn.dropout(convt2, 0.5)
        convt2 = tf.nn.relu(convt2)

        # (4 x 4 x 512)
        convt3 = convt2d(convt2, (None, 32, 32, 128), kernal=(5, 5), strides=(4, 4), padding="SAME", name="convt3")
        convt3 = batch_layer(convt3, name="convt3_bn")
        # convt3 = tf.nn.dropout(convt3, 0.5)
        convt3 = tf.nn.relu(convt3)

        # (8 x 8 x 512)
        convt4 = convt2d(convt3, (None, 64, 64, 64), kernal=(5, 5), strides=(2, 2), padding="SAME", name="convt4")
        convt4 = batch_layer(convt4, name="convt4_bn")
        convt4 = tf.nn.relu(convt4)

        # (16 x 16 x 512)
        convt5 = convt2d(convt4, (None, 128, 128, 3), kernal=(5, 5), strides=(2, 2), padding="SAME", name="convt5")
        convt5 = tf.nn.tanh(convt5)

        return convt5


def generator_mnist(input_, z=None, batch_layer=batch_norm, reuse=False):
    gf_dim = 64

    with tf.variable_scope('gen', reuse=tf.AUTO_REUSE):
        # attr = tf.layers.dense(attr, units=256, activation=leaky_relu,
        #                               kernel_initializer=tf.random_normal_initializer(stddev=0.01))

        if z is not None:
            input_ = tf.concat([input_, z], axis=1)
        input_ = tf.reshape(input_, [-1, 1, 1, input_.get_shape()[1]])

        # (1 x 1 x 512)
        convt1 = convt2d(input_, (None, 4, 4, gf_dim*4), kernal=(4, 4), strides=(1, 1), padding="VALID", name="convt1")
        convt1 = batch_layer(convt1, name="convt1_bn")
        # convt1 = tf.nn.dropout(convt1, 0.5)
        convt1 = tf.nn.relu(convt1)

        # (2 x 2 x 512)
        convt2 = convt2d(convt1, (None, 7, 7, gf_dim*2), kernal=(4, 4), strides=(2, 2), padding="SAME", name="convt2")
        convt2 = batch_layer(convt2, name="convt2_bn")
        # convt2 = tf.nn.dropout(convt2, 0.5)
        convt2 = tf.nn.relu(convt2)

        # (4 x 4 x 512)
        convt3 = convt2d(convt2, (None, 14, 14, gf_dim), kernal=(4, 4), strides=(2, 2), padding="SAME", name="convt3")
        convt3 = batch_layer(convt3, name="convt3_bn")
        # convt3 = tf.nn.dropout(convt3, 0.5)
        convt3 = tf.nn.relu(convt3)

        # (8 x 8 x 512)
        convt4 = convt2d(convt3, (None, 28, 28, 1), kernal=(4, 4), strides=(2, 2), padding="SAME", name="convt4")
        convt4 = tf.nn.tanh(convt4)

        return convt4



def generator(input_, z=None, batch_layer=batch_norm, reuse=False):
    with tf.variable_scope('gen', reuse=tf.AUTO_REUSE):
        # attr = tf.layers.dense(attr, units=256, activation=leaky_relu,
        #                               kernel_initializer=tf.random_normal_initializer(stddev=0.01))
        if z is not None:
            input_ = tf.concat([input_, z], axis=1)
        input_ = tf.reshape(input_, [-1, 1, 1, input_.get_shape()[1]])
        # (1 x 1 x 512)
        convt1 = convt2d(input_, (None, 2, 2, 512), kernal=(4, 4), strides=(2, 2), padding="SAME", name="convt1")
        convt1 = batch_layer(convt1, name="convt1_bn")
        # convt1 = tf.nn.dropout(convt1, 0.5)
        convt1 = tf.nn.relu(convt1)

        # (2 x 2 x 512)
        convt2 = convt2d(convt1, (None, 4, 4, 512), kernal=(4, 4), strides=(2, 2), padding="SAME", name="convt2")
        convt2 = batch_layer(convt2, name="convt2_bn")
        # convt2 = tf.nn.dropout(convt2, 0.5)
        convt2 = tf.nn.relu(convt2)

        # (4 x 4 x 512)
        convt3 = convt2d(convt2, (None, 8, 8, 512), kernal=(4, 4), strides=(2, 2), padding="SAME", name="convt3")
        convt3 = batch_layer(convt3, name="convt3_bn")
        # convt3 = tf.nn.dropout(convt3, 0.5)
        convt3 = tf.nn.relu(convt3)

        # (8 x 8 x 512)
        convt4 = convt2d(convt3, (None, 16, 16, 256), kernal=(4, 4), strides=(2, 2), padding="SAME", name="convt4")
        convt4 = batch_layer(convt4, name="convt4_bn")
        convt4 = tf.nn.relu(convt4)

        # (16 x 16 x 512)
        convt5 = convt2d(convt4, (None, 32, 32, 128), kernal=(4, 4), strides=(2, 2), padding="SAME", name="convt5")
        convt5 = batch_layer(convt5, name="convt5_bn")
        convt5 = tf.nn.relu(convt5)

        # (32 x 32 x 256)
        convt6 = convt2d(convt5, (None, 64, 64, 64), kernal=(4, 4), strides=(2, 2), padding="SAME", name="convt6")
        convt6 = batch_layer(convt6, name="convt6_bn")
        convt6 = tf.nn.relu(convt6)
        #
        # # (64 x 64 x 128)
        convt7 = convt2d(convt6, (None, 128, 128, 3), kernal=(4, 4), strides=(2, 2), padding="SAME", name="convt7")
        convt7 = batch_layer(convt7, name="convt7_bn")
        convt7 = tf.nn.relu(convt7)
        #
        # # (128 x 128 x 64)
        convt8 = convt2d(convt7, (None, 256, 256, 3), kernal=(4, 4), strides=(2, 2), padding="SAME", name="convt8")
        convt8 = tf.nn.tanh(convt8)

        return convt8

def generator_resnet(input_, norm_layer=instance_norm, num_blocks=6, reuse=False):
    with tf.variable_scope('gen_res', reuse=tf.AUTO_REUSE):

        padded_input = tf.pad(input_, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")

        conv1 = conv2d(padded_input, 32, kernal=(7, 7), strides=(1, 1), padding="VALID", name="conv1")
        conv1 = norm_layer(conv1, name="conv1_bn")
        conv1 = tf.nn.relu(conv1)

        conv2 = conv2d(conv1, 64, kernal=(3, 3), strides=(2, 2), padding="SAME", name="conv2")
        conv2 = norm_layer(conv2, name="conv2_bn")
        conv2 = tf.nn.relu(conv2)

        conv3 = conv2d(conv2, 128, kernal=(3, 3), strides=(2, 2), padding="SAME", name="conv3")
        conv3 = norm_layer(conv3, name="conv3_bn")
        conv3 = tf.nn.relu(conv3)

        res_out = conv3
        for r in range(num_blocks):
            res_out = build_residual_block(res_out, 128, norm_layer=norm_layer, use_dropout=False, name="res" + str(r))

        convt1 = convt2d(res_out, (None, 128, 128, 64), kernal=(3, 3), strides=(2, 2), padding="SAME", name="convt1")
        convt1 = norm_layer(convt1, name="convt1_bn")
        convt1 = tf.nn.relu(convt1)

        convt2 = convt2d(convt1, (None, 256, 256, 32), kernal=(3, 3), strides=(2, 2), padding="SAME", name="convt2")
        convt2 = norm_layer(convt2, name="convt2_bn")
        convt2 = tf.nn.relu(convt2)

        padded_output = tf.pad(convt2, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        output = conv2d(padded_output, 3, kernal=(7, 7), strides=(1, 1), padding="VALID", name="output")
        output = tf.tanh(output)

        return output