try:
    import tensorflow.compat.v1 as tf
except ImportError:
    import tensorflow as tf
from .utils.ops import *

def get_norm_layer(norm_type):
    if norm_type == 'batch_norm':
        return batch_norm
    elif norm_type == 'instance_norm':
        return instance_norm
    elif norm_type == 'none':
        return lambda x: x
    else:
        raise NotImplementedError(
            'Normalization type %s is not implemented.' % norm_type)


def descriptor_mnist(inputs, attr, norm_type='batch_norm', net={}):
    norm_layer = get_norm_layer(norm_type)
    with tf.variable_scope('des', reuse=tf.AUTO_REUSE):
        df_dim = 64
        gf_dim  = 64
        # input ConvNet structure
        net['attr'] = tf.reshape(attr, [-1, 1, 1, attr.get_shape()[1]])

        convt1 = convt2d(net['attr'], (None, 4, 4, gf_dim*4), kernal=(4, 4), strides=(1, 1), padding="VALID", name="convt1")
        convt1 = norm_layer(convt1, name="convt1_bn")
        # convt1 = tf.nn.dropout(convt1, 0.5)
        convt1 = tf.nn.relu(convt1)

        # (2 x 2 x 512)
        convt2 = convt2d(convt1, (None, 7, 7, gf_dim*2), kernal=(4, 4), strides=(2, 2), padding="SAME", name="convt2")
        convt2 = norm_layer(convt2, name="convt2_bn")
        # convt2 = tf.nn.dropout(convt2, 0.5)
        convt2 = tf.nn.relu(convt2)

        # (4 x 4 x 512)
        convt3 = convt2d(convt2, (None, 14, 14, gf_dim), kernal=(4, 4), strides=(2, 2), padding="SAME", name="convt3")
        convt3 = norm_layer(convt3, name="convt3_bn")
        # convt3 = tf.nn.dropout(convt3, 0.5)
        convt3 = tf.nn.relu(convt3)

        # (8 x 8 x 512)
        convt4 = convt2d(convt3, (None, 28, 28, 1), kernal=(4, 4), strides=(2, 2), padding="SAME", name="convt4")
        convt4 = tf.nn.tanh(convt4)


        net['enc'] = tf.concat([convt4, inputs], axis=3)

        net['conv1'] = conv2d(net['enc'], df_dim, kernal=(4, 4), strides=(2, 2), padding="SAME", name="conv1")
        net['conv1'] = tf.nn.leaky_relu(net['conv1'])

        net['conv2'] = conv2d(net['conv1'], df_dim*2, kernal=(4, 4), strides=(2, 2), padding="SAME", name="conv2")
        net['conv2'] = tf.nn.leaky_relu(net['conv2'])

        net['conv3'] = conv2d(net['conv2'], df_dim*4, kernal=(4, 4), strides=(2, 2), padding="SAME", name="conv3")
        net['conv3'] = tf.nn.leaky_relu(net['conv3'])
        #
        net['fc'] = fully_connected(net['conv3'], 100, name="fc")

        #print(net['fc'])

        return net['fc']


def descriptor_img2img(inputs, condition, norm_type='none', net={}):
    with tf.variable_scope('des', reuse=tf.AUTO_REUSE):
        inputs_cond = tf.concat([condition, inputs], axis=3)

        # input ConvNet structure
        net['conv1'] = conv2d(inputs_cond, 64, kernal=(4, 4), strides=(2, 2), padding="SAME", name="conv1")
        # conv1 = batch_layer(conv1, name="conv1_bn")
        net['conv1'] = tf.nn.leaky_relu(net['conv1'])

        net['conv2'] = conv2d(net['conv1'], 128, kernal=(4, 4), strides=(2, 2), padding="SAME", name="conv2")
        # conv2 = batch_layer(conv2, name="conv2_bn")
        net['conv2'] = tf.nn.leaky_relu(net['conv2'])

        net['conv3'] = conv2d(net['conv2'], 256, kernal=(4, 4), strides=(2, 2), padding="SAME", name="conv3")
        # conv3 = batch_layer(conv3, name="conv3_bn")
        net['conv3'] = tf.nn.leaky_relu(net['conv3'])

        net['conv4'] = conv2d(net['conv3'], 512, kernal=(4, 4), strides=(2, 2), padding="SAME", name="conv4")
        # conv4 = batch_layer(conv4, name="conv4_bn")
        net['conv4'] = tf.nn.leaky_relu(net['conv4'])

        # net['conv5'] = conv2d(net['conv4'], 512, kernal=(3, 3), strides=(1, 1), padding="SAME", name="conv5")
        # conv4 = batch_layer(conv4, name="conv4_bn")
        # net['conv5'] = leaky_relu(net['conv5'] )

        net['fc'] = fully_connected(net['conv4'] , 100, name="fc")

        return net['fc']


def generator_mnist(input_, z=None, norm_type='batch_norm'):
    gf_dim = 64
    norm_layer = get_norm_layer(norm_type)

    with tf.variable_scope('gen', reuse=tf.AUTO_REUSE):
        # attr = tf.layers.dense(attr, units=256, activation=leaky_relu,
        #                               kernel_initializer=tf.random_normal_initializer(stddev=0.01))

        if z is not None:
            input_ = tf.concat([input_, z], axis=1)
        input_ = tf.reshape(input_, [-1, 1, 1, input_.get_shape()[1]])

        # (1 x 1 x 512)
        convt1 = convt2d(input_, (None, 4, 4, gf_dim*4), kernal=(4, 4), strides=(1, 1), padding="VALID", name="convt1")
        convt1 = norm_layer(convt1, name="convt1_bn")
        # convt1 = tf.nn.dropout(convt1, 0.5)
        convt1 = tf.nn.relu(convt1)

        # (2 x 2 x 512)
        convt2 = convt2d(convt1, (None, 7, 7, gf_dim*2), kernal=(4, 4), strides=(2, 2), padding="SAME", name="convt2")
        convt2 = norm_layer(convt2, name="convt2_bn")
        # convt2 = tf.nn.dropout(convt2, 0.5)
        convt2 = tf.nn.relu(convt2)

        # (4 x 4 x 512)
        convt3 = convt2d(convt2, (None, 14, 14, gf_dim), kernal=(4, 4), strides=(2, 2), padding="SAME", name="convt3")
        convt3 = norm_layer(convt3, name="convt3_bn")
        # convt3 = tf.nn.dropout(convt3, 0.5)
        convt3 = tf.nn.relu(convt3)

        # (8 x 8 x 512)
        convt4 = convt2d(convt3, (None, 28, 28, 1), kernal=(4, 4), strides=(2, 2), padding="SAME", name="convt4")
        convt4 = tf.nn.tanh(convt4)

        return convt4

def generator_resnet(input_, z=None, norm_type='instance_norm', num_blocks=9):
    norm_layer = get_norm_layer(norm_type)
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

def generator_resnet128(input_, noise=None, norm_type='instance_norm', num_blocks=6):
    norm_layer = get_norm_layer(norm_type)

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

        if noise is not None:
            noise = tf.reshape(noise, [-1, 1, 1, noise.get_shape()[1]])
            res_out = tf.add(res_out, noise)

        convt1 = convt2d(res_out, (None, 64, 64, 64), kernal=(3, 3), strides=(2, 2), padding="SAME", name="convt1")
        convt1 = norm_layer(convt1, name="convt1_bn")
        convt1 = tf.nn.relu(convt1)

        convt2 = convt2d(convt1, (None, 128, 128, 32), kernal=(3, 3), strides=(2, 2), padding="SAME", name="convt2")
        convt2 = norm_layer(convt2, name="convt2_bn")
        convt2 = tf.nn.relu(convt2)

        padded_output = tf.pad(convt2, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        output = conv2d(padded_output, 3, kernal=(7, 7), strides=(1, 1), padding="VALID", name="output")
        output = tf.tanh(output)

        return output