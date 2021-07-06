try:
    import tensorflow.compat.v1 as tf
except ImportError:
    import tensorflow as tf
try:
    from StringIO import StringIO
except ImportError:
    from io import BytesIO
from PIL import Image
from utils.data_io import img2cell

__all__ = [
    'image_summary',
    'batch_norm',
    'instance_norm',
    'conv2d',
    'convt2d',
    'build_residual_block',
    'fully_connected',
    'linear'
]


def image_summary(tag, images, row_num=10, col_num=10, margin_syn=2):
    cell_images = img2cell(images, row_num=row_num, col_num=col_num, margin_syn=margin_syn)
    cell_image = cell_images[0]
    try:
        s = StringIO()
    except:
        s = BytesIO()
    Image.fromarray(cell_image).save(s, format="png")
    img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(), height=cell_image.shape[0], width=cell_image.shape[1])
    return tf.Summary(value=[tf.Summary.Value(tag=tag, image=img_sum)])


def batch_norm(x, train=True, name="batch_norm"):
    return tf.layers.batch_normalization(x,
                                        momentum=0.9,
                                        epsilon=1e-5,
                                        scale=True,
                                        training=train, name=name)


def instance_norm(input_, name="instance_norm"):
    with tf.variable_scope(name):
        depth = input_.get_shape()[3]
        scale = tf.get_variable("scale", [depth],
                                initializer=tf.random_normal_initializer(stddev=0.01, dtype=tf.float32))
        offset = tf.get_variable("offset", [depth], initializer=tf.constant_initializer(0.0))
        mean, variance = tf.nn.moments(input_, axes=[1, 2], keep_dims=True)
        epsilon = 1e-5
        inv = tf.rsqrt(variance + epsilon)
        normalized = (input_ - mean) * inv
        return scale * normalized + offset


def leaky_relu(input_, leakiness=0.2):
    assert leakiness <= 1
    return tf.maximum(input_, leakiness * input_)


def conv2d(input_, output_dim, kernal=(5, 5), strides=(2, 2), padding='SAME', activate_fn=None, name="conv2d"):
    if type(kernal) == list or type(kernal) == tuple:
        [k_h, k_w] = list(kernal)
    else:
        k_h = k_w = kernal
    if type(strides) == list or type(strides) == tuple:
        [d_h, d_w] = list(strides)
    else:
        d_h = d_w = strides

    with tf.variable_scope(name):
        if type(padding) == list or type(padding) == tuple:
            padding = [0] + list(padding) + [0]
            input_ = tf.pad(input_, [[p, p] for p in padding], "CONSTANT")
            padding = 'VALID'

        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.random_normal_initializer(stddev=0.01))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding=padding)
        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.bias_add(conv, biases)
        if activate_fn:
            conv = activate_fn(conv)
        return conv


def fully_connected(input_, output_dim, name="fc"):
    shape = input_.shape
    return conv2d(input_, output_dim, kernal=list(shape[1:3]), strides=(1, 1), padding="VALID", name=name)


def build_residual_block(input_, dim, norm_layer=batch_norm, use_dropout=False, name="residule_block"):
    with tf.variable_scope(name):
        conv_block = tf.pad(input_, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
        conv_block = conv2d(conv_block, dim, kernal=(3, 3), strides=(1, 1), padding="VALID", name=name + "_c1")
        conv_block = tf.nn.relu(norm_layer(conv_block, name=name + "_bn1"))
        if use_dropout:
            conv_block = tf.nn.dropout(conv_block, 0.5)
        conv_block = tf.pad(conv_block, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
        conv_block = conv2d(conv_block, dim, kernal=(3, 3), strides=(1, 1), padding="VALID", name=name + "_c2")
        conv_block = norm_layer(conv_block, name=name + "_bn2")
        return conv_block + input_


def convt2d(input_, output_shape, kernal=(5, 5), strides=(2, 2), padding='SAME', activate_fn=None, name="convt2d"):
    assert type(kernal) in [list, tuple, int]
    assert type(strides) in [list, tuple, int]
    assert type(padding) in [list, tuple, int, str]
    if type(kernal) == list or type(kernal) == tuple:
        [k_h, k_w] = list(kernal)
    else:
        k_h = k_w = kernal
    if type(strides) == list or type(strides) == tuple:
        [d_h, d_w] = list(strides)
    else:
        d_h = d_w = strides
    output_shape = list(output_shape)
    output_shape[0] = tf.shape(input_)[0]
    with tf.variable_scope(name):
        if type(padding) in [tuple, list, int]:
            if type(padding) == int:
                p_h = p_w = padding
            else:
                [p_h, p_w] = list(padding)
            pad_ = [0, p_h, p_w, 0]
            input_ = tf.pad(input_, [[p, p] for p in pad_], "CONSTANT")
            padding = 'VALID'

        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=0.01))
        convt = tf.nn.conv2d_transpose(input_, w, output_shape=tf.stack(output_shape, axis=0), strides=[1, d_h, d_w, 1],
                                       padding=padding)
        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        convt = tf.nn.bias_add(convt, biases)
        if activate_fn:
            convt = activate_fn(convt)
        return convt


def linear(inputs, out_dims, name='linear', stddev=0.02, use_bias=True):
    with tf.variable_scope(name or 'linear'):
        weights = tf.get_variable('weights', [inputs.get_shape()[-1], out_dims], dtype=tf.float32, initializer=tf.random_normal_initializer(stddev=stddev))
        out = tf.matmul(inputs, weights)
        if use_bias:
            bias = tf.get_variable('bias', [out_dims], dtype=tf.float32, initializer=tf.zeros_initializer)
            out = tf.nn.bias_add(out, bias)
        return out