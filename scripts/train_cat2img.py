import os
import time
import numpy as np
import tensorflow
import dataloader
if tensorflow.__version__ >= '2.0':
    tf = tensorflow.compat.v1
    tf.disable_eager_execution()
    tf.disable_v2_behavior()
else:
    tf = tensorflow

from models.ccoopets_cat2img import CCoopNetsCat2Img

FLAGS = tf.app.flags.FLAGS

tf.flags.DEFINE_integer('image_size', 128, 'Image size to rescale images')
tf.flags.DEFINE_integer('image_nc', 3, 'Image channels')
tf.flags.DEFINE_integer('batch_size', 100, 'Batch size of training images')
tf.flags.DEFINE_integer('num_epochs', 3000, 'Number of epochs to train')
tf.flags.DEFINE_integer('nTileRow', 10, 'Row number of synthesized images')
tf.flags.DEFINE_integer('nTileCol', 10, 'Column number of synthesized images')
tf.flags.DEFINE_float('beta1', 0.5, 'Momentum term of adam')
tf.flags.DEFINE_string('des_net', 'descriptor_attr2img', 'descriptor network')
tf.flags.DEFINE_string('gen_net', 'generator128', 'generator network')

# parameters for descriptorNet
tf.flags.DEFINE_float('d_lr', 0.003, 'Initial learning rate for descriptor')
tf.flags.DEFINE_float('des_refsig', 0.016, 'Standard deviation for reference distribution of descriptor')
tf.flags.DEFINE_integer('des_sample_steps', 10, 'Sample steps for Langevin dynamics of descriptor')
tf.flags.DEFINE_float('des_step_size', 0.002, 'Step size for descriptor Langevin dynamics')

# parameters for generatorNet
tf.flags.DEFINE_float('g_lr', 0.0001, 'Initial learning rate for generator')
tf.flags.DEFINE_float('gen_refsig', 0.3, 'Standard deviation for reference distribution of generator')
tf.flags.DEFINE_integer('gen_sample_steps', 0, 'Sample steps for Langevin dynamics of generator')
tf.flags.DEFINE_float('gen_step_size', 0.1, 'Step size for generator Langevin dynamics')

# parameters for conditioned data
tf.flags.DEFINE_integer('code_size', 100, 'latent dimension of the conditional data')
tf.flags.DEFINE_integer('attr_size', 40, 'Size of attributes')
tf.flags.DEFINE_integer('z_size', 40, 'Size of hidden latent variable')
tf.flags.DEFINE_string('dataroot', './datasets', 'The data directory')
tf.flags.DEFINE_string('category', 'mnist', 'The name of dataset')
tf.flags.DEFINE_string('output_dir', './output', 'The output directory for saving results')
tf.flags.DEFINE_integer('log_step', 5, 'Number of epochs to save output results')
tf.flags.DEFINE_boolean('test', False, 'True if in testing mode')
tf.flags.DEFINE_string('ckpt', None, 'Checkpoint path to load')
tf.flags.DEFINE_integer('sample_size', 144, 'Number of images to generate during test.')

def main(_):
    category = FLAGS.category

    RANDOM_SEED = 1
    np.random.seed(RANDOM_SEED)
    tf.set_random_seed(RANDOM_SEED)

    dataset_path = os.path.join(FLAGS.dataroot, category)
    dataset = None
    if category == 'CUB':
        dataset = dataloader.CUBTextDataSet(dataset_path, cache=os.path.join(dataset_path, 'birds','train_data.h5'),train=True, num_images=1024,
                                img_width=FLAGS.image_size, img_height=FLAGS.image_size, shuffle=True,
                                low=-1, high=1)
    if category == 'LFW':
        dataset = dataloader.LFWDataSet(dataset_path, cache=os.path.join(dataset_path, 'data128.h5'),train= True, num_images=None,
                                img_width=FLAGS.image_size, img_height=FLAGS.image_size, shuffle=True,
                                low=-1, high=1)

    if category == 'CelebA':
        dataset = dataloader.CelebADataSet(dataset_path, cache=os.path.join(dataset_path, 'data128_center_crop.h5'), num_images=20000, 
                                img_width=FLAGS.image_size, img_height=FLAGS.image_size, shuffle=True,
                                low=-1, high=1, crop='center')

    if category == 'mnist':
        FLAGS.image_size = 28
        dataset = dataloader.MNISTDataSet(dataset_path, train= True, num_images=None,
                                img_width=FLAGS.image_size, img_height=FLAGS.image_size, shuffle=True,
                                low=-1, high=1)
    if category == 'Oxford17':
        dataset = dataloader.Oxford17DataSet(dataset_path, cache=os.path.join(dataset_path, 'data64.h5'), 
                                img_width=FLAGS.image_size, img_height=FLAGS.image_size, shuffle=True,
                                low=-1, high=1)

    output_dir = os.path.join(FLAGS.output_dir, '{}_{}'.format(category, time.strftime('%Y-%m-%d_%H-%M-%S')))
    sample_dir = os.path.join(output_dir, 'synthesis')
    log_dir = os.path.join(output_dir, 'log')
    model_dir = os.path.join(output_dir, 'checkpoints')
    test_dir = os.path.join(FLAGS.output_dir, 'test')

    model = CCoopNetsCat2Img(
        dataset_path=dataset_path,
        sample_dir=sample_dir, log_dir=log_dir, model_dir=model_dir, test_dir=test_dir
    )

    with tf.Session() as sess:
        if FLAGS.test:
            tf.gfile.MakeDirs(test_dir)

            model.test(sess, FLAGS.ckpt)
        else:
            tf.gfile.MakeDirs(log_dir)
            tf.gfile.MakeDirs(sample_dir)
            tf.gfile.MakeDirs(model_dir)

            model.train(sess, dataset)


if __name__ == '__main__':
    tf.app.run()