import os
import tensorflow
if tensorflow.__version__ >= '2.0':
    tf = tensorflow.compat.v1
    tf.disable_eager_execution()
    tf.disable_v2_behavior()
else:
    tf = tensorflow
from models.ccoopnets_img2img import CCoopNetsImg2Img
import time
from dataloader import AlignedDataSet

FLAGS = tf.app.flags.FLAGS

tf.flags.DEFINE_integer('load_size', 154, 'Image size to load images')
tf.flags.DEFINE_integer('image_size', 128, 'Image size to rescale images')
tf.flags.DEFINE_integer('image_nc', 3, 'Image channels')

tf.flags.DEFINE_integer('batch_size', 50, 'Batch size of training images')
tf.flags.DEFINE_integer('num_epochs', 3000, 'Number of epochs to train')
tf.flags.DEFINE_integer('nTileRow', 10, 'Row number of synthesized images')
tf.flags.DEFINE_integer('nTileCol', 10, 'Column number of synthesized images')
tf.flags.DEFINE_float('beta1', 0.5, 'Momentum term of adam')
tf.flags.DEFINE_integer('gpu', 0, 'GPU id')

# parameters for descriptorNet
tf.flags.DEFINE_float('d_lr', 0.003, 'Initial learning rate for descriptor')
tf.flags.DEFINE_float('des_refsig', 0.016, 'Standard deviation for reference distribution of descriptor')
tf.flags.DEFINE_integer('des_sample_steps', 20, 'Sample steps for Langevin dynamics of descriptor')
tf.flags.DEFINE_float('des_step_size', 0.002, 'Step size for descriptor Langevin dynamics')
tf.flags.DEFINE_string('des_net', 'descriptor_img2img', 'descriptor network')
tf.flags.DEFINE_string('gen_net', 'generator_resnet', 'generator network')

# parameters for generatorNet
tf.flags.DEFINE_float('g_lr', 0.0002, 'Initial learning rate for generator')
tf.flags.DEFINE_float('gen_refsig', 0.3, 'Standard deviation for reference distribution of generator')
tf.flags.DEFINE_integer('gen_sample_steps', 10, 'Sample steps for Langevin dynamics of generator')
tf.flags.DEFINE_float('gen_step_size', 0.1, 'Step size for generator Langevin dynamics')

# parameters for conditioned data
tf.flags.DEFINE_integer('code_size', 100, 'latent dimension of the conditional data')

tf.flags.DEFINE_string('dataroot', './datasets', 'The data directory')
tf.flags.DEFINE_string('category', 'cityscapes', 'The name of dataset')
tf.flags.DEFINE_string('output_dir', './output', 'The output directory for saving results')
tf.flags.DEFINE_integer('log_step', 10, 'Number of epochs to save output results')
tf.flags.DEFINE_boolean('test', False, 'True if in testing mode')
tf.flags.DEFINE_string('ckpt', None, 'Checkpoint path to load')
tf.flags.DEFINE_integer('sample_size', 144, 'Number of images to generate during test.')


def main(_):
    category = FLAGS.category

    src_img_path, tgt_img_path, src_test_path, tgt_test_path = None, None, None, None
    datapath = os.path.join(FLAGS.dataroot, category)
    if category == 'cityscapes':
        src_img_path = os.path.join(datapath, 'train/gtFine/')
        tgt_img_path = os.path.join(datapath, 'train/leftImg8bit/')
        src_test_path = os.path.join(datapath, 'val/gtFine/')
        tgt_test_path = os.path.join(datapath, 'val/leftImg8bit/')
    elif category == 'CMP':
        src_img_path= os.path.join(datapath, 'base/facade')
        tgt_img_path= os.path.join(datapath, 'base/photos')
    elif category == 'CUHK':
        src_img_path= os.path.join(datapath, 'train/cropped_sketches')
        tgt_img_path= os.path.join(datapath, 'train/cropped_photos')
        src_test_path= os.path.join(datapath, 'test/cropped_sketches')
        tgt_test_path= os.path.join(datapath, 'test/cropped_photos')
    elif category == 'UT_ZAP50K':
        src_img_path= os.path.join(datapath, 'ut-zap50k-images-edges/Slippers/train/edges')
        tgt_img_path= os.path.join(datapath, 'ut-zap50k-images-edges/Slippers/train/photos')
        src_test_path= os.path.join(datapath, 'ut-zap50k-images-edges/Slippers/test/edges')
        tgt_test_path= os.path.join(datapath, 'ut-zap50k-images-edges/Slippers/test/photos')

    output_dir = os.path.join(FLAGS.output_dir, '{}_{}'.format(category, time.strftime('%Y-%m-%d_%H-%M-%S')))
    sample_dir = os.path.join(output_dir, 'synthesis')
    log_dir = os.path.join(output_dir, 'log')
    model_dir = os.path.join(output_dir, 'checkpoints')
    test_dir = os.path.join(FLAGS.output_dir, 'test')

    model = CCoopNetsImg2Img(
        src_img_path=src_img_path, tgt_img_path=tgt_img_path,
        src_test_path=src_test_path, tgt_test_path=tgt_test_path, category=category,
        sample_dir=sample_dir, log_dir=log_dir, model_dir=model_dir, test_dir=test_dir
    )

    gpu_options = tf.GPUOptions(visible_device_list=str(FLAGS.gpu), allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_options)
    with tf.Session(config=config) as sess:
        if FLAGS.test:
            tf.gfile.MakeDirs(test_dir)
            dataset = AlignedDataSet(
                src_test_path, 
                tgt_test_path, 
                cache=os.path.join(datapath, 'data_test.h5'), 
                random_flip=False, 
                random_crop=False, 
                load_size=FLAGS.image_size, 
                img_nc=FLAGS.image_nc, 
                shuffle=False
            )

            model.test(sess, FLAGS.ckpt, dataset)
        else:
            tf.gfile.MakeDirs(log_dir)
            tf.gfile.MakeDirs(sample_dir)
            tf.gfile.MakeDirs(model_dir)

            dataset = AlignedDataSet(
                src_img_path, 
                tgt_img_path, 
                cache=os.path.join(datapath, 'data_train.h5'), 
                random_flip=True, 
                random_crop=True, 
                load_size=FLAGS.load_size, 
                img_nc=FLAGS.image_nc, 
                crop_size=FLAGS.image_size, 
                shuffle=True
            )

            model.train(sess, dataset)


if __name__ == '__main__':
    tf.app.run()
