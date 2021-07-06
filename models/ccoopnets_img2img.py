import os
import time
import math
import numpy as np
try:
    import tensorflow.compat.v1 as tf
except ImportError:
    import tensorflow as tf

from models import modules
from .utils.interpolate import *
from .utils.ops import *
from utils.data_io import saveSampleImages, imsave, numpy2image

FLAGS = tf.app.flags.FLAGS

class CCoopNetsImg2Img(object):
    def __init__(self, 
                 src_img_path=None, tgt_img_path=None, category=None,
                 src_test_path=None, tgt_test_path=None, log_step=10,
                 sample_dir='./synthesis', model_dir='./checkpoints', log_dir='./log', test_dir='./test'):

        self.num_epochs = FLAGS.num_epochs
        self.batch_size = FLAGS.batch_size
        self.image_size = FLAGS.image_size
        self.nTileRow = FLAGS.nTileRow
        self.nTileCol = FLAGS.nTileCol
        self.num_chain = self.nTileRow * self.nTileCol
        self.beta1 = FLAGS.beta1

        self.d_lr = FLAGS.d_lr
        self.g_lr = FLAGS.g_lr
        self.delta1 = FLAGS.des_step_size
        self.sigma1 = FLAGS.des_refsig
        self.delta2 = FLAGS.gen_step_size
        self.sigma2 = FLAGS.gen_refsig
        self.t1 = FLAGS.des_sample_steps
        self.t2 = FLAGS.gen_sample_steps

        self.code_size = FLAGS.code_size

        self.src_img_path = src_img_path
        self.tgt_img_path = tgt_img_path
        self.src_test_path = src_test_path
        self.tgt_test_path = tgt_test_path
        self.category = category

        self.log_step = log_step

        self.log_dir = log_dir
        self.sample_dir = sample_dir
        self.model_dir = model_dir
        self.test_dir = test_dir

        self.generator = getattr(modules, FLAGS.gen_net)
        self.descriptor = getattr(modules, FLAGS.des_net)

        self.z_size = 128
        self.gamma = 0.2

        self.syn = tf.placeholder(shape=[None, self.image_size, self.image_size, FLAGS.image_nc], dtype=tf.float32, name='syn')
        self.obs = tf.placeholder(shape=[None, self.image_size, self.image_size, FLAGS.image_nc], dtype=tf.float32, name='obs')
        self.condition = tf.placeholder(shape=[None, self.image_size, self.image_size, FLAGS.image_nc], dtype=tf.float32,
                                        name='condition')
        self.z = tf.placeholder(shape=[None, self.z_size], dtype=tf.float32, name='z')

        self.verbose = False

    def build_model(self):
        self.gen_res = self.generator(self.condition, self.z)

        des_net = {}

        obs_res = self.descriptor(self.obs, self.condition, net=des_net)
        syn_res = self.descriptor(self.syn, self.condition)

        with open("%s/config" % self.sample_dir, "w") as f:
            for k in self.__dict__:
                f.write(str(k) + ':' + str(self.__dict__[k]) + '\n')
            f.write('\ndescriptor:\n')
            for layer in sorted(des_net):
                f.write(str(layer) + ':' + str(des_net[layer]) + '\n')

        self.recon_err = tf.reduce_mean(
            tf.pow(tf.subtract(tf.reduce_mean(self.syn, axis=0), tf.reduce_mean(self.obs, axis=0)), 2))
        # self.recon_err_mean, self.recon_err_update = tf.contrib.metrics.streaming_mean(self.recon_err)

        # descriptor variables
        des_vars = [var for var in tf.trainable_variables() if var.name.startswith('des')]

        self.des_loss = tf.reduce_mean(tf.subtract(tf.reduce_mean(syn_res, axis=0), tf.reduce_mean(obs_res, axis=0)))
        # self.des_loss_mean, self.des_loss_update = tf.contrib.metrics.streaming_mean(self.des_loss)

        des_optim = tf.train.AdamOptimizer(self.d_lr, beta1=self.beta1)
        des_grads_vars = des_optim.compute_gradients(self.des_loss, var_list=des_vars)
        des_grads = [tf.reduce_mean(tf.abs(grad)) for (grad, var) in des_grads_vars if '/w' in var.name]
        # update by mean of gradients
        self.apply_d_grads = des_optim.apply_gradients(des_grads_vars)

        # generator variables
        gen_vars = [var for var in tf.trainable_variables() if var.name.startswith('gen')]

        # self.gen_loss = tf.reduce_mean(1.0 / (2 * self.sigma2 * self.sigma2) * tf.square(self.obs - self.gen_res))
        self.gen_loss = (1 - self.gamma) * tf.reduce_sum(tf.reduce_mean(1.0 / (2 * self.sigma2 * self.sigma2) * tf.square(self.obs - self.gen_res), axis=0)) \
                        + self.gamma * tf.reduce_sum(tf.abs(self.syn - self.gen_res))

        # self.gen_loss_mean, self.gen_loss_update = tf.contrib.metrics.streaming_mean(self.gen_loss)

        gen_optim = tf.train.AdamOptimizer(self.g_lr, beta1=self.beta1)
        gen_grads_vars = gen_optim.compute_gradients(self.gen_loss, var_list=gen_vars)
        gen_grads = [tf.reduce_mean(tf.abs(grad)) for (grad, var) in gen_grads_vars if '/w' in var.name]
        self.apply_g_grads = gen_optim.apply_gradients(gen_grads_vars)

        # symbolic langevins
        self.langevin_conditional_descriptor = self.langevin_dynamics_conditional_descriptor(self.syn, self.condition)

    def langevin_dynamics_conditional_descriptor(self, syn_arg, condition_arg):
        def cond(i, syn, condition):
            return tf.less(i, self.t1)

        def body(i, syn, condition):
            noise = tf.random_normal(shape=tf.shape(syn), name='noise')
            syn_res = self.descriptor(syn, condition)
            grad = tf.gradients(syn_res, syn, name='grad_des')[0]
            syn = syn - 0.5 * self.delta1 * self.delta1 * (syn / self.sigma1 / self.sigma1 - grad)
            return tf.add(i, 1), syn, condition

        with tf.name_scope("langevin_dynamics_descriptor"):
            i = tf.constant(0)
            i, syn, condition = tf.while_loop(cond, body, [i, syn_arg, condition_arg])
            return syn

    def train(self, sess, train_data):
        self.build_model()

        num_batches = int(math.ceil(len(train_data) / self.batch_size))

        # initialize training
        sess.run(tf.global_variables_initializer())
        # sess.run(tf.local_variables_initializer())

        # sample_results = np.random.randn(self.num_chain * num_batches, self.image_size, self.image_size, 3)

        saver = tf.train.Saver(max_to_keep=50)

        writer = tf.summary.FileWriter(self.log_dir, sess.graph)

        # make graph immutable
        tf.get_default_graph().finalize()

        sample_results = np.random.randn(train_data.num_images, train_data.img_size, train_data.img_size, train_data.img_nc)
        saveSampleImages(train_data.src_images, "%s/ref_condition.png" % self.sample_dir,
                         row_num=self.nTileRow, col_num=self.nTileCol)
        saveSampleImages(train_data.tgt_images, "%s/ref_target.png" % self.sample_dir,
                         row_num=self.nTileRow, col_num=self.nTileCol)


        img_summary = image_summary('ref_condition', train_data.src_images, row_num=self.nTileRow,
                                 col_num=self.nTileCol)

        writer.add_summary(img_summary)
        writer.flush()

        img_summary = image_summary('ref_target', train_data.tgt_images, row_num=self.nTileRow,
                                 col_num=self.nTileCol)

        writer.add_summary(img_summary)
        writer.flush()


        # train
        for epoch in range(self.num_epochs):
            start_time = time.time()
            log = {}

            d_loss_epoch, g_loss_epoch, mse_epoch = [], [], []

            for i in range(num_batches):

                index = slice(i * self.batch_size, min(len(train_data), (i + 1) * self.batch_size))
                src_img, tgt_img = train_data[index]

                # Step G0: generate X ~ N(0, 1)
                z_vec = np.random.normal(size=(len(src_img), self.z_size), scale=0.003)
                g_res = sess.run(self.gen_res, feed_dict={self.condition: src_img, self.z: z_vec})

                # fc = sess.run(self.fc, feed_dict={self.z: z_vec, self.condition: src_img})
                # print(fc.max())
                # Step D1: obtain synthesized images Y
                syn = sess.run(self.langevin_conditional_descriptor,
                               feed_dict={self.syn: g_res, self.condition: src_img})
                # Step G1: update X using Y as training image
                # if self.t2 > 0:
                #     z_vec = sess.run(self.langevin_conditional_generator, feed_dict={self.z: z_vec, self.obs: syn, self.condition: src_img})
                # Step D2: update D net
                d_loss = sess.run([self.des_loss,  self.apply_d_grads],
                                  feed_dict={self.obs: tgt_img, self.syn: syn, self.condition: src_img})[0]
                # Step G2: update G net
                g_loss = sess.run([self.gen_loss, self.apply_g_grads],
                                  feed_dict={self.obs: syn, self.condition: src_img, self.z: z_vec, self.syn: tgt_img})[0]

                # Compute MSE
                mse = sess.run(self.recon_err,
                               feed_dict={self.obs: tgt_img, self.syn: syn})
                d_loss_epoch.append(d_loss)
                g_loss_epoch.append(g_loss)
                mse_epoch.append(mse)
                sample_results[index] = syn
                if self.verbose:
                    print('Epoch #{:d}, [{:2d}]/[{:2d}], descriptor loss: {:.4f}, generator loss: {:.4f}, '
                          'L2 distance: {:4.4f}'.format(epoch, i + 1, num_batches, d_loss, g_loss, mse))

            log['des_loss_avg'], log['gen_loss_avg'], log['mse_avg'] = np.mean(d_loss_epoch), np.mean(g_loss_epoch), np.mean(mse_epoch)

            end_time = time.time()
            print('Epoch #{:d}, avg.descriptor loss: {:.4f}, avg.generator loss: {:.4f}, avg.L2 distance: {:4.4f}, '
                  'time: {:.2f}s'.format(epoch, log['des_loss_avg'], log['gen_loss_avg'], log['mse_avg'], end_time - start_time))
            
            if np.isnan(log['mse_avg']) or  log['mse_avg'] > 2:
                break
            
            for tag, value in log.items():
                summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
                writer.add_summary(summary, epoch)

            if epoch % self.log_step == 0:
                if not os.path.exists(self.model_dir):
                    os.makedirs(self.model_dir)
                saver.save(sess, "%s/%s" % (self.model_dir, 'model.ckpt'), global_step=epoch)

                img_summary = image_summary('sample', sample_results, row_num=self.nTileRow,
                                 col_num=self.nTileCol)

                writer.add_summary(img_summary, epoch)
                writer.flush()

                if not os.path.exists(self.sample_dir):
                    os.makedirs(self.sample_dir)
                saveSampleImages(sample_results, "%s/des%03d.png" % (self.sample_dir, epoch), row_num=self.nTileRow,
                                 col_num=self.nTileCol)


    def test(self, sess, ckpt, test_data):
        assert ckpt is not None, 'no checkpoint provided.'

        gen_res = self.generator(self.condition, noise=self.z, reuse=False)
        obs_res = self.descriptor(self.obs, self.condition, reuse=False)
        langevin_conditional_descriptor = self.langevin_dynamics_conditional_descriptor(self.syn, self.condition)

        num_batches = int(math.ceil(len(test_data) / self.batch_size))

        saver = tf.train.Saver()

        # sess.run(tf.global_variables_initializer())
        saver.restore(sess, ckpt)
        print('Loading checkpoint {}.'.format(ckpt))

        saveSampleImages(test_data.src_images, "%s/ref_src.png" % self.test_dir, row_num=self.nTileRow,
                         col_num=self.nTileCol, save_all=True)
        saveSampleImages(test_data.tgt_images, "%s/ref_tgt.png" % self.test_dir, row_num=self.nTileRow,
                         col_num=self.nTileCol, save_all=True)
        samples = np.zeros(shape=(len(test_data), self.image_size, self.image_size, 3))
        g_results = np.zeros(shape=(len(test_data), self.image_size, self.image_size, 3))

        for i in range(num_batches):
            index = slice(i * self.batch_size, min(len(test_data), (i + 1) * self.batch_size))
            src_img, tgt_img = test_data[index]

            z_vec = np.random.normal(size=(len(src_img), self.z_size), scale=1.0)
            # print(src_img.shape)
            g_res = sess.run(gen_res, feed_dict={self.condition: src_img, self.z: z_vec})
            syn = sess.run(langevin_conditional_descriptor,
                           feed_dict={self.syn: g_res, self.condition: src_img})

            samples[index] = syn
            g_results[index] = g_res
        saveSampleImages(samples, "%s/des.png" % self.test_dir, row_num=self.nTileRow, col_num=self.nTileCol, save_all=True)
        saveSampleImages(g_results, "%s/gen.png" % self.test_dir, row_num=self.nTileRow, col_num=self.nTileCol, save_all=True)

        if self.category == 'cityscapes':
            pred_dir = os.path.join(self.test_dir, 'result_val')

            if not os.path.exists(pred_dir):
                os.makedirs(pred_dir)

            pred_list = sorted([f for f in os.listdir(self.tgt_test_path) if f.endswith('.png')])

            for i, np_arr in enumerate(samples):
                img = numpy2image(np_arr)
                imsave(os.path.join(pred_dir, pred_list[i]), img)