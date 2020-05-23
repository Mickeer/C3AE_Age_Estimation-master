#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import tensorflow as tf
import sys
sys.path.append('..')
from models.run_net import C3AENet
from prepare_data.gen_data_batch import gen_data_batch
from config import cfg
import os
import tensorflow.contrib.slim as slim

gpu_list = np.arange(cfg.train.num_gpus)
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(i) for i in gpu_list)  #设置当前使用的GPU设备

def get_variables_to_restore(include_vars=[], exclude_global_pool=False):    #恢复每层变量
    variables_to_restore = []
    for var in slim.get_model_variables():
        if exclude_global_pool and 'global_pool' in var.op.name:
            continue
        variables_to_restore.append(var)
    for var in slim.get_variables_to_restore(include=include_vars):
        if exclude_global_pool and 'global_pool' in var.op.name:
            continue
        variables_to_restore.append(var)
    return variables_to_restore


def average_gradients(tower_grads):
    """计算所有towers上每个共享变量的                 →→→→→→平均梯度。

    请注意，此功能在所有towers上提供了一个同步点。

    Args:
          tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
          List of pairs of (gradient, variable) where the gradient has been averaged
        across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def train(finetune):
    is_training = True

    # data pipeline   批量引入数据
    imgs, age_labels, age_vectors = gen_data_batch(cfg.data_path, cfg.batch_size*cfg.train.num_gpus)  #配置函数
    imgs = tf.reshape(imgs, (-1, imgs.get_shape()[2], imgs.get_shape()[3], imgs.get_shape()[4]))      #将imgs处理成参数shape的列表形式
    imgs_split = tf.split(imgs, cfg.train.num_gpus)                                                   #图片分割
    age_labels_split = tf.split(age_labels, cfg.train.num_gpus)
    age_vectors_split = tf.split(age_vectors, cfg.train.num_gpus)

    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0.), trainable=False)
    learn_rate_decay_step = int(cfg.train.num_samples / cfg.batch_size / cfg.train.num_gpus * cfg.train.learn_rate_decay_epoch)
    learning_rate = tf.train.exponential_decay(cfg.train.learn_rate, global_step, learn_rate_decay_step, cfg.train.learn_rate_decay, staircase=True)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    # Calculate the gradients for each model tower.  计算每个模型tower的梯度。
    tower_grads = []
    with tf.variable_scope(tf.get_variable_scope()):
        for i in range(cfg.train.num_gpus):
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('%s_%d' % (cfg.train.tower, i)) as scope:
                    model = C3AENet(imgs_split[i], age_labels_split[i], age_vectors_split[i], is_training)
                    loss = model.compute_loss()
                    tf.get_variable_scope().reuse_variables()
                    grads = optimizer.compute_gradients(loss)
                    tower_grads.append(grads)
                    if i == 0:
                        current_loss = loss
                        update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    grads = average_gradients(tower_grads)
    with tf.control_dependencies(update_op):                #tf.control_dependencies是控制先update_op操作再计算的函数
        apply_gradient_op = optimizer.apply_gradients(grads, global_step=global_step)
        train_op = tf.group(apply_gradient_op,*update_op)

    # GPU config
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # Create a saver     使用Saver类保存模型的参数
    saver = tf.train.Saver(max_to_keep=1000)
    ckpt_dir = cfg.ckpt_path

    # init  初始化模型的参数
    sess.run(tf.global_variables_initializer())
    if finetune:
        checkpoint = './ckpt/pre_train.ckpt'

        variables_to_restore = get_variables_to_restore(exclude_global_pool=True)
        #variables_to_restore是TensorFlow为滑动平均值提供的一个函数，通过使用滑动平均值可以让神经网络模型更加的健壮
        init_assign_op, init_feed_dict = slim.assign_from_checkpoint(checkpoint, variables_to_restore, ignore_missing_vars=True)
        sess.run(init_assign_op, init_feed_dict)

    # running
    cnt_epoch = 0

    for i in range(1, cfg.train.max_batches):
        _, loss_, lr_ = sess.run([train_op, current_loss, learning_rate])  #训练节点、损失、学习率 进行训练
        if(i % 10 == 0):
            print(i,': ', loss_, '          lr: ', lr_)
        if int(i) % int(cfg.train.num_samples / cfg.train.num_gpus / cfg.batch_size) == 0:
            cnt_epoch += 1
            saver.save(sess, ckpt_dir+'C3AEDet', global_step=cnt_epoch, write_meta_graph=True)

if __name__ == '__main__':
    train(finetune=False)
