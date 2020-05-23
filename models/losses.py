#!/usr/bin/env python
# encoding: utf-8

import tensorflow as tf
import sys
sys.path.append('..')
from config import cfg

def l1_loss(preds, labels):        #L1损失
    preds = tf.reshape(preds, (cfg.batch_size,))  #预测
    # print("preds", preds.get_shape())
    labels = tf.reshape(labels, (cfg.batch_size,)) #标签
    # print("labels", labels.get_shape())
    l1_loss = tf.abs(labels - preds)   #L1损失计算方式
    # print("l1_loss", l1_loss.get_shape())
    _, k_idx = tf.nn.top_k(l1_loss, tf.cast(cfg.batch_size * cfg.ohem_ratio, tf.int32)) #求最大损失和索引位置
    loss = tf.gather(l1_loss, k_idx)
    return tf.reduce_mean(loss)

def kl_loss(preds, labels, l1):   #KL损失
    kl_loss = tf.reduce_sum(labels * tf.log(preds + 1e-10), axis=1)  #按照横向（1）求和的方式对矩阵降维
    # print("kl_loss", kl_loss.get_shape())
    kl_loss = tf.reshape(kl_loss, (cfg.batch_size,))
    # print("kl_loss", kl_loss.get_shape())
    _, k_idx = tf.nn.top_k(kl_loss, tf.cast(cfg.batch_size * cfg.ohem_ratio, tf.int32))
    loss = tf.gather(kl_loss, k_idx)
    return -tf.reduce_mean(loss) + l1
