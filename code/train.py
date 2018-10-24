# -*- coding: utf-8 -*-

'''
Training
'''

import mxnet as mx
from mxnet import autograd, contrib, gluon, image, init, nd
from mxnet.gluon import loss as gloss, nn
from resnet import *
import os,argparse,time,logging
from mxnet.gluon.block import HybridBlock
import numpy as np


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Training.')
    parser.add_argument('--rec', dest='rec_path', help='Path to rec file', 
                        default='None',type=str)
    parser.add_argument('--lst', dest='lst_path', help='Path to idx file',
                        default='None',type=str)
    parser.add_argument('--batch_size', dest='batch_size', help='Batch size',
                        default=256, type=int)
    parser.add_argument('--epoch', dest='epoch_num', help='epoch num',
                        default=26, type=int)
    parser.add_argument('--dev', dest='dev', help='0:cpu,1:gpu',
                        default=1, type=int)
    parser.add_argument('--save', dest='save_path', help='save path', 
                        default='/model/',type=str)
    args = parser.parse_args()
    return args

def stable_softmax(z):
    z = nd.exp(z - nd.max(z, axis=1, keepdims=True))
    return z / mx.nd.sum(z, axis=1).reshape((batch_size, 1))

def bin_label(z):
    # Real label to Bin label
    z = (z + 102) / 3.0
    z = nd.ceil(z)
    return z

def calc_loss(pred, label):

    softmax_pred = stable_softmax(pred) + NEAR_0
    label_bin = bin_label(label)

    expectation_bin = nd.sum(idx_tensor * softmax_pred,1)
    expectation = expectation_bin * 3 - 102

    # classification loss : softmax loss
    cls = cls_loss(softmax_pred, label_bin)
    # expectation loss  : calculate the real L2 loss
    reg = reg_loss(expectation,label)
    
    loss = cls + reg
    return loss

if __name__ == '__main__':
    args = parse_args()
    train_rec = args.rec_path
    train_lst = args.lst_path
    batch_size = args.batch_size
    save_path = args.save_path
    
    resize_w, resize_h = 48, 48
    channel = 1
    begin_epoch = 1
    
    # data
    train_data = mx.io.ImageRecordIter(
      path_imgrec=train_rec,
      path_imglist=train_lst,
      label_width=3,
      label_name=['pitch_label', 'roll_label', 'yaw_label'],
      data_shape=(channel, resize_h, resize_w),
      batch_size=batch_size,
      shuffle=True)
    
    # backbone network
    #net = resnet18()
    net = classical()

    # init
    epoch_num = args.epoch_num
    
    dev = args.dev
    if dev == -1:
      ctx = mx.cpu()
    else:
      ctx = mx.gpu(dev)
      
    net.initialize(init=init.Xavier(), ctx=ctx)
    net.hybridize()
    net.collect_params().reset_ctx(ctx)
    trainer = gluon.Trainer(net.collect_params(),
                            'adam', {'learning_rate': 0.001, 'wd': 5e-4})

    lr_period = int(epoch_num / 3)
    lr_decay = 0.1
    NEAR_0 = 1e-10
    idx_tensor = nd.array([idx for idx in range(1,67)]).as_in_context(ctx)
    
    cls_loss = gloss.SoftmaxCrossEntropyLoss()
    reg_loss = gloss.L2Loss()
    
    # training
    for epoch in range(epoch_num):
        train_data.reset()
        tic = time.time()
        i = 0

        # Dynamic lr
        if epoch > 0 and epoch % lr_period == 0:
            trainer.set_learning_rate(trainer.learning_rate * lr_decay)
        
        for i, batch in enumerate(train_data):
            with autograd.record():
              output = net(batch.data[0].as_in_context(ctx))
              pred_pitch, pred_roll, pred_yaw = output[:,0:66],output[:,66:66*2],output[:,66*2:66*3]
              
              label_pitch = batch.label[0][:, 0].as_in_context(ctx)
              label_roll = batch.label[0][:, 1].as_in_context(ctx)
              label_yaw = batch.label[0][:, 2].as_in_context(ctx)
        
              pitch_loss = calc_loss(pred_pitch, label_pitch)
              roll_loss = calc_loss(pred_roll, label_roll)
              yaw_loss = calc_loss(pred_yaw, label_yaw)
            
            # backpropagate
            autograd.backward([pitch_loss, roll_loss, yaw_loss])
            trainer.step(batch_size)
            
            if i % 100 == 0:
                print('epoch %d, batch: %d,Losses: Pitch %.4f, Roll %.4f, Yaw %.4f,time: %.4f '
                      % (
                          epoch, i, 
                          pitch_loss.mean().asscalar(),roll_loss.mean().asscalar(),yaw_loss.mean().asscalar(),
                          time.time() - tic))
            i += 1
            
        # Model save
        if epoch % 5 == 0:
          net.export(save_path+'/model_'+str(epoch))
