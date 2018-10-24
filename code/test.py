# -*- coding: utf-8 -*-

'''
Model testing:
  Predict stablity of model with your own input data. Default mod = 1
    mod = 0 : input single image to predict
    mod = 1 : input single image folder to predict, result will be saved under /test_results
    mod = 2 : test on testing set
'''

from mxnet import gluon
from mxnet.gluon import nn
from mxnet import nd
from collections import namedtuple
import random
import mxnet as mx
import cv2
import argparse
import numpy as np
from PIL import Image
import os
import utils
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Predicting.')
    parser.add_argument('--mode', dest='mode', help='Input mode',
                        default=1, type=int)
    parser.add_argument('--image_size', dest='image_size', help='size of image',
                        default=48, type=int)
    parser.add_argument('--input', dest='image_path', help='Images path',
                        default="/test_images/", type=str)
    parser.add_argument('--save', dest='out_path', help='Path to save plot image',
                        default="/test_images/results/", type=str)
    parser.add_argument('--batch_size', dest='batch_size', help='Batch size',
                        default=1, type=int)
    parser.add_argument('--model_json', dest='json', help='Model structure path',
                        default='/model/model_15-symbol.json', type=str)
    parser.add_argument('--model_params', dest='params', help='Model params path',
                        default='/model/model_15-0000.params', type=str)
    parser.add_argument('--rec', dest='rec_path', help='Path to rec file',
                        default="/test_48x48.rec", type=str)
    parser.add_argument('--lst', dest='lst_path', help='Path to idx file',
                        default="/test_48x48.lst", type=str)
    parser.add_argument('--dev', dest='dev', help='default is cpu,if you need to use gpu,dev should be gpu device number',
                        default=-1, type=int)
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
    # Mean average loss
    softmax_pred = stable_softmax(pred) + NEAR_0
    expectation = idx_tensor * softmax_pred
    expectation = nd.sum(expectation, 1) * 3 - 102
    loss = nd.abs(expectation - label)
    loss = nd.sum(loss) / len(loss)
    return loss

if __name__ == '__main__':
    args = parse_args()
    mode = args.mode
    test_rec = args.rec_path
    test_lst = args.lst_path
    in_dir = args.image_path
    out_dir = args.out_path
    batch_size = args.batch_size
    image_size = args.image_size
    model_structure = args.json
    model_params = args.params
    dev = args.dev
    
    if dev == -1:
      ctx = mx.cpu()
    else:
      ctx = mx.gpu(dev)
      
    NEAR_0 = 1e-10
    idx_tensor = nd.array([idx for idx in range(1,67)]).as_in_context(ctx)
    
    resize_w, resize_h = 48, 48
    channel = 1
      
    symnet = mx.symbol.load(model_structure)
    mod = mx.mod.Module(symbol=symnet, context=ctx)
    mod.bind(data_shapes=[('data', (batch_size, 1, 48, 48))])
    mod.load_params(model_params)
    Batch = namedtuple('Batch', ['data'])

    if mode == 0:
        # read a single image and reshape to [batch_size,channel,width,height]
        img = cv2.imread(in_dir, 0)
        img = cv2.resize(img, (image_size, image_size))
        img = np.array([img,img,img])
        img = img[np.newaxis]
        
        # net forward
        mod.forward(Batch([mx.nd.array(img)]),is_train=False)

        # get resules
        pred = mod.get_outputs()
        pred_pitch,pred_roll,pred_yaw = pred[0][0,0:66].reshape([1,66]),pred[0][0,66:66*2].reshape([1,66]),pred[0][0,66*2:66*3].reshape([1,66])
        
        # expectation
        softmax_pred_pitch = stable_softmax(pred_pitch) + NEAR_0
        expectation_pitch = idx_tensor * softmax_pred_pitch
        expectation_pitch = nd.sum(expectation_pitch, 1) * 3 - 99

        softmax_pred_roll = stable_softmax(pred_roll) + NEAR_0
        expectation_roll = idx_tensor * softmax_pred_roll
        expectation_roll = nd.sum(expectation_roll, 1) * 3 - 99

        softmax_pred_yaw = stable_softmax(pred_yaw) + NEAR_0
        expectation_yaw = idx_tensor * softmax_pred_yaw
        expectation_yaw = nd.sum(expectation_yaw, 1) * 3 - 99
        
        show_img = cv2.imread(in_dir, 1)
        show_img = cv2.resize(show_img,(224,224))
            
        cv2.putText(show_img, "p_pred:" + str(round(expectation_pitch.asnumpy()[0], 3)), (10, 180), 1, 0.8, (0, 0, 255),1)
        cv2.putText(show_img, "r_pred:" + str(round(expectation_roll.asnumpy()[0], 3)), (10, 190), 1, 0.8, (0, 0, 255),1)
        cv2.putText(show_img, "y_pred:" + str(round(expectation_yaw.asnumpy()[0], 3)), (10, 200), 1, 0.8, (0, 0, 255),1)
        utils.draw_axis(show_img, expectation_yaw.asnumpy()[0], expectation_pitch.asnumpy()[0], expectation_roll.asnumpy()[0], None, None, 50)
        
        cv2.imwrite(out_dir, show_img)
    elif mode == 1:
        # read images
        image_names = os.listdir(in_dir)
        for image_name in image_names:
            image_file = in_dir + '/' + image_name
            img = cv2.imread(image_file, 0)
            
            if img is None:
                print('Fail to open image:', image_file)
                continue
            
            img = cv2.resize(img, (image_size, image_size))
            img = img[np.newaxis]
            img = img[np.newaxis]

            # net forward
            mod.forward(Batch([mx.nd.array(img)]),is_train=False)

            # get result
            pred = mod.get_outputs()
            pred_pitch,pred_roll,pred_yaw = pred[0][0,0:66].reshape([1,66]),pred[0][0,66:66*2].reshape([1,66]),pred[0][0,66*2:66*3].reshape([1,66])
            
            # expectation
            softmax_pred_pitch = stable_softmax(pred_pitch) + NEAR_0
            expectation_pitch = idx_tensor * softmax_pred_pitch
            expectation_pitch = nd.sum(expectation_pitch, 1) * 3 - 99

            softmax_pred_roll = stable_softmax(pred_roll) + NEAR_0
            expectation_roll = idx_tensor * softmax_pred_roll
            expectation_roll = nd.sum(expectation_roll, 1) * 3 - 99

            softmax_pred_yaw = stable_softmax(pred_yaw) + NEAR_0
            expectation_yaw = idx_tensor * softmax_pred_yaw
            expectation_yaw = nd.sum(expectation_yaw, 1) * 3 - 99
            
            show_img = cv2.imread(image_file, 1)
            show_img = cv2.resize(show_img,(224,224))
            
            cv2.putText(show_img, "p_pred:" + str(round(expectation_pitch.asnumpy()[0], 3)), (10, 180), 1, 0.8, (0, 0, 255),1)
            cv2.putText(show_img, "r_pred:" + str(round(expectation_roll.asnumpy()[0], 3)), (10, 190), 1, 0.8, (0, 0, 255),1)
            cv2.putText(show_img, "y_pred:" + str(round(expectation_yaw.asnumpy()[0], 3)), (10, 200), 1, 0.8, (0, 0, 255),1)
            utils.draw_axis(show_img, expectation_yaw.asnumpy()[0], expectation_pitch.asnumpy()[0], expectation_roll.asnumpy()[0], None, None, 50)

            cv2.imwrite(out_dir + image_name , show_img)
    elif mode == 2:
        test_data = mx.io.ImageRecordIter(
            path_imgrec=test_rec,
            path_imglist=test_lst,
            label_width=3,
            label_name=['pitch_label', 'roll_label', 'yaw_label'],
            data_shape=(channel, resize_h, resize_w),
            batch_size=batch_size,
            shuffle=False)
      
        index = 0
        total_pitch = 0
        total_roll = 0
        total_yaw = 0
        
        test_data.reset()
        for i, batch in enumerate(test_data):

            data = batch.data[0].as_in_context(ctx)
            mod.forward(Batch([data]),is_train=False)
            pred = mod.get_outputs()
            pred_pitch,pred_roll,pred_yaw = pred[0][:,0:66].reshape([batch_size,66]),pred[0][:,66:66*2].reshape([batch_size,66]),pred[0][:,66*2:66*3].reshape([batch_size,66])
                        
            label_pitch = batch.label[0][:, 0].as_in_context(ctx)
            label_roll = batch.label[0][:, 1].as_in_context(ctx)
            label_yaw = batch.label[0][:, 2].as_in_context(ctx)
            
            pitch_loss = calc_loss(pred_pitch, label_pitch)
            roll_loss = calc_loss(pred_roll, label_roll)
            yaw_loss = calc_loss(pred_yaw, label_yaw)
            
            total_pitch += pitch_loss
            total_roll += roll_loss
            total_yaw += yaw_loss

            index += 1
        print('Mean Average Losses of testing : Pitch %.4f, Roll %.4f, Yaw %.4f'
                  % (total_pitch.mean().asscalar() / index, total_roll.mean().asscalar() / index,
                     total_yaw.mean().asscalar() / index))
                     