#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 02:58:03 2017

@author: PeterTsai
"""

import os
import shutil
import tensorflow as tf
import numpy as np

import miNet


def main(optimizer,scale,fld=1):
    FLAGS = lambda: None
    FLAGS.pretrain_batch_size = 2
    FLAGS.finetune_batch_size = None
    FLAGS.finetuning_epochs_epochs = 200
    
    #optimizer = input('Select Optimizer [0:GradientDescent, 1:Adam]:')
    if optimizer == '0':
        FLAGS.pretraining_epochs = 2000
        FLAGS.pre_layer_learning_rate = (0.01*scale,0.01*scale)#GD[0.01,0.01]
        FLAGS.supervised_learning_rate = 0.5*scale#GD 0.5
        FLAGS.optim_method = tf.train.GradientDescentOptimizer
        FLAGS.exp_dir = 'experiment/GradientDescent/pretrain={0}_finetune={1}'.format(
                FLAGS.pre_layer_learning_rate,FLAGS.supervised_learning_rate)
    elif optimizer == '1':
        FLAGS.pretraining_epochs = 600
        FLAGS.supervised_learning_rate = 0.001*scale#GD 0.5
        FLAGS.pre_layer_learning_rate = (0.001*scale,0.001*scale)#GD[0.01,0.01]
        FLAGS.optim_method = tf.train.AdamOptimizer
        FLAGS.exp_dir = 'experiment/Adam/pretrain={0}_finetune={1}'.format(
                FLAGS.pre_layer_learning_rate,FLAGS.supervised_learning_rate)
    
    FLAGS.flush_secs  = 120
    FLAGS.summary_dir = FLAGS.exp_dir + '/summaries'
    FLAGS._ckpt_dir   = FLAGS.exp_dir + '/modelAe'
    FLAGS._confusion_dir = FLAGS.exp_dir + '/confusionMatrix'
    FLAGS._result_txt = FLAGS.exp_dir + '/final_result.txt'
    
    FLAGS.tacticName =['F23','EV','HK','PD','PT','RB','SP','WS','WV','WW']
    #tacticNumKP=[3,3,3,3,5,3,2,3,5,2]
    #NUM_CLASS = len(tacticName)
    #C5k k=3,2,5
    FLAGS.C5k_CLASS = [[0,1,2,3,5,7],[6,9],[4,8]]
    FLAGS.k = [3,2,5]
    FLAGS.playerMap = [[[1,1,1,0,0],[1,1,0,1,0],[1,1,0,0,1],[1,0,1,1,0],[1,0,1,0,1],
                           [1,0,0,1,1],[0,1,1,1,0],[0,1,1,0,1],[0,1,0,1,1],[0,0,1,1,1]],
                       [[1,1,0,0,0],[1,0,1,0,0],[1,0,0,1,0],[1,0,0,0,1],[0,1,1,0,0],
                           [0,1,0,1,0],[0,1,0,0,1],[0,0,1,1,0],[0,0,1,0,1],[0,0,0,1,1]],
                       [[1,1,1,1,1]]];
                        
    
    #instNet_shape = [1040,130,10,1] #[1040,10,1]
    instNet_shape = np.array([[1040,535,10,len(FLAGS.C5k_CLASS[0])],
                              [1040,535,10,len(FLAGS.C5k_CLASS[1])],
                              [1040,535,10,len(FLAGS.C5k_CLASS[2])]],
                             np.int32)    
    num_inst = np.array([10,10,1],np.int32) # 5 choose 3 key players, 5 choose 2 key players, 5 choose 3 key players
    if fld is None:
        fold_str = input('Select Fold to run (0~5)[0:all fold]:') # 0 stay for all fold
        if fold_str == '0':
            queue = range(5)
        else:
            queue = [int(fold_str) - 1]
    else:
        queue = [fld-1]
    
    for fold in queue:
        miList = miNet.main_unsupervised(instNet_shape,fold,FLAGS)
        miNet.main_supervised(miList,num_inst,fold,FLAGS)

if __name__ == '__main__':
    optimizer = ['1','0']
    step     = [-2,-1,0,1,2]#[-4,-3,-2,-1,0,1,2,3,4]
    for m in optimizer:
        for s in step:
            scale = 2**s
            main(m,scale)
