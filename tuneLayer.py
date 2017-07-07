0#!/usr/bin/env python3
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

def createPretrainShape(num_input,num_output,num_hidden_layer):
    reduce_dim = round((num_input-num_output)/(num_hidden_layer+1))
    shape = np.zeros(num_hidden_layer+2,dtype=np.int32)
    shape[0] = num_input
    shape[num_hidden_layer+1] = num_output
    for l in range(num_hidden_layer):
        shape[l+1] = shape[l] - reduce_dim
        
    return shape

def main(optimizer,num_hidden_layer,fld=1):
    FLAGS = lambda: None
    FLAGS.pretrain_batch_size = 2
    FLAGS.finetune_batch_size = None
    FLAGS.finetuning_epochs_epochs = 200
   
    if num_hidden_layer is None:
        num_hidden_layer = input('how many hidden layer?')
        num_hidden_layer = int(num_hidden_layer)
        if num_hidden_layer < 1:
            print("number of hidden layer can't be less than 1")
            return
    else:
        num_hidden_layer = int(num_hidden_layer)
    
    num_input = 1040
    num_output= 10
    pretrain_shape = createPretrainShape(num_input,num_output,num_hidden_layer)
    print(pretrain_shape)
    #optimizer = input('Select Optimizer [0:GradientDescent, 1:Adam]:')
    FLAGS.pre_layer_learning_rate = []
    if optimizer == '0':
        FLAGS.pretraining_epochs = 2000
        
        for h in range(num_hidden_layer+1):
            FLAGS.pre_layer_learning_rate.extend([0.01])#GD[0.01,0.01]
        FLAGS.supervised_learning_rate = 0.5#GD 0.5
        FLAGS.optim_method = tf.train.GradientDescentOptimizer
        FLAGS.exp_dir = 'experiment/GradientDescent/numHiddenLayer{0}'.format(num_hidden_layer)
    elif optimizer == '1':
        FLAGS.pretraining_epochs = 600
        FLAGS.supervised_learning_rate = 0.001#GD 0.5
        for h in range(num_hidden_layer+1):
            FLAGS.pre_layer_learning_rate.extend([0.001])#GD[0.01,0.01]
        FLAGS.optim_method = tf.train.AdamOptimizer
        FLAGS.exp_dir = 'experiment//Adam/numHiddenLayer{0}'.format(num_hidden_layer)
    
    FLAGS.flush_secs  = 120
    FLAGS.summary_dir = FLAGS.exp_dir + '/summaries'
    FLAGS._ckpt_dir   = FLAGS.exp_dir + '/model'
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
    instNet_shape = np.array([np.append(pretrain_shape,len(FLAGS.C5k_CLASS[0])),
                              np.append(pretrain_shape,len(FLAGS.C5k_CLASS[1])),
                              np.append(pretrain_shape,len(FLAGS.C5k_CLASS[2]))],
                             np.int32)
    print(instNet_shape)    
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
    optimizer = ['1']
    for numHL in range(6):
        main(optimizer[0],numHL+1)
