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

#FLAGS = {}

def main():
    FLAGS = lambda: None
    FLAGS.pretrain_batch_size = 2;
    FLAGS.finetune_batch_size = 2;
    FLAGS.pre_layer_learning_rate = [0.001,0.001]
    FLAGS.pretraining_epochs = 2000
    FLAGS.supervised_learning_rate = 0.5
    FLAGS.finetuning_epochs_epochs = 20


    FLAGS.tacticName =['F23','EV','HK','PD','PT','RB','SP','WS','WV','WW']
    #tacticNumKP=[3,3,3,3,5,3,2,3,5,2]
    #NUM_CLASS = len(tacticName)
    FLAGS.C53_CLASS = [0,1,2,3,5,7]
    FLAGS.C52_CLASS = [6,9]
    FLAGS.C55_CLASS = [4,8]
    
    #instNet_shape = [1040,130,10,1] #[1040,10,1]
    instNet_shape = np.array([[1040,130,10,len(FLAGS.C53_CLASS)],
                              [1040,130,10,len(FLAGS.C52_CLASS)],
                              [1040,130,10,len(FLAGS.C55_CLASS)]],
                             np.int32)    
    num_inst = np.array([10,10,1],np.int32) # 5 choose 3 key players, 5 choose 2 key players, 5 choose 3 key players 
    fold     =  0
    mi = miNet.main_unsupervised(instNet_shape,fold,FLAGS)
    miNet.main_supervised(mi,num_inst,fold,FLAGS)

if __name__ == '__main__':
    main()
