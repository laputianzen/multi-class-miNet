#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 02:58:03 2017

@author: PeterTsai
"""

import os
import shutil
import tensorflow as tf

import miNet

#FLAGS = {}

def main():
    instNet_shape = [1040,130,10,1] #[1040,10,1]
    FLAGS = lambda: None
    FLAGS.pretrain_batch_size = 2;
    FLAGS.finetune_batch_size = 2;
    FLAGS.pre_layer_learning_rate = [0.001,0.001]
    FLAGS.pretraining_epochs = 2000
    FLAGS.supervised_learning_rate = 0.5
    FLAGS.finetuning_epochs_epochs = 20
    
    num_inst = 10 # 5 choose 3 key players
    fold     =  0
    mi = miNet.main_unsupervised(instNet_shape,fold,FLAGS)
    miNet.main_supervised(mi,num_inst,fold,FLAGS)

if __name__ == '__main__':
    main()
