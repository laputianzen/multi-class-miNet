#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 01:32:17 2017

@author: PeterTsai
"""

from __future__ import division
from __future__ import print_function
import time
import os
from os.path import join as pjoin

import numpy as np
import tensorflow as tf
import pandas as pd 


import readmat
#from flags import FLAGS

class miNet(object):
    
    _weights_str = "weights{0}"
    _biases_str = "biases{0}"
    _inputs_str = "x{0}"
    _instNets_str = "I{0}"

    def __init__(self, shape, sess):
        """Autoencoder initializer

        Args:
            shape: list of ints specifying
                  num input, hidden1 units,...hidden_n units, num logits
        sess: tensorflow session object to use
        """
        self.__shape = shape  # [input_dim,hidden1_dim,...,hidden_n_dim,output_dim]
        self.__num_hidden_layers = len(self.__shape) - 2

        self.__variables = {}
        self.__sess = sess

        self._setup_instNet_variables()
    
    @property
    def shape(self):
        return self.__shape
    
    @property
    def num_hidden_layers(self):
        return self.__num_hidden_layers
    
    @property
    def session(self):
        return self.__sess
    
    def __getitem__(self, item):
        """Get autoencoder tf variable

        Returns the specified variable created by this object.
        Names are weights#, biases#, biases#_out, weights#_fixed,
        biases#_fixed.

        Args:
            item: string, variables internal name
        Returns:
            Tensorflow variable
        """
        return self.__variables[item]
    
    def __setitem__(self, key, value):
        """Store a tensorflow variable

        NOTE: Don't call this explicity. It should
        be used only internally when setting up
        variables.

        Args:
            key: string, name of variable
            value: tensorflow variable
        """
        self.__variables[key] = value
    
    def _setup_instNet_variables(self):
        with tf.name_scope("InstNet_variables"):
            for i in range(self.__num_hidden_layers + 1):
                # Train weights
                name_w = self._weights_str.format(i + 1)
                w_shape = (self.__shape[i], self.__shape[i + 1])
                #a = tf.multiply(4.0, tf.sqrt(6.0 / (w_shape[0] + w_shape[1])))
                #w_init = tf.random_uniform(w_shape, -1 * a, a)
                w_init = tf.random_normal(w_shape,stddev=0.01)
                self[name_w] = tf.Variable(w_init, name=name_w, trainable=True)
                # Train biases
                name_b = self._biases_str.format(i + 1)
                b_shape = (self.__shape[i + 1],)
                b_init = tf.zeros(b_shape) + 0.01
                self[name_b] = tf.Variable(b_init, trainable=True, name=name_b)

                if i < self.__num_hidden_layers:
                    # Hidden layer fixed weights (after pretraining before fine tuning)
                    self[name_w + "_fixed"] = tf.Variable(tf.identity(self[name_w]),
                        name=name_w + "_fixed", trainable=False)
                    
                    # Hidden layer fixed biases
                    self[name_b + "_fixed"] = tf.Variable(tf.identity(self[name_b]),
                        name=name_b + "_fixed", trainable=False)
                    
                    # Pretraining output training biases
                    name_b_out = self._biases_str.format(i + 1) + "_out"
                    b_shape = (self.__shape[i],)
                    b_init = tf.zeros(b_shape) + 0.01
                    self[name_b_out] = tf.Variable(b_init, trainable=True, name=name_b_out)
                    
    def _w(self, n, suffix=""):
        return self[self._weights_str.format(n) + suffix]
    
    def _b(self, n, suffix=""):
        return self[self._biases_str.format(n) + suffix]
    
    def get_variables_to_init(self, n):
        """Return variables that need initialization

        This method aides in the initialization of variables
        before training begins at step n. The returned
        list should be than used as the input to
        tf.initialize_variables

        Args:
            n: int giving step of training
        """
        assert n > 0
        assert n <= self.__num_hidden_layers + 1

        vars_to_init = [self._w(n), self._b(n)]

        if n <= self.__num_hidden_layers:
            vars_to_init.append(self._b(n, "_out"))
            
        if 1 < n <= self.__num_hidden_layers:
            vars_to_init.append(self._w(n - 1, "_fixed"))
            vars_to_init.append(self._b(n - 1, "_fixed"))
            
        return vars_to_init
    
    @staticmethod
    def _activate(x, w, b, transpose_w=False, acfun=None):
        if acfun is not None:
            y = acfun(tf.nn.bias_add(tf.matmul(x, w, transpose_b=transpose_w), b))
        else:
            y = tf.nn.bias_add(tf.matmul(x, w, transpose_b=transpose_w), b)
            
        return y
    
    def pretrain_net(self, input_pl, n, is_target=False):
        """Return net for step n training or target net
        
        Args:
            input_pl:  tensorflow placeholder of AE inputs
            n:         int specifying pretrain step
            is_target: bool specifying if required tensor
                       should be the target tensor
        Returns:
            Tensor giving pretraining net or pretraining target
        """
        assert n > 0
        assert n <= self.__num_hidden_layers

        last_output = input_pl
        for i in range(n - 1):
            if i == self.__num_hidden_layers+1:
                acfun = tf.sigmoid
            else:
                acfun = tf.nn.relu
            #acfun = "sigmoid"
            w = self._w(i + 1, "_fixed")
            b = self._b(i + 1, "_fixed")
            
            last_output = self._activate(last_output, w, b, acfun=acfun)         
            
        if is_target:
            return last_output
        
        if n == self.__num_hidden_layers+1:
            acfun = tf.sigmoid
        else:
            acfun = tf.nn.relu       
        last_output = self._activate(last_output, self._w(n), self._b(n), acfun=acfun)
        
        out = self._activate(last_output, self._w(n), self._b(n, "_out"),
                         transpose_w=True, acfun=acfun)
        out = tf.maximum(out, 1.e-9)
        out = tf.minimum(out, 1 - 1.e-9)
        return out
    
    def single_instNet(self, input_pl):
        """Get the supervised fine tuning net

        Args:
            input_pl: tf placeholder for ae input data
        Returns:
            Tensor giving full ae net
        """
        last_output = input_pl
        
        for i in range(self.__num_hidden_layers + 1):
            if i+1 == self.__num_hidden_layers+1:
                acfun = tf.sigmoid
            else:
                acfun = tf.nn.relu
            #acfun = "sigmoid" acfun = "relu"            
            # Fine tuning will be done on these variables
            w = self._w(i + 1)
            b = self._b(i + 1)
            
            last_output = self._activate(last_output, w, b, acfun=acfun)
            
        return last_output
    
    def MIL(self,input_plist):
        #input_dim = self.shape[0]
        tmpList = list()
        for i in range(input_plist.shape[0]):
            name_input = self._inputs_str.format(i + 1)
            #self[name_input] = tf.placeholder(tf.float32,[None, input_dim])
            self[name_input] = input_plist[i]
            
            name_instNet = self._instNets_str.format(i + 1)
            with tf.variable_scope("mil") as scope:
                if i == 0:
                #if scope.reuse == False:
                    self[name_instNet] = self.single_instNet(self[name_input])
                    #scope.reuse = True
                    scope.reuse_variables()
                else:    
                    self[name_instNet] = self.single_instNet(self[name_input])
            
            tmpList.append(self[name_instNet])
            #if not i == 0:
                #self["y"]  = tf.concat([self["y"],self[name_instNet]],1)
            #    self["y"]  = [self["y"],self[name_instNet]]
            #else:
            #    self["y"] = self[name_instNet]
        
        self["y"] = tf.stack(tmpList,axis=1)
        self["Y"] =  tf.reduce_max(self["y"],axis=1,name="MILPool")#,keep_dims=True)
        
        #batch_size = int(self["y"].shape[0])
        #topInstIdx = tf.reshape(tf.argmax(self["y"],axis=1),[batch_size,1])
        #self["kinst"] = tf.multiply(tf.round(self["Y"]),
        #    tf.cast(tf.argmax(self["y"],axis=1)+1,tf.float32),name='key_instance')
        
        #topInstIdx = tf.argmax(self["y"],axis=1)
        #self["kinst"] = tf.multiply(tf.round(self["Y"]),
        #    tf.cast(topInstIdx+1,tf.float32),name='key_instance')
        # consider tf.expand_dims to support tf.argmax
        
        #return self["Y"], self["kinst"]
        return self["Y"], self["y"]


loss_summaries = {}

def training(loss, learning_rate, loss_key=None, optimMethod=tf.train.AdamOptimizer):
  """Sets up the training Ops.

  Creates a summarizer to track the loss over time in TensorBoard.

  Creates an optimizer and applies the gradients to all trainable variables.

  The Op returned by this function is what must be passed to the
  `sess.run()` call to cause the model to train.

  Args:
    loss: Loss tensor, from loss().
    learning_rate: The learning rate to use for gradient descent.
    loss_key: int giving stage of pretraining so we can store
                loss summaries for each pretraining stage

  Returns:
    train_op: The Op for training.
  """
  if loss_key is not None:
    # Add a scalar summary for the snapshot loss.
    loss_summaries[loss_key] = tf.summary.scalar(loss.op.name, loss)
  else:
    tf.summary.scalar(loss.op.name, loss)
    for var in tf.trainable_variables():
      tf.summary.histogram(var.op.name, var)
  # Create the gradient descent optimizer with the given learning rate.
  #optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  #optimizer = tf.train.AdamOptimizer(learning_rate)
  optimizer = optimMethod(learning_rate)
  # Create a variable to track the global step.
  global_step = tf.Variable(0, name='global_step', trainable=False)
  # Use the optimizer to apply the gradients that minimize the loss
  # (and also increment the global step counter) as a single training step.
  train_op = optimizer.minimize(loss, global_step=global_step)
  return train_op, global_step

def loss_x_entropy(output, target):
    """Cross entropy loss
    
    See https://en.wikipedia.org/wiki/Cross_entropy

    Args:
        output: tensor of net output
        target: tensor of net we are trying to reconstruct
    Returns:
        Scalar tensor of cross entropy
        """
    with tf.name_scope("xentropy_loss"):
        net_output_tf = tf.convert_to_tensor(output, name='input')
        target_tf = tf.convert_to_tensor(target, name='target')
        cross_entropy = tf.add(tf.multiply(tf.log(net_output_tf, name='log_output'),
                                           target_tf),
                             tf.multiply(tf.log(1 - net_output_tf),
                                    (1 - target_tf)))
        return -1 * tf.reduce_mean(tf.reduce_sum(cross_entropy, 1),
                                   name='xentropy_mean')

_chkpt_dir = 'modelAe'
if not os.path.exists(_chkpt_dir):
    os.mkdir(_chkpt_dir)

def main_unsupervised(ae_shape,fold,FLAGS):
    tf.reset_default_graph()
    sess = tf.Session()
   
    aeList = list()
    for a in range(len(ae_shape)):
        aeList.append(miNet(ae_shape[a], sess))
    #aeC53 = miNet(ae_shape[0], sess)
    #aeC52 = miNet(ae_shape[1], sess)
    #aeC55 = miNet(ae_shape[2], sess)
    
    #aeList = [aeC53, aeC52, aeC55]
    
    writer = tf.summary.FileWriter(pjoin(FLAGS.summary_dir,
                                      'pre_training'),tf.get_default_graph())
    writer.close()  
    
    learning_rates = FLAGS.pre_layer_learning_rate
    
#for fold in range(5):
    print('fold %d' %(fold+1))
    for k in range(len(ae_shape)):
        tactic = FLAGS.tacticName[FLAGS.C5k_CLASS[k][0]]
        #datadir = 'dataGood/multiPlayers/syncLargeZoneVelocitySoftAssign(R=16,s=10)/train/fold%d/pretraining' %(fold+1)
        datadir = 'dataGood/multiPlayers/syncLargeZoneVelocitySoftAssign(R=16,s=10)/train/fold%d/' %(fold+1)
        fileName= '%sZoneVelocitySoftAssign(R=16,s=10)%d_training%d.mat' %(tactic,FLAGS.k[k],fold+1)
    
        batch_X, batch_Y, _ = readmat.read(datadir,fileName)
        num_train = len(batch_Y)
        strBagShape = "the shape of bags is ({0},{1})".format(batch_Y.shape[0],batch_Y.shape[1])
        print(strBagShape)   
        
        testdir = 'dataGood/multiPlayers/syncLargeZoneVelocitySoftAssign(R=16,s=10)/test/fold%d' %(fold+1)
        testFileName= '%sZoneVelocitySoftAssign(R=16,s=10)%d_test%d.mat' %(tactic,FLAGS.k[k],fold+1) 
        test_X, test_Y, test_label = readmat.read(testdir,testFileName)    
        strBagShape = "the shape of bags is ({0},{1})".format(test_Y.shape[0],test_Y.shape[1])
        print(strBagShape)  
        
        print("\nae_shape has %s pretarined layer" %(len(ae_shape[k])-2))
        for i in range(len(ae_shape[k]) - 2):
            n = i + 1
            _pretrain_model_dir = '{0}/{1}/C5{2}/pretrain{3}/'.format(_chkpt_dir,fold+1,FLAGS.k[k],n)
            if not os.path.exists(_pretrain_model_dir):
                os.makedirs(_pretrain_model_dir)
            
            with tf.variable_scope("pretrain_{0}_mi{1}".format(n,k+1)):
                input_ = tf.placeholder(dtype=tf.float32,
                                        shape=(None, ae_shape[k][0]),
                                        name='ae_input_pl')
                target_ = tf.placeholder(dtype=tf.float32,
                                         shape=(None, ae_shape[k][0]),
                                         name='ae_target_pl')
                layer = aeList[k].pretrain_net(input_, n)

                with tf.name_scope("target"):
                    target_for_loss = aeList[k].pretrain_net(target_, n, is_target=True)
                    
                if n == aeList[k].num_hidden_layers+1:
                    loss = loss_x_entropy(layer, target_for_loss)
                else:
                    loss = tf.sqrt(tf.nn.l2_loss(tf.subtract(layer, target_for_loss)))
                        
                train_op, global_step = training(loss, learning_rates[i], i, optimMethod=FLAGS.optim_method)
    
                summary_dir = pjoin(FLAGS.summary_dir, 'fold{0}/mi{1}/pretraining_{2}'.format(fold+1,k+1,n))
                summary_writer = tf.summary.FileWriter(summary_dir,
                                                        graph_def=sess.graph_def,
                                                        flush_secs=FLAGS.flush_secs)
                summary_vars = [aeList[k]["biases{0}".format(n)], aeList[k]["weights{0}".format(n)]]
    
                hist_summarries = [tf.summary.histogram(v.op.name, v)
                               for v in summary_vars]
                hist_summarries.append(loss_summaries[i])
                summary_op = tf.summary.merge(hist_summarries)

                vars_to_init = aeList[k].get_variables_to_init(n)
                vars_to_init.append(global_step)
                
                # adam special parameter beta1, beta2
                optim_vars = [var for var in tf.global_variables() if ('beta' in var.name or 'Adam' in var.name)]
#                for var in adam_vars:
#                    vars_to_init.append(var)
                            
                pretrain_test_loss  = tf.summary.scalar('pretrain_test_loss',loss)
                
                saver = tf.train.Saver(vars_to_init)
                model_ckpt = _pretrain_model_dir+ 'model.ckpt'    
            
                if os.path.isfile(model_ckpt+'.meta'):
                    #tf.reset_default_graph()
                    print("|---------------|---------------|---------|----------|")
                    saver.restore(sess, model_ckpt)
                    for v in vars_to_init:
                        print("%s with value %s" % (v.name, sess.run(tf.is_variable_initialized(v))))
#                text_file = open("Reload.txt", "a")
#                for b in range(len(ae_shape) - 2):
#                    if sess.run(tf.is_variable_initialized(ae._b(b+1))):
#                        #print("%s with value in [pretrain %s]\n %s" % (ae._b(b+1).name, n, ae._b(b+1).eval(sess)))
#                        text_file.write("%s with value in [pretrain %s]\n %s\n" % (ae._b(b+1).name, n, ae._b(b+1).eval(sess)))
#                text_file.close()                    
                else:
                    if FLAGS.pretrain_batch_size is None:
                        FLAGS.pretrain_batch_size = batch_X.shape[0]
                    sess.run(tf.variables_initializer(vars_to_init))
                    sess.run(tf.variables_initializer(optim_vars))
                    print("\n\n")
                    print("| Training Step | Cross Entropy |  Layer  |   Epoch  |")
                    print("|---------------|---------------|---------|----------|")
        
                    for epochs in range(FLAGS.pretraining_epochs):
                        perm = np.arange(num_train)
                        np.random.shuffle(perm)
                        for step in range(int(num_train/FLAGS.pretrain_batch_size)):
                            selectIndex = perm[FLAGS.pretrain_batch_size*step:FLAGS.pretrain_batch_size*step+FLAGS.pretrain_batch_size]
                            #for I in range(len(batch_X[0])):
                            #input_feed = batch_X[perm[2*step:2*step+2],i,:]
                            ##target_feed = batch_Y[2*step:2*step+2,1]
                            #target_feed = batch_X[perm[2*step:2*step+2],i,:]
                            #feed_dict = {input_: input_feed,target_: target_feed}
                            #feed_dict = fill_feed_dict_ae(data.train, input_, target_, noise[i])
                            input_feed = np.reshape(batch_X[selectIndex,:,:],
                                                    (batch_X[selectIndex,:,:].shape[0]*batch_X[selectIndex,:,:].shape[1],batch_X[selectIndex,:,:].shape[2]))
                            target_feed = input_feed
                            loss_summary, loss_value = sess.run([train_op, loss],
                                                            feed_dict={
                                                                input_: input_feed,
                                                                target_: target_feed
                                                                })

#                                count = epochs*num_train*len(batch_X[0])+step*len(batch_X[0])*len(input_feed)+(I+1)*len(input_feed)
#                            #if count % 100 == 0:
#                                if count % (10*len(input_feed)*len(batch_X[0])) == 0:
                            if step % 2 ==0:
                                summary_str = sess.run(summary_op, feed_dict={
                                                                input_: input_feed,
                                                                target_: target_feed
                                                                })
                                summary_writer.add_summary(summary_str, step)
                        #image_summary_op = \
                        #tf.image_summary("training_images",
                        #                 tf.reshape(input_,
                        #                            (FLAGS.batch_size,
                        #                             FLAGS.image_size,
                        #                             FLAGS.image_size, 1)),
                        #                 max_images=FLAGS.batch_size)
        
                        #summary_img_str = sess.run(image_summary_op,
                        #                       feed_dict=feed_dict)
                        #summary_writer.add_summary(summary_img_str)
        
                                output = "| {0:>13} | {1:13.4f} | Layer {2} | Epoch {3}  |"\
                                        .format(step, loss_value, n, epochs + 1)
    
                                print(output)
                        test_input_feed = np.reshape(test_X,(test_X.shape[0]*test_X.shape[1],test_X.shape[2]))
                        test_target_feed = np.reshape(test_X,(test_X.shape[0]*test_X.shape[1],test_X.shape[2]))
                        #test_target_feed = test_Y.astype(np.int32)     
                        loss_summary, loss_value = sess.run([train_op, loss],
                                                            feed_dict={
                                                                    input_: test_input_feed,
                                                                    target_: test_target_feed
                                                                    })
    
                        pretrain_test_loss_str = sess.run(pretrain_test_loss,
                                                  feed_dict={input_: test_input_feed,target_: test_target_feed
                                                     })                                          
                        summary_writer.add_summary(pretrain_test_loss_str, epochs)
                        print ('epoch %d: test loss = %.3f' %(epochs,loss_value))                           
                    summary_writer.close()         
#                text_file = open("Output.txt", "a")
#                for b in range(len(ae_shape) - 2):
#                    if sess.run(tf.is_variable_initialized(ae._b(b+1))):
#                        #print("%s with value in [pretrain %s]\n %s" % (ae._b(b+1).name, n, ae._b(b+1).eval(sess)))
#                        text_file.write("%s with value in [pretrain %s]\n %s\n" % (ae._b(b+1).name, n, ae._b(b+1).eval(sess)))
#                text_file.close()
                    save_path = saver.save(sess, model_ckpt)
                    print("Model saved in file: %s" % save_path)
                                      
                #input("\nPress ENTER to CONTINUE\n")  
    
        time.sleep(0.5)
                                      
    return aeList

def multiClassEvaluation(logits, labels):
    """Evaluate the quality of the logits at predicting the label.
    
    Args:
        logits: Logits tensor, float - [batch_size, NUM_CLASSES].
        labels: Labels tensor, int32 - [batch_size], with values in the
            range [0, NUM_CLASSES).

    Returns:
        A scalar int32 tensor with the number of examples (out of batch_size)
        that were predicted correctly.
        """    
    correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(labels,1))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))    

def calculateAccu(Y_pred,inst_pred,test_Y,test_label,FLAGS):
          
    
    KP_pred = np.zeros((len(Y_pred),5))
    for bagIdx in range(len(Y_pred)):
        for k in range(len(FLAGS.k)):
            if Y_pred[bagIdx] in FLAGS.C5k_CLASS[k]:
                c = FLAGS.C5k_CLASS[k].index(Y_pred[bagIdx])
                kinst = np.argmax(inst_pred[k][bagIdx,:,c])
                KP_pred[bagIdx] = FLAGS.playerMap[k][kinst]
                
    Y_correct = np.equal(Y_pred,np.argmax(test_Y,1))
    bagAccu = np.sum(Y_correct) / Y_pred.size
    
    y_correct = np.equal(KP_pred[Y_correct,:],test_label[Y_correct,:])
    
    pAccu = np.sum(y_correct) / KP_pred[Y_correct,:].size
    print('bag accuracy %.5f, inst accuracy %.5f' %(bagAccu, pAccu))
    time.sleep(1)
    return bagAccu, pAccu

text_file = open("final_result.txt", "w")

def main_supervised(instNetList,num_inst,fold,FLAGS):
    with instNetList[0].session.graph.as_default():
        sess = instNetList[0].session
        
#        text_file = open("FineTune.txt", "a")
#        for b in range(instNet.num_hidden_layers + 1):
#            if sess.run(tf.is_variable_initialized(instNet._b(b+1))):
#                #print("%s with value in [pretrain %s]\n %s" % (ae._b(b+1).name, n, ae._b(b+1).eval(sess)))
#                text_file.write("%s with value before fine-tuning\n %s\n" % (instNet._b(b+1).name, instNet._b(b+1).eval(sess)))
#        text_file.write("\n\n")
#        with tf.Session() as sess1:
#            saver = tf.train.Saver(tf.global_variables())
#            model_ckpt = '{0}/{1}/model_unsp.ckpt'.format(_chkpt_dir,fold+1)
#            save_path = saver.save(sess1, model_ckpt)
#            print("Model saved in file: %s" % save_path)       
#        
#        new = True
#
#        if not os.path.exists('{0}/{1}'.format(_chkpt_dir,fold+1)):
#            os.mkdir('{0}/{1}'.format(_chkpt_dir,fold+1))        
#            
#        model_ckpt = '{0}/{1}/model_sp.ckpt'.format(_chkpt_dir,fold+1)
#    
#        if os.path.isfile(model_ckpt+'.meta'):
#            input_var = None
#            while input_var not in ['yes', 'no']:
#                input_var = input(">>> We found model.ckpt file. Do you want to load it [yes/no]?")
#            if input_var == 'yes':
#                new = False
        bagOuts = []
        instOuts = []
        totalNumInst = np.sum(num_inst)
        instIdx = np.insert(np.cumsum(num_inst),0,0)
        input_pl = tf.placeholder(tf.float32, shape=(totalNumInst,None,
                                                instNetList[0].shape[0]),name='input_pl')

        hist_summaries = []
        for k in range(len(instNetList)):            
            out_Y, out_y = instNetList[k].MIL(input_pl[instIdx[k]:instIdx[k+1]])
            #bagOuts.append(tf.transpose(out_Y,perm=[1,0]))
            bagOuts.append(out_Y)
            instOuts.append(out_y)
            
            hist_summaries.extend([instNetList[k]['biases{0}'.format(i + 1)]
                              for i in range(instNetList[k].num_hidden_layers + 1)])
            hist_summaries.extend([instNetList[k]['weights{0}'.format(i + 1)]
                                   for i in range(instNetList[k].num_hidden_layers + 1)])
    
        hist_summaries = [tf.summary.histogram(v.op.name + "_fine_tuning", v)
                              for v in hist_summaries]
        summary_op = tf.summary.merge(hist_summaries)            
        
        #Y = tf.dynamic_stitch(FLAGS.C5k_CLASS,bagOuts)
        Y = tf.concat(bagOuts,1)
#        saver = tf.train.Saver()
#        if new:
        print("")
        print('fold %d' %(fold+1))
        datadir = 'dataGood/multiPlayers/syncLargeZoneVelocitySoftAssign(R=16,s=10)/train/fold%d' %(fold+1)
        file_str= '{0}ZoneVelocitySoftAssign(R=16,s=10){1}_training%d.mat' %(fold+1)

        batch_multi_X, batch_multi_Y, batch_multi_KPlabel = readmat.multi_class_read(datadir,file_str,num_inst,FLAGS)
        num_train = len(batch_multi_Y)
        strBagShape = "the shape of bags is ({0},{1})".format(batch_multi_Y.shape[0],batch_multi_Y.shape[1])
        print(strBagShape)       

        testdir = 'dataGood/multiPlayers/syncLargeZoneVelocitySoftAssign(R=16,s=10)/test/fold%d' %(fold+1)
        test_file_str= '{0}ZoneVelocitySoftAssign(R=16,s=10){1}_test%d.mat' %(fold+1) 
        test_multi_X, test_multi_Y, test_multi_label = readmat.multi_class_read(testdir,test_file_str,num_inst,FLAGS)       
        strBagShape = "the shape of bags is ({0},{1})".format(test_multi_Y.shape[0],test_multi_Y.shape[1])
        print(strBagShape)  
      
        if FLAGS.finetune_batch_size is None:
            FLAGS.finetune_batch_size = len(test_multi_Y)
            
        NUM_CLASS = len(FLAGS.tacticName)
        Y_placeholder = tf.placeholder(tf.int32,
                                        shape=(None,NUM_CLASS),
                                        name='target_pl')
        
        #loss = loss_x_entropy(tf.nn.softmax(Y), tf.cast(Y_placeholder, tf.float32))
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Y,
                                labels=tf.cast(Y_placeholder, tf.float32),name='softmax_cross_entropy'))
        loss_op = tf.summary.scalar('loss',loss)
        #loss = loss_supervised(logits, labels_placeholder)
        train_op, global_step = training(loss, FLAGS.supervised_learning_rate, None, optimMethod=FLAGS.optim_method)
        accu = multiClassEvaluation(Y, Y_placeholder)
        correct =tf.equal(tf.argmax(Y,1),tf.argmax(Y_placeholder,1))
        error = 1 - tf.reduce_mean(tf.cast(correct, tf.float32))
        error_op = tf.summary.scalar('error',error)
        accu_op = tf.summary.scalar('accuracy',accu)

        output_op = tf.summary.histogram('tactic_logits',Y)
        label_op = tf.summary.histogram('tactic_labels',Y_placeholder)
        merged = tf.summary.merge([loss_op,error_op,accu_op,summary_op,output_op,label_op])
        summary_writer = tf.summary.FileWriter(pjoin(FLAGS.summary_dir,
                                                      'fold{0}/fine_tuning'.format(fold+1)),tf.get_default_graph())
                                                #graph_def=sess.graph_def,
                                                #flush_secs=FLAGS.flush_secs)
        vars_to_init = []
        for k in range(len(instNetList)):
            instNet = instNetList[k]
            vars_to_init.extend(instNet.get_variables_to_init(instNet.num_hidden_layers + 1))
        
        vars_to_init.append(global_step)
        # adam special parameter beta1, beta2
        optim_vars = [var for var in tf.global_variables() if ('beta' in var.name or 'Adam' in var.name)] 
        
        sess.run(tf.variables_initializer(vars_to_init))
        sess.run(tf.variables_initializer(optim_vars))
    
        train_loss  = tf.summary.scalar('train_loss',loss)
        #steps = FLAGS.finetuning_epochs * num_train
        for epochs in range(FLAGS.finetuning_epochs_epochs):
            perm = np.arange(num_train)
            np.random.shuffle(perm)
            print("|-------------|-----------|-------------|----------|")
            text_file.write("|-------------|-----------|-------------|----------|")
            for step in range(int(num_train/FLAGS.finetune_batch_size)):
                start_time = time.time()
            
                selectIndex = perm[FLAGS.finetune_batch_size*step:FLAGS.finetune_batch_size*step+FLAGS.finetune_batch_size]
                input_feed = batch_multi_X[selectIndex,:,:]
                input_feed = np.transpose(input_feed, (1,0,2))
                target_feed = batch_multi_Y[selectIndex].astype(np.int32)         
            
                _, loss_value, logit, label = sess.run([train_op, loss, Y, Y_placeholder],
                                        feed_dict={
                                                input_pl: input_feed,
                                                Y_placeholder: target_feed
                                        })
            
                duration = time.time() - start_time
                
                #count = epochs*(num_train/FLAGS.batch_size)+step
                # Write the summaries and print an overview fairly often.
                #if step % 10 == 0:
                if step % 10 == 0:
                    # Print status to stdout.
                    #print('Step %d: loss = %.2f (%.3f sec)' % (count, loss_value, duration))
                    print('|   Epoch %d  |  Step %d  |  loss = %.3f | (%.3f sec)' % (epochs+1, step, loss_value, duration))
                    text_file.write('|   Epoch %d  |  Step %d  |  loss = %.3f | (%.3f sec)\n' % (epochs+1, step, loss_value, duration))
                    
                # Update the events file.
                input_feed = np.transpose(batch_multi_X, (1,0,2))
                target_feed = batch_multi_Y.astype(np.int32) 
                train_loss_str = sess.run(train_loss,
                                          feed_dict={input_pl: input_feed,Y_placeholder: target_feed
                                                     })                                          
                summary_writer.add_summary(train_loss_str, epochs)                    

            test_input_feed = np.transpose(test_multi_X, (1,0,2))
            test_target_feed = test_multi_Y.astype(np.int32) 
            bagAccu, Y_pred, inst_pred = sess.run([accu, tf.argmax(tf.nn.softmax(Y),axis=1), instOuts],
                                                   feed_dict={input_pl: test_input_feed,Y_placeholder: test_target_feed})
            print('Epochs %d: accuracy = %.5f '  % (epochs+1, bagAccu)) 
            text_file.write('Epochs %d: accuracy = %.5f\n\n'  % (epochs+1, bagAccu))
            
            result = sess.run(merged,feed_dict={input_pl: test_input_feed,Y_placeholder: test_target_feed})
            i = epochs * num_train/FLAGS.finetune_batch_size +step
            summary_writer.add_summary(result,i)
            #baseline = 1-len(test_Y.nonzero())/len(test_Y)
            #if bagAccu > baseline:
            calculateAccu(Y_pred,inst_pred,test_multi_Y,test_multi_label,FLAGS)
            
            filename = _confusion_dir + '/Fold{0}_Epoch{1}_test.csv'.format(fold,epochs)
            ConfusionMatrix(Y_pred,test_multi_Y,FLAGS,filename)
            #print("")

                    #summary_str = sess.run(summary_op, feed_dict=feed_dict)
                    #summary_writer.add_summary(summary_str, step)
                    #summary_img_str = sess.run(
                    #    tf.image_summary("training_images",
                    #                tf.reshape(input_pl,
                    #                        (FLAGS.batch_size,
                    #                         FLAGS.image_size,
                    #                         FLAGS.image_size, 1)),
                    #             max_images=FLAGS.batch_size),
                    #    feed_dict=feed_dict
                    #)
                    #summary_writer.add_summary(summary_img_str)
                    
#            for b in range(instNet.num_hidden_layers + 1):
#                if sess.run(tf.is_variable_initialized(instNet._b(b+1))):
#                    #print("%s with value in [pretrain %s]\n %s" % (ae._b(b+1).name, n, ae._b(b+1).eval(sess)))
#                    text_file.write("%s with value after fine-tuning\n %s\n" % (instNet._b(b+1).name, instNet._b(b+1).eval(sess)))
#            text_file.close()
#            save_path = saver.save(sess, model_ckpt)
#            print("Model saved in file: %s" % save_path)    
#        else:
#            saver = tf.train.import_meta_graph(model_ckpt+'.meta')
#            saver.restore(sess, model_ckpt)                    
           
        input_feed = np.transpose(test_multi_X, (1,0,2))
        target_feed = test_multi_Y.astype(np.int32) 
        bagAccu, Y_pred, inst_pred = sess.run([accu, tf.argmax(tf.nn.softmax(Y),axis=1), instOuts],
                                               feed_dict={input_pl: test_input_feed,Y_placeholder: test_target_feed})
        
        print('\nAfter %d Epochs: accuracy = %.5f'  % (epochs+1, bagAccu))
        calculateAccu(Y_pred,inst_pred,test_multi_Y,test_multi_label,FLAGS)
        time.sleep(0.5)
        filename = _confusion_dir + '/Fold{0}_Epoch{1}_test_final.csv'.format(fold,epochs)
        ConfusionMatrix(Y_pred,test_multi_Y,FLAGS,filename)        
 
        summary_writer.close()           

def ConfusionMatrix(logits,labels,FLAGS,filename):
    C = np.zeros((len(FLAGS.tacticName),len(FLAGS.tacticName)))
    CM = C    
    flattenC5k = [val for sublist in FLAGS.C5k_CLASS for val in sublist]
    for bagIdx in range(len(labels)):
        gt = np.argmax(labels[bagIdx])
        #pred = np.argmax(logits[bagIdx])
        pred = logits[bagIdx]
        new_gt = flattenC5k[gt]
        new_pred= flattenC5k[pred]
        C[new_gt,new_pred] = C[new_gt,new_pred] + 1
        
    print(C)
    cumC = np.sum(C,axis=1)
    
    for p in range(len(C)):
        CM[p,:] = np.divide(C[p,:],cumC[p])
    
    df = pd.DataFrame(CM)
    df.round(3)
    df.to_csv(filename)    
             
_confusion_dir = 'confusionMatrix'
if not os.path.exists(_confusion_dir):
    os.mkdir(_confusion_dir)   