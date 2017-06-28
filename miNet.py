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
    def _activate(x, w, b, transpose_w=False, acfun=""):
        if acfun is "sigmoid":
            y = tf.sigmoid(tf.nn.bias_add(tf.matmul(x, w, transpose_b=transpose_w), b))
        elif acfun is "relu":
            y = tf.nn.relu(tf.nn.bias_add(tf.matmul(x, w, transpose_b=transpose_w), b))
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
                acfun = "sigmoid"
            else:
                acfun = "relu"
            #acfun = "sigmoid"
            w = self._w(i + 1, "_fixed")
            b = self._b(i + 1, "_fixed")
            
            last_output = self._activate(last_output, w, b, acfun=acfun)         
            
        if is_target:
            return last_output
        
        if n == self.__num_hidden_layers+1:
            acfun = "sigmoid"
        else:
            acfun = "relu"        
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
                acfun = "sigmoid"
            else:
                acfun = "relu"
            #acfun = "sigmoid"            
            # Fine tuning will be done on these variables
            w = self._w(i + 1)
            b = self._b(i + 1)
            
            last_output = self._activate(last_output, w, b, acfun=acfun)
            
        return last_output
    
    def MIL(self,input_plist):
        #input_dim = self.shape[0] 
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
            
            if not i == 0:
                self["y"]  = tf.concat([self["y"],self[name_instNet]],1)
            else:
                self["y"] = self[name_instNet]
            
        self["Y"] =  tf.reduce_max(self["y"],axis=1,name="MILPool",keep_dims=True)
        topInstIdx = tf.reshape(tf.argmax(self["y"],axis=1),[-1,1])
        #batch_size = int(self["y"].shape[0])
        #topInstIdx = tf.reshape(tf.argmax(self["y"],axis=1),[batch_size,1])
        #self["kinst"] = tf.multiply(tf.round(self["Y"]),
        #    tf.cast(tf.argmax(self["y"],axis=1)+1,tf.float32),name='key_instance')
        self["kinst"] = tf.multiply(tf.round(self["Y"]),
            tf.cast(topInstIdx+1,tf.float32),name='key_instance')
        
        return self["Y"], self["kinst"]


loss_summaries = {}

def training(loss, learning_rate, loss_key=None):
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
  optimizer = tf.train.GradientDescentOptimizer(learning_rate)
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
    
    writer = tf.summary.FileWriter('logs',tf.get_default_graph())
    writer.close()
    
    learning_rates = FLAGS.pre_layer_learning_rate
    
#for fold in range(5):
    print('fold %d' %(fold+1))
    datadir = 'dataGood/multiPlayers/syncLargeZoneVelocitySoftAssign(R=16,s=10)/train/fold%d' %(fold+1)
    fileName= 'EVZoneVelocitySoftAssign(R=16,s=10)3_training%d.mat' %(fold+1)

    batch_X, batch_Y, _ = readmat.read(datadir,fileName)
    num_train = len(batch_Y)

    
    testdir = 'dataGood/multiPlayers/syncLargeZoneVelocitySoftAssign(R=16,s=10)/test/fold%d' %(fold+1)
    testFileName= 'EVZoneVelocitySoftAssign(R=16,s=10)3_test%d.mat' %(fold+1) 
    test_X, test_Y, test_label = readmat.read(testdir,testFileName)    

    print("\nae_shape has %s pretarined layer" %(len(ae_shape)-2))
    for i in range(len(ae_shape) - 2):
        n = i + 1
        _pretrain_model_dir = '{0}/{1}/pretrain{2}'.format(_chkpt_dir,fold+1,n)
        if not os.path.exists(_pretrain_model_dir):
            os.makedirs(_pretrain_model_dir)
            
        with tf.variable_scope("pretrain_{0}".format(n)):
            input_ = tf.placeholder(dtype=tf.float32,
                                shape=(FLAGS.pretrain_batch_size, ae_shape[0]),
                                name='ae_input_pl')
            target_ = tf.placeholder(dtype=tf.float32,
                                 shape=(FLAGS.pretrain_batch_size, ae_shape[0]),
                                 name='ae_target_pl')
            layer = ae.pretrain_net(input_, n)

            with tf.name_scope("target"):
                target_for_loss = ae.pretrain_net(target_, n, is_target=True)

            if n == ae.num_hidden_layers+1:
                loss = loss_x_entropy(layer, target_for_loss)
            else:
                loss = tf.sqrt(tf.nn.l2_loss(tf.subtract(layer, target_for_loss)))
            train_op, global_step = training(loss, learning_rates[i], i)
    
            #summary_dir = pjoin(FLAGS.summary_dir, 'pretraining_{0}'.format(n))
            #summary_writer = tf.train.SummaryWriter(summary_dir,
            #                                        graph_def=sess.graph_def,
            #                                        flush_secs=FLAGS.flush_secs)
            #summary_vars = [ae["biases{0}".format(n)], ae["weights{0}".format(n)]]
    
            #hist_summarries = [tf.histogram_summary(v.op.name, v)
            #                   for v in summary_vars]
            #hist_summarries.append(loss_summaries[i])
            #summary_op = tf.merge_summary(hist_summarries)

            vars_to_init = ae.get_variables_to_init(n)
            vars_to_init.append(global_step)
            
                    
            saver = tf.train.Saver(vars_to_init)
            model_ckpt = _pretrain_model_dir+ '/model.ckpt'    
            
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
                sess.run(tf.variables_initializer(vars_to_init))
                print("\n\n")
                print("| Training Step | Cross Entropy |  Layer  |   Epoch  |")
                print("|---------------|---------------|---------|----------|")
        
                for epochs in range(FLAGS.pretraining_epochs):
                    perm = np.arange(num_train)
                    np.random.shuffle(perm)
                    for step in range(int(num_train/FLAGS.pretrain_batch_size)):
                        selectIndex = perm[FLAGS.pretrain_batch_size*step:FLAGS.pretrain_batch_size*step+FLAGS.pretrain_batch_size]
                        for I in range(len(batch_X[0])):
                            #input_feed = batch_X[perm[2*step:2*step+2],i,:]
                            ##target_feed = batch_Y[2*step:2*step+2,1]
                            #target_feed = batch_X[perm[2*step:2*step+2],i,:]
                            #feed_dict = {input_: input_feed,target_: target_feed}
                            #feed_dict = fill_feed_dict_ae(data.train, input_, target_, noise[i])
                            input_feed = batch_X[selectIndex,I,:]
                            target_feed = batch_X[selectIndex,I,:]
                            loss_summary, loss_value = sess.run([train_op, loss],
                                                            feed_dict={
                                                                input_: input_feed,
                                                                target_: target_feed
                                                                })
                    
                            count = epochs*num_train*len(batch_X[0])+step*len(batch_X[0])*len(input_feed)+(I+1)*len(input_feed)
                            #if count % 100 == 0:
                            if count % (10*len(input_feed)*len(batch_X[0])) == 0:
                        #summary_str = sess.run(summary_op, feed_dict=feed_dict)
                        #summary_writer.add_summary(summary_str, step)
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
                                        .format(count, loss_value, n, epochs + 1)
        
                                print(output)
                                
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
    return ae

def evaluation(logits, labels):
    """Evaluate the quality of the logits at predicting the label.
    
    Args:
        logits: Logits tensor, float - [batch_size, NUM_CLASSES].
        labels: Labels tensor, int32 - [batch_size], with values in the
            range [0, NUM_CLASSES).

    Returns:
        A scalar int32 tensor with the number of examples (out of batch_size)
        that were predicted correctly.
        """    
    correct_prediction = tf.equal(tf.round(logits), tf.cast(labels,tf.float32))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))    

def calculateKP(accu,kinst,fold,test_Y,test_label):
    playerMap =  np.array([[1,1,1,0,0],[1,1,0,1,0],[1,1,0,0,1],[1,0,1,1,0],[1,0,1,0,1],
                           [1,0,0,1,1],[0,1,1,1,0],[0,1,1,0,1],[0,1,0,1,1],[0,0,1,1,1]])
    
    bagAccu = np.zeros((5,1))
    pPrec = np.zeros((5,1))
    pAccu = np.zeros((5,1))         
    
    pred_positiveBag = np.flatnonzero(kinst)
    pred_pinst = np.zeros((len(pred_positiveBag),len(playerMap[0])))
    for p in range(len(pred_positiveBag)):
        k = int(kinst[int(pred_positiveBag[p])]-1)
        pred_pinst[p,:] = playerMap[k]
            
    test_positiveBag = np.flatnonzero(test_Y)
    test_pinst = np.zeros((len(test_positiveBag),len(playerMap[0])))
    test_kinst = np.zeros((len(test_positiveBag),1))
    for t in range(len(test_positiveBag)):
        pidx = test_positiveBag[t]
        test_kinst[t] = np.argmax(test_label[pidx,:])
        test_pinst[t,:] = playerMap[int(test_kinst[t])]
            
    correct_kP = len(np.flatnonzero(np.equal(test_pinst,pred_pinst)))
    total_P = test_pinst.shape[0]* test_pinst.shape[1]
    pAccu[fold] = correct_kP / total_P
            
    total_pP = len(np.flatnonzero(np.logical_or(test_pinst,pred_pinst)))
    tpP  = len(np.flatnonzero(np.logical_and(test_pinst,pred_pinst)))
    pPrec[fold] = tpP/total_pP
    print('bag accuracy %.5f, inst accuracy %.5f, inst precision %.5f' %(accu, pAccu[fold], pPrec[fold]))
    bagAccu[fold] = accu   
    #input("Press ENTER to CONTINUE")
    
    text_file.write('bag accuracy %.5f, inst accuracy %.5f, inst precision %.5f\n' %(accu, pAccu[fold], pPrec[fold]))
    time.sleep(1)

text_file = open("final_result.txt", "w")

def main_supervised(instNet,num_inst,fold,FLAGS):
    with instNet.session.graph.as_default():
        sess = instNet.session
        
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
        
        input_pl = tf.placeholder(tf.float32, shape=(num_inst,None,
                                                instNet.shape[0]),name='input_pl')
        Y, kinst = instNet.MIL(input_pl)
        
#        saver = tf.train.Saver()
#        if new:
        print("")
        print('fold %d' %(fold+1))
        datadir = 'dataGood/multiPlayers/syncLargeZoneVelocitySoftAssign(R=16,s=10)/train/fold%d' %(fold+1)
        fileName= 'EVZoneVelocitySoftAssign(R=16,s=10)3_training%d.mat' %(fold+1)

        batch_X, batch_Y, _ = readmat.read(datadir,fileName)
        num_train = len(batch_Y)
        
        testdir = 'dataGood/multiPlayers/syncLargeZoneVelocitySoftAssign(R=16,s=10)/test/fold%d' %(fold+1)
        testFileName= 'EVZoneVelocitySoftAssign(R=16,s=10)3_test%d.mat' %(fold+1) 
        test_X, test_Y, test_label = readmat.read(testdir,testFileName)        
        
        if FLAGS.finetune_batch_size is None:
            FLAGS.finetune_batch_size = len(test_Y)
        Y_placeholder = tf.placeholder(tf.int32,
                                        shape=(None,1),
                                        name='target_pl')
        
        loss = loss_x_entropy(Y, tf.cast(Y_placeholder, tf.float32))
        #loss = loss_supervised(logits, labels_placeholder)
        train_op, global_step = training(loss, FLAGS.supervised_learning_rate)
        accu = evaluation(Y, Y_placeholder)

    #hist_summaries = [ae['biases{0}'.format(i + 1)]
    #                  for i in xrange(ae.num_hidden_layers + 1)]
    #hist_summaries.extend([ae['weights{0}'.format(i + 1)]
    #                       for i in xrange(ae.num_hidden_layers + 1)])

    #hist_summaries = [tf.histogram_summary(v.op.name + "_fine_tuning", v)
    #                  for v in hist_summaries]
    #summary_op = tf.merge_summary(hist_summaries)

    #summary_writer = tf.train.SummaryWriter(pjoin(FLAGS.summary_dir,
    #                                              'fine_tuning'),
    #                                        graph_def=sess.graph_def,
    #                                        flush_secs=FLAGS.flush_secs)
        vars_to_init = instNet.get_variables_to_init(instNet.num_hidden_layers + 1)
        vars_to_init.append(global_step)
        sess.run(tf.variables_initializer(vars_to_init))
        
    
        #steps = FLAGS.finetuning_epochs * num_train
        for epochs in range(FLAGS.finetuning_epochs_epochs):
            perm = np.arange(num_train)
            np.random.shuffle(perm)
            print("|-------------|-----------|-------------|----------|")
            text_file.write("|-------------|-----------|-------------|----------|")
            for step in range(int(num_train/FLAGS.finetune_batch_size)):
                start_time = time.time()
            
                selectIndex = perm[FLAGS.finetune_batch_size*step:FLAGS.finetune_batch_size*step+FLAGS.finetune_batch_size]
                input_feed = batch_X[selectIndex,:,:]
                input_feed = np.transpose(input_feed, (1,0,2))
                target_feed = batch_Y[selectIndex].astype(np.int32)         
            
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
                    
            test_input_feed = np.transpose(test_X, (1,0,2))
            test_target_feed = test_Y.astype(np.int32) 
            bagAccu, kinst_pred = sess.run([accu, kinst],feed_dict={input_pl: test_input_feed,Y_placeholder: test_target_feed})
            print('Epochs %d: accuracy = %.5f '  % (epochs+1, bagAccu)) 
            text_file.write('Epochs %d: accuracy = %.5f\n\n'  % (epochs+1, bagAccu))
            
            baseline = 1-len(test_Y.nonzero())/len(test_Y)
            if bagAccu > baseline:
                calculateKP(bagAccu,kinst_pred,fold,test_Y,test_label)
            print("")

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
                    
                #if (step + 1) % 1000 == 0 or (step + 1) == steps:
                #    rain_sum = do_eval_summary("training_error",
                #                        sess,
                #                        eval_correct,
                #                        input_pl,
                #                        labels_placeholder,
                #                        data.train)
                #    val_sum = do_eval_summary("validation_error",
                #                      sess,
                #                      eval_correct,
                #                      input_pl,
                #                      labels_placeholder,
                #                      data.validation)
                #    test_sum = do_eval_summary("test_error",
                #                       sess,
                #                       eval_correct,
                #                       input_pl,
                #                       labels_placeholder,
                #                       data.test)
    
                #    summary_writer.add_summary(train_sum, step)
                #    summary_writer.add_summary(val_sum, step)
                #    summary_writer.add_summary(test_sum, step)
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
            
        input_feed = np.transpose(test_X, (1,0,2))
        target_feed = test_Y.astype(np.int32) 
        bagAccu, kinst_pred = sess.run([accu, kinst],feed_dict={input_pl: input_feed,Y_placeholder: target_feed})
        baseline = 1-len(test_Y.nonzero())/len(test_Y)
        print('After %d Epochs: accuracy = %.5f (baseline: %.5f)'  % (epochs+1, bagAccu, baseline))
        text_file.write('After %d Epochs: accuracy = %.5f (baseline: %.5f)'  % (epochs+1, bagAccu, baseline))
        time.sleep(0.5)
 
        if bagAccu > baseline:
            calculateKP(bagAccu,kinst_pred,fold,test_Y,test_label)
            
        text_file.close()
            
