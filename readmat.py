
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 21:58:32 2017

@author: PeterTsai
"""
import os
import scipy.io
import numpy as np

def read(datadir, fileName):
    #print(os.path.join(os.path.sep,os.getcwd(),datadir,fileName))
    
    #print(os.path.exists(os.path.join(os.path.sep,os.getcwd(),datadir,fileName)))
    mat = scipy.io.loadmat(os.path.join(os.path.sep,os.getcwd(),datadir,fileName))
    
    #print(mat['bags'].shape)
    #print(mat['bags']['instance'].shape)
    #print(mat['bags']['instance'].dtype)
    #print(mat['bags']['instance'][0][0].shape)
    bags = mat['bags']

    #strHello = "the length of (%s) is %d" %('Hello World',len('Hello World'))
    #print(strHello)
    
    strBagShape = "the shape of bags is ({0},{1})".format(bags.shape[0],bags.shape[1])
    print(strBagShape)
    #print(bags.shape)
    
    bagNum = len(bags['instance'][0])
    instNum, featDim = (bags['instance'][0][0]).shape
    X = np.zeros([bagNum,instNum,featDim])
    #print(len(bags['instance'][0]))
    #for b in range(len(bags['instance'][0])):
    #    X = bags['instance'][0,b]
    #    print(np.sum(X,axis=1))
    #    Bag1 = np.array(bags['instance'][0][0])
    #    print(np.sum(Bag1,axis=1))
    #X = np.array(bags['instance'])
    for b in range(len(bags['instance'][0])):
        X[b,:,:] = bags['instance'][0,b]

    #print(len(bags['inst_label'][0]))
    Y = np.zeros((len(bags['label'][0]),1))
    for b in range(len(bags['label'][0])):
        Y[b] = int(bags['label'][0][b])
    
    labels = np.zeros([bagNum,instNum])
    for i in range(len(bags['inst_label'][0,:])):
        labels[i,:] = np.squeeze(bags['inst_label'][0][i][0])
        #print(label.shape)
        #print(label.dtype)
        #print(label)
        #print(np.sum(label,axis=1))
    
    #print(Y)
    return X,Y,labels
    

if __name__ == "__main__":
    datadir = 'dataGood/multiPlayers/syncLargeZoneVelocitySoftAssign(R=16,s=10)'
    fileName = 'EVZoneVelocitySoftAssign(R=16,s=10)3.mat'
    
    X_batch, Y_batch = read(datadir,fileName)
    #print(X_batch[:,1])
    #print(type(X_batch))
    #print(type(Y_batch))
