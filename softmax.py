# coding=utf-8

import sys
import numpy as np
import math
import random

from common import read_dense_data
from common import sigmoid
from common import align


random.seed(1024 * 1024)

from cg import CG
from gd import SGDOption
from gd import SGD

class SoftmaxRegression:

    def __init__(self):
        self.w = None
        self.c = 0

    def train(self, X, Y, lamb = 1.0):

        O = len(set([v[0] for v in Y.tolist()]))
        m, n = X.shape
 
        w = np.matrix(0.005 * np.random.random([O, n])).reshape(-1, 1)
    
        opt = SGDOption()
        opt.max_iter = 50
        opt.mini_batch_size = 5000

        # w_opt = SGD(self.cost, w, X, Y, opt, lamb = lamb, O = O)

        w_opt = CG(self.cost, w, 80, X = X, Y = Y, lamb = lamb, O = O)

        self.w = w_opt.reshape(O, n)

        print >> sys.stderr, 'c = ', self.c
    
    def cost(self, w, X, Y, lamb, O):

        m, n = X.shape

        w = w.reshape(O, n)
 
        I = Y.T
        Y = np.matrix(np.zeros((O, m)))
        Y[(I, np.matrix(range(m)))] = 1
        
        P = np.exp(w * X.T)
        P = P / P.sum(0)
        
        L = - (1.0 / m) * np.multiply(Y, np.log(P)).sum()
        R = (lamb / 2.0) * np.square(w).sum()
      
        J = L + R
       
        grad = - (1.0 / m) * (Y - P) * X + lamb * w
        grad = grad.reshape(-1, 1)       
 
        self.c += 1

        return J, grad
            
    def predict(self, X):
        P = np.exp(self.w * X.T)
        P = P / P.sum(0)
        return np.argmax(P, 0).T

    def test(self, X, Y):

        Y_pred = self.predict(X)
        P = np.matrix(np.zeros(Y.shape))
        P[np.where(Y_pred == Y)] = 1
       
        acc = 1.0 * P.sum() / len(Y)
        print >> sys.stderr, 'Accuracy %lf%% (%d/%d)' % (100.0 * acc, P.sum(), len(Y))
 
        return 1.0 * P.sum() / len(Y)

if __name__ == '__main__':
    
    train_path = 'data/mnist.train'
    test_path = 'data/mnist.test'

    X_train, Y_train = read_dense_data(open(train_path)) 
    print >> sys.stderr, 'read training data done.'
    X_train = np.matrix(X_train)
    Y_train = [int(y) for y in Y_train]
    Y_train = np.matrix(Y_train).T
    print >> sys.stderr, 'create training matrix done.'

    X_test, Y_test = read_dense_data(open(test_path))
    print >> sys.stderr, 'read test data done'
    X_test = np.matrix(X_test)
    Y_test = [int(y) for y in Y_test]    
    Y_test = np.matrix(Y_test).T
    print >> sys.stderr, 'create test matrix done.'

    X_train, X_test = align(X_train, X_test)

    '''
    X_all = np.row_stack([X_train, X_test])
    print X_all.shape
    mean = X_all.mean(0)
    std = X_all.std(0)
    del X_all
    
    X_train = 1.0 * (X_train - mean) / (std + 0.0001)
    X_test = 1.0 * (X_test - mean) / (std + 0.0001)
    '''

    clf = SoftmaxRegression()
    clf.train(X_train, Y_train) 
    # clf.train(X_test, Y_test) 
    acc_train = clf.test(X_train, Y_train)
    acc_test = clf.test(X_test, Y_test)

    print >> sys.stderr, 'Training accuracy for Softmax Regression : %lf%%' % (100.0 * acc_train)
    print >> sys.stderr, 'Test accuracy for Softmax Regression : %lf%%' % (100.0 * acc_test)

