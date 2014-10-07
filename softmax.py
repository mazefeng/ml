# coding=utf-8

import sys
import numpy as np
import math
import random

from common import read_dense_data
from common import align_X

random.seed(1024 * 1024)

from cg import CG
from gd import GD

def unroll(x):
    if type(x) == np.matrix:
        theta = x.reshape(-1, 1)
    else:
        theta = x[0].reshape(-1, 1)
        for i in range(1, len(x)):
            theta = np.append(theta, x[i].reshape(-1, 1), 0)
    return theta

class SoftmaxRegression:

    def __init__(self):
        self.w = None
        self.c = 0

    def train(self, X, Y, lamb = 1.0):

        L = len(set([v[0] for v in Y.tolist()]))
        m, n = X.shape
 
        w_init = np.matrix(0.005 * np.random.random([L, n]))  
        theta_init = unroll(w_init)

        theta = CG(self.cost, theta_init, X = X, Y = Y, lamb = lamb, L = L)

        self.w = theta.reshape(L, n)

        print 'c = ', self.c
    
    def cost(self, theta, X, Y, lamb, L):

        m, n = X.shape

        w = theta.reshape(L, n) 
        Y_encode = np.matrix(np.zeros((L, m)))
        Y_encode[(Y.T, np.matrix(range(m)))] = 1
        
        likelihood = np.exp(w * X.T)
        likelihood = likelihood / np.sum(likelihood, 0)
        
        cost = - (1.0 / m) * np.sum( np.multiply(Y_encode, np.log(likelihood)) ) + (lamb / 2.0) * np.sum(np.square(theta))
        grad = - (1.0 / m) * (Y_encode - likelihood) * X + lamb * w
        grad = unroll(grad)
        
        self.c += 1

        return cost, grad
            
    def predict(self, X):
        likelihood = np.exp(self.w * X.T)
        likelihood = likelihood / np.sum(likelihood, 0)
        return np.argmax(likelihood, 0).T

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

    X_test, Y_test = read_dense_data(open(test_path))

    print >> sys.stderr, 'read test data done'

    X_train = np.matrix(X_train)
    Y_train = [int(y) for y in Y_train]
    Y_train = np.matrix(Y_train).T

    print >> sys.stderr, 'create training matrix done.'
 
    X_test = np.matrix(X_test)
    Y_test = [int(y) for y in Y_test]    
    Y_test = np.matrix(Y_test).T
 
    print >> sys.stderr, 'create test matrix done.'

    X_train, X_test = align_X(X_train, X_test)

    clf = SoftmaxRegression()
    clf.train(X_train, Y_train) 
    acc_train = clf.test(X_train, Y_train)
    acc_test = clf.test(X_test, Y_test)

    print >> sys.stderr, 'Training accuracy for Softmax Regression : %lf%%' % (100.0 * acc_train)
    print >> sys.stderr, 'Test accuracy for Softmax Regression : %lf%%' % (100.0 * acc_test)

