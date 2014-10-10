# coding=utf-8

import sys
import random
import numpy as np
from math import sqrt
from common import read_dense_data
from common import sigmoid

random.seed(1024 * 1024)

from cg import CG

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

class LogisticRegression:

    def __init__(self):        
        self.w = None
        self.c = 0

    def train(self, X, Y, lamb = 1.0):
        
        m, n = X.shape

        x0 = np.matrix(np.ones([m, 1]))
        X = np.column_stack([X, x0])

        w_init = np.matrix(np.zeros([n + 1, 1]))

        # self.w = GD(w_init, self.cost)
        self.w = CG(self.cost, w_init, X = X, Y = Y, lamb = lamb)
        
        print 'Done with function evalution C = %d' % self.c

    def test(self, X, Y):
        m, n = X.shape
        x0 = np.matrix(np.ones([m, 1]))
        X = np.column_stack([X, x0])

        Y_pred = np.sign(X * self.w)
        Y_pred[np.where(Y_pred == -1)] = 0

        P = np.matrix(np.zeros(Y.shape))
        P[np.where(Y == Y_pred)] = 1

        print >> sys.stderr, 'Accuracy : %lf%% (%d/%d)' % (100.0 * P.sum() / len(Y), P.sum(), len(Y))

        return 1.0 * P.sum() / len(Y) 

    
    def cost(self, w, X, Y, lamb):   
        m = len(X)
        S = sigmoid(X * w)
    
        L = (1.0 / (2 * m)) * (- Y.T * np.log(S) - (1.0 - Y).T * np.log(1.0 - S))
        fx = float(L + (lamb / 2.0) * (w.T * w))

        df = (1.0 / m) * X.T * (S - Y) + 1.0 * lamb * w

        self.c += 1

        return fx, df
    

if __name__ == '__main__':
    
    train_path = 'data/heart_scale.train'
    test_path = 'data/heart_scale.test'

    X_train, Y_train = read_dense_data(open(train_path))
    X_test, Y_test = read_dense_data(open(test_path))

    X_train = np.matrix(X_train)
    Y_train = [int(y) for y in Y_train]
    Y_train = np.matrix(Y_train).T
    Y_train[np.where(Y_train == -1)] = 0
 
    X_test = np.matrix(X_test)
    Y_test = [int(y) for y in Y_test]    
    Y_test = np.matrix(Y_test).T
    Y_test[np.where(Y_test == -1)] = 0   
 
    clf = LogisticRegression()
    clf.train(X_train, Y_train) 
    acc_lr = clf.test(X_train, Y_train)
    print acc_lr
    acc_lr = clf.test(X_test, Y_test)

    print acc_lr

    print >> sys.stderr, 'Accuracy for Logistic Regression : %lf%%' % (100.0 * acc_lr)

