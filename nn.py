# coding=utf-8

import sys
import numpy as np
import math
import random

from common import read_dense_data
from common import align
from common import sigmoid

random.seed(1024 * 1024)

from cg import CG
from gd import SGDOption
from gd import SGD

def normalize(X0, X1):
    X_all = np.row_stack([X0, X1])
    mean = X_all.mean(0)
    std = X_all.std(0)
    del X_all

    X0 = 1.0 * (X0 - mean) / (std + 0.0001)
    X1 = 1.0 * (X1 - mean) / (std + 0.0001)

    return X0, X1

class NeuralNetwork:

    def __init__(self):
        self.w1 = None
        self.w2 = None
        self.c = 0

    def train(self, X, Y, lamb = 1.0, H = 25):

        O = len(set([v[0] for v in Y.tolist()]))
        m, I = X.shape

        w1 = np.matrix(0.005 * np.random.random([H, I + 1]))
        w2 = np.matrix(0.005 * np.random.random([O, H + 1]))

        w = np.row_stack([w1.reshape(-1, 1), w2.reshape(-1, 1)])
        
        '''
        opt = SGDOption()
        opt.max_iter = 10
        opt.mini_batch_size = 500
        opt.eps = 1e-8
        opt.alpha_decay = lambda x : 0.4 / math.sqrt(x)
        
        w_opt = SGD(self.cost, w, X, Y, opt, lamb = lamb, I = I, H = H, O = O)
        '''

        w_opt = CG(self.cost, w, 90, X = X, Y = Y, lamb = lamb, I = I, H = H, O = O)

        self.w1 = w_opt[0 : H * (I + 1)].reshape(H, I + 1)
        self.w2 = w_opt[H * (I + 1) :  ].reshape(O, H + 1)

        print 'c = ', self.c

    def cost(self, w, X, Y, lamb, I, H, O):

        m, n = X.shape

        w1 = w[0 : H * (I + 1)].reshape(H, I + 1)
        w2 = w[H * (I + 1) :  ].reshape(O, H + 1)
        # m, n + 1
        a1 = np.column_stack([X, np.matrix(np.ones([m, 1]))])
        # m, H
        z2 = a1 * w1.T
        s2 = sigmoid(z2)
        # m, H + 1
        a2 = np.column_stack([s2, np.matrix(np.ones([m, 1]))])
        # m, O
        z3 = a2 * w2.T
        # m, O
        a3 = sigmoid(z3)

        I = Y.T
        Y = np.matrix(np.zeros([m, O]))
        Y[(np.matrix(range(m)), I)] = 1
        
        L = (1.0 / m) * (- np.multiply(Y, np.log(a3)) - np.multiply(1.0 - Y, np.log(1.0 - a3))).sum()

        R = (lamb / (2.0 * m)) * (np.square(w1[:, 0 : -1]).sum() + np.square(w2[:, 0 : -1]).sum())

        J = L + R
        # m, O
        delta3 = a3 - Y
        # m, H 
        delta2 = np.multiply(delta3 * w2[:, 0 : -1], np.multiply(s2, 1.0 - s2))
        # H, n + 1
        l1_grad = delta2.T * a1
        # O, H + 1
        l2_grad = delta3.T * a2
        
        r1_grad = np.column_stack([w1[:, 0 : -1], np.matrix(np.zeros([H, 1]))])
        r2_grad = np.column_stack([w2[:, 0 : -1], np.matrix(np.zeros([O, 1]))])

        w1_grad = (1.0 / m) * l1_grad + (1.0 * lamb / m) * r1_grad
        w2_grad = (1.0 / m) * l2_grad + (1.0 * lamb / m) * r2_grad
        
        grad = np.row_stack([w1_grad.reshape(-1, 1), w2_grad.reshape(-1, 1)])
    
        self.c += 1

        return J, grad
     
            
    def predict(self, X):

        m, n = X.shape
        O = len(self.w2)
        # m, I + 1
        X = np.column_stack([X, np.matrix(np.ones([m, 1]))])
        # m, H
        h1 = sigmoid(X * self.w1.T)
        # m, H + 1
        h1 = np.column_stack([h1, np.matrix(np.ones([m, 1]))])
        # m, O
        h2 = sigmoid(h1 * self.w2.T)
        
        return np.argmax(h2, 1) 
        
    def test(self, X, Y):

        Y_pred = self.predict(X)
        P = np.matrix(np.zeros(Y.shape))
        P[np.where(Y_pred == Y)] = 1
       
        acc = 1.0 * P.sum() / len(Y)
        print >> sys.stderr, 'Accuracy %lf%% (%d/%d)' % (100.0 * acc, P.sum(), len(Y))
 
        return 1.0 * P.sum() / len(Y)

if __name__ == '__main__':
    
    # train_path = 'data/mini_mnist'
    train_path = 'data/mnist.train'
    # test_path = 'data/mini_mnist'
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
    X_train, X_test = normalize(X_train, X_test)
   
    clf = NeuralNetwork()
    clf.train(X_train, Y_train) 
    # clf.train(X_test, Y_test) 
    acc_train = clf.test(X_train, Y_train)
    acc_test = clf.test(X_test, Y_test)

    print >> sys.stderr, 'Training accuracy for Neural Network : %lf%%' % (100.0 * acc_train)
    print >> sys.stderr, 'Test accuracy for Neural Network : %lf%%' % (100.0 * acc_test)

