# coding=utf-8

import sys
import random
import math
import numpy as np
from common import read_dense_data
from common import map_label

random.seed(1024 * 1024)

class Perceptron:

    def __init__(self):
        self.w = None

    def train(self, X, Y, alpha = 1.0, max_update = 1000, max_iter = 10000):
        m, n = X.shape
        I = range(m)
        self.w = np.matrix(np.zeros([n, 1]))
        
        update = 0
        for iter in range(max_iter):
            random.shuffle(I)
            err = 0
            for i in I:
                if np.sign(X[i] * self.w) == Y[i]: continue       
                self.w += alpha * X[i].T * Y[i]
                update += 1
                if update > max_update: return
                err += 1
            print >> sys.stderr, 'Iter : %d\ttraining loss: %lf' % (iter, 1.0 * err / len(Y))

    def test(self, X, Y):
       
        Y_pred = np.sign(X * self.w)
        P = np.matrix(np.zeros(Y.shape))
        P[np.where(Y == Y_pred)] = 1
        return 1.0 * P.sum() / len(Y)
        
class Pocket(Perceptron):

    def train(self, X, Y, alpha = 1.0, max_update = 1000):
        m, n = X.shape
        self.w = np.matrix(np.zeros([n, 1]))

        update = 0
        min_err = m
        w_test = np.matrix(np.zeros(self.w.shape))
        while update <= max_update:
            i = random.randint(0, m - 1)
            if np.sign(X[i] * w_test) == Y[i]: continue
            w_test += alpha * X[i].T * Y[i]
            update += 1
            
            P = np.matrix(np.zeros(Y.shape))            
            Y_pred = np.sign(X * w_test)
            P[np.where(Y != Y_pred)] = 1
            err = P.sum()

            if update % 100 == 0:
                print >> sys.stderr, 'Update : %d\ttraining loss: %lf' % (update, 1.0 * err / len(Y))

            if err < min_err:
                min_err = err
                self.w = w_test

if __name__ == '__main__':
    
    train_path = 'data/heart_scale.train'
    test_path = 'data/heart_scale.test'

    X_train, Y_train = read_dense_data(open(train_path))
    X_test, Y_test = read_dense_data(open(test_path))

    X_train = np.matrix(X_train)
    Y_train = np.matrix(map_label(Y_train)).T
    Y_train[np.where(Y_train == 0)] = -1
 
    X_test = np.matrix(X_test)
    Y_test = np.matrix(map_label(Y_test)).T
    Y_test[np.where(Y_test == 0)] = -1    

    clf = Perceptron()
    clf.train(X_train, Y_train)
    perceptron_acc = clf.test(X_test, Y_test)
    print >> sys.stderr, 'Accuracy for Perceptron : %lf%%' % (100 *  perceptron_acc)

    clf = Pocket()
    clf.train(X_train, Y_train)
    pocket_acc = clf.test(X_test, Y_test)
    print >> sys.stderr, 'Accuracy for Pocket : %lf%%' % (100 *  pocket_acc)
 
