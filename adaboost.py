# coding=utf-8

import sys
import random
import numpy as np
from math import log

from common import read_dense_data
from common import map_label

random.seed(1024 * 1024)

class WeakLearner:

    def train(self, X, Y, D, alpha = 1.0, max_update = 100, eps = 5e-2):
        m, n = X.shape
        update = 0

        min_loss = 1.0
        w_best = np.matrix(np.zeros([n, 1]))
        Y_best = np.matrix(np.zeros([m, 1]))

        w = np.matrix(np.zeros([n, 1]))

        while update <= max_update and min_loss > eps:
            i = random.randint(0, m - 1)
            if np.sign(X[i] * w) == Y[i]: continue
            w += alpha * X[i].T * Y[i] * D[i]
            update += 1
            # w += alpha * m * X[i].T * Y[i] * D[i]
            loss, Y_pred = self.loss(X, Y, D, w)
            if loss < min_loss:
                min_loss = loss
                w_best = np.matrix(w)
                Y_best = np.matrix(Y_pred)
        
        return min_loss, w_best, Y_best
    
    def classify(self, X, w):
        Y_pred= np.sign(X * w)
        return Y_pred

    def loss(self, X, Y, D, w):
        Y_pred = self.classify(X, w)
        P = np.matrix(np.zeros(Y.shape))
        P[np.where(Y_pred != Y)] = 1
        return (D.T * P).sum(), Y_pred

class AdaBoost:
    def __init__(self):
        
        self.weak_learner = WeakLearner()

    def train(self, X, Y, run = 40):
        
        m, n = X.shape        

        D = (1.0 / m) * np.matrix(np.ones([m, 1]))
 
        self.alpha = [0.0] * run
        self.W = np.matrix(np.zeros([n, run]))

        for i in range(run):
            
            loss, self.W[:, i], Y_pred = self.weak_learner.train(X, Y, D)
            
            self.alpha[i] = 0.5 * log((1.0 - loss) / loss)
            assert self.alpha[i] > 0
            
            D = np.multiply(D, np.exp(-self.alpha[i] * np.multiply(Y, Y_pred)))
            D = D / D.sum()

            # test training data after each round
            # set a larger run, loss on training data will be 0.0
            print >> sys.stderr, 'run: %d' % i
            self.test(X, Y)

    def classify(self, X):
        m, n = X.shape
        S = np.matrix(np.zeros([m, 1]))
        for i in range(len(self.alpha)):
            S += self.alpha[i] * self.weak_learner.classify(X, self.W[:, i])
        Y_pred = np.sign(S)
        return Y_pred
        
    def test(self, X, Y):
        P = np.matrix(np.zeros(Y.shape))
        Y_pred = self.classify(X)
        P[np.where(Y_pred == Y)] = 1
        c = P.sum()
    
        print >> sys.stderr, 'Accuracy = %lf%% (%d/%d)' % (100.0 * c / len(Y), c, len(Y))

        return 1.0 * c / len(Y)
        
   
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

    clf = AdaBoost()
    clf.train(X_train, Y_train)
    acc = clf.test(X_test, Y_test)
    print >> sys.stderr, 'Accuracy for AdaBoost : %lf%%' % (100 *  acc)

