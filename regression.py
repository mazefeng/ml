# coding=utf-8

import sys
import random
import math
import numpy as np
from math import sqrt
from common import read_dense_data

random.seed(1024 * 1024)

from cg import CG
from gd import SGDOption
from gd import SGD


class LinearRegression:

    def __init__(self):        
        self.w = None
        self.c = 0

    def train(self, X, Y, lamb = 0.0001):
        
        m, n = X.shape

        x0 = np.matrix(np.ones([m, 1]))
        X = np.column_stack([X, x0])

        self.X = X
        self.Y = Y
        self.lamb = lamb

        w = np.matrix(np.zeros([n + 1, 1]))

        '''
        opt = SGDOption()
        opt.mini_batch_size = 100
        opt.eps = 1e-8

        self.w = SGD(self.cost, w, X, Y, opt, lamb = lamb)
        '''
        self.w = CG(self.cost, w, 200, X = X, Y = Y, lamb = lamb)
        
        print 'Done with function evalution C = %d' % self.c

    def test(self, X, Y):
        m, n = X.shape
        x0 = np.matrix(np.ones([m, 1]))
        X = np.column_stack([X, x0])

        r = X * self.w - Y
        rmse = sqrt(float(r.T * r) / len(Y))

        print >> sys.stderr, 'Test RMSE : %lf' % rmse
        return rmse
    
    # (1 / (2 * m)) * (X * w.T - Y) ^ 2 + (lamb / 2) * (w.T * w)
    # (1 / m) * (X * w.T - Y) * X + lamb * w
    def cost(self, w, X, Y, lamb): 

  
        m = len(X)
        D = (X * w - Y)
        fx = float((1.0 / (2 * m)) * (D.T * D) + (lamb / 2.0) * (w.T * w))
        df = (1.0 / m) * X.T * (X * w - Y) + 1.0 * lamb * w

        self.c += 1

        return fx, df
    

if __name__ == '__main__':
    
    train_path = 'data/housing.train'
    test_path = 'data/housing.test'

    X_train, Y_train = read_dense_data(open(train_path))
    X_test, Y_test = read_dense_data(open(test_path))

    X_train = np.matrix(X_train)
    Y_train = [float(y) for y in Y_train]
    Y_train = np.matrix(Y_train).T
 
    X_test = np.matrix(X_test)
    Y_test = [float(y) for y in Y_test]    
    Y_test = np.matrix(Y_test).T
    
    reg = LinearRegression()
    reg.train(X_train, Y_train) 
    reg.test(X_train, Y_train) 
    reg.test(X_test, Y_test)


