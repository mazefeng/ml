# coding=utf-8

import sys
import random

import numpy as np
from math import sqrt
from scipy.spatial.distance import squareform, cdist

import matplotlib
matplotlib.use('wx')
import matplotlib.pylab as plt

random.seed(1024 * 1024)

from common import read_dense_data
from common import map_label

##################################################################################################################
'''
Sequential Minimization Optimization(SMO) for training SVMs.
'''

L = lambda Xi, Xj: Xi * Xj.T

linear_kernel = L

def rbf_kernel(X_i, X_j, sigma = 2.0):
    pairwise_dists = cdist(X_i, X_j, 'euclidean')
    K = np.exp(-(pairwise_dists ** 2) / (2 * sigma * sigma))
    return np.matrix(K)

class SMO:

    def __init__(self):
        
        self.model = None

    def train(self, X, Y, K = L, C = 1.0, tol = 1e-3, max_passes = 5):
 
        m, n = X.shape
        alphas = np.matrix(np.zeros([m, 1]))
        b = 0.0
       
        K_cache = K(X, X)

        print >> sys.stderr, 'Done with K_cache'
 
        iter = 0
        passes = 0
        
        while passes < max_passes:
            iter += 1
            
            if iter % 10 == 0: sys.stderr.write('.')
            if iter % 500 == 0: sys.stderr.write('%d Iters\n' % iter)

            # print >> sys.stderr, 'Iter :', iter
            num_changed_alphas = 0
            for i in range(m):
                fx_i = alphas.T * np.multiply(Y, K_cache[:, i]) + b
                y_i = Y[i]
                E_i = fx_i - y_i
                alpha_i = alpha_ii = alphas[i]

                if (y_i * E_i < -tol and alpha_i < C) or (y_i * E_i > tol and alpha_i > 0.0):
                    while True:
                        j = random.randint(0, m - 1)
                        if i != j: break
                    
                    fx_j = alphas.T * np.multiply(Y, K_cache[:, j]) + b
                    y_j = Y[j]
                    E_j = fx_j - y_j
                    alpha_j = alpha_jj = alphas[j]
                    if y_i != y_j:
                        L = max(0.0, alpha_j - alpha_i)
                        H = min(C, C + alpha_j - alpha_i)
                    else:
                        L = max(0.0, alpha_i + alpha_j - C)
                        H = min(C, alpha_i + alpha_j)
                    if L == H: continue
                
                    eta = 2 * K_cache[i, j] - K_cache[i, i] - K_cache[j, j]
                    if eta >= 0.0: continue
                    
                    alpha_j = alpha_j - (y_j * (E_i - E_j) / eta)
                    if alpha_j > H: alpha_j = H
                    if alpha_j < L: alpha_j = L
                    if abs(alpha_jj - alpha_j) < tol: continue
                    
                    alpha_i = alpha_i + (y_i * y_j * (alpha_jj - alpha_j))
                    
                    # b_i = b - E_i - y_i * (alpha_i - alpha_ii) * K_cache[i, j] - y_j * (alpha_j - alpha_jj) * K_cache[i, j]
                    b_i = b - E_i - y_i * (alpha_i - alpha_ii) * K_cache[i, i] - y_j * (alpha_j - alpha_jj) * K_cache[i, j]
                    b_j = b - E_j - y_i * (alpha_i - alpha_ii) * K_cache[i, j] - y_j * (alpha_j - alpha_jj) * K_cache[j, j]
                    
                    if alpha_i > 0.0 and alpha_i < C:
                        b = b_i
                    elif alpha_j > 0.0 and alpha_j < C:
                        b = b_j
                    else:
                        b = (b_i + b_j) / 2.0
                    
                    alphas[i] = alpha_i
                    alphas[j] = alpha_j
                    
                    num_changed_alphas = num_changed_alphas + 1

            if num_changed_alphas == 0:
                passes += 1
            else:
                passes = 0
                        
        sys.stderr.write('\nDone training with Iter %d\n' % iter)

        self.model = dict()
        alpha_index = [ index for index, alpha in enumerate(alphas) if alpha > 0.0 ]
        
        self.model['X'] = X[alpha_index]
        self.model['Y'] = Y[alpha_index]
        self.model['kernel'] = K
        self.model['alphas'] = alphas[alpha_index]
        self.model['b'] = b
        self.model['w'] = X.T * np.multiply(alphas, Y)  
   
    def predict(self, X_test):
        m, n = X_test.shape
        fx = np.matrix(np.zeros([m, 1]))
        if self.model['kernel'] == L:
            w = self.model['w']
            b = self.model['b']
            fx = X_test * w + b
        else:
            alphas = self.model['alphas']
            X = self.model['X']
            Y = self.model['Y']
            K = self.model['kernel']
            b = self.model['b']
            fx = np.multiply(np.tile(Y, [m]), K(X, X_test)).T * alphas + b
        return fx
    
    def test(self, X, Y):
        fx = self.predict(X)
        Y_pred = np.matrix(np.zeros(Y.shape))
        Y_pred[np.where(fx >= 0)] = 1
        Y_pred[np.where(fx < 0)] = -1

        P = np.matrix(np.zeros(Y.shape))
        P[np.where(Y_pred == Y)] = 1 

        return 1.0 * P.sum() / len(Y)

##################################################################################################################
'''
Primal Estimated sub-GrAdient SOlver for SVMs(Pegasos) for training linear SVMs in primal form
'''

class Pegasos:

    def __init__(self):

        self.w = None
        
    def train(self, X, Y, T = 1000, lamb = 0.01, k = 80):

        x0 = np.matrix(np.ones([len(X), 1]))
        X = np.column_stack([X, x0])

        m, n = X.shape
        
        I = range(m)
        self.w = np.matrix(np.random.random([n, 1]))
        self.w = self.w / sqrt(lamb * np.square(self.w).sum())
        for t in range(1, T + 1):
            random.shuffle(I)
            J = np.where(np.multiply(X[I[0 : k]] * self.w, Y[I[0 : k]]) < 1)
            eta = 1.0 / (lamb * t)
            self.w = (1 - eta * lamb) * self.w + (eta / k) * X[J[0].tolist()].T * Y[J[0].tolist()]
            self.w = min(1.0, 1.0 / sqrt(lamb * np.square(self.w).sum())) * self.w

    def predict(self, X):
        
        x0 = np.matrix(np.ones([len(X), 1]))
        X = np.column_stack([X, x0])
        return X * self.w

    def test(self, X, Y):
        
        Y_pred = np.sign(self.predict(X))
        P = np.matrix(np.zeros(Y.shape))
        P[np.where(Y == Y_pred)] = 1
        return 1.0 * P.sum() / len(Y)


##################################################################################################################
def plot_hyperplane(X, Y, model, K, plot_id, d = 500):
    I0 = np.where(Y==-1)[0]
    I1 = np.where(Y==1)[0]

    plt.subplot(plot_id)

    plt.plot(X[I1, 0], X[I1, 1], 'og')
    plt.plot(X[I0, 0], X[I0, 1], 'xb')

    min_val = np.min(X, 0)
    max_val = np.max(X, 0)

    clf = model()
    clf.train(X, Y, K)

    x0_plot = np.linspace(min_val[0, 0], max_val[0, 0], d)
    x1_plot = np.linspace(min_val[0, 1], max_val[0, 1], d)

    [x0, x1] = plt.meshgrid(x0_plot, x1_plot);

    Y_all = np.matrix(np.zeros([d, d]))

    for i in range(d):
        X_all = np.matrix(np.zeros([d, 2]))
        X_all[:, 0] = np.matrix(x0[:, i]).T
        X_all[:, 1] = np.matrix(x1[:, i]).T
        Y_all[:, i] = clf.predict(X_all)

    plt.contour(np.array(x0), np.array(x1), np.array(Y_all), levels = [0.0], colors = 'red')

if __name__ == '__main__':

    # set rbf_kernel.gamma = 0.1
    '''
    X_1, Y_1 = read_dense_data(open('data/sample_data_1.txt'))
    X_2, Y_2 = read_dense_data(open('data/sample_data_2.txt'))
    X_3, Y_3 = read_dense_data(open('data/sample_data_3.txt'))

    X_1 = np.matrix(X_1)
    X_2 = np.matrix(X_2)
    X_3 = np.matrix(X_3)

    Y_1 = np.matrix(map_label(Y_1)).T
    Y_2 = np.matrix(map_label(Y_2)).T
    Y_3 = np.matrix(map_label(Y_3)).T
    
    Y_1[np.where(Y_1 == 0)] = -1
    Y_2[np.where(Y_2 == 0)] = -1
    Y_3[np.where(Y_3 == 0)] = -1
   
    plot_hyperplane(X_1, Y_1, SMO, linear_kernel, 231)
    plot_hyperplane(X_2, Y_2, SMO, linear_kernel, 232)
    plot_hyperplane(X_3, Y_3, SMO, linear_kernel, 233)
    
    plot_hyperplane(X_1, Y_1, SMO, rbf_kernel, 234)
    plot_hyperplane(X_2, Y_2, SMO, rbf_kernel, 235)
    plot_hyperplane(X_3, Y_3, SMO, rbf_kernel, 236)
    # plt.savefig('svm.png') 
    plt.show()
    '''

    # heart_scale dataset
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

    clf = Pegasos()
    clf.train(X_train, Y_train)
    acc_train = clf.test(X_train, Y_train)
    acc_test  = clf.test(X_test, Y_test)
    print >> sys.stderr, 'Training accuracy for Pegasos : %lf%%' % (100 *  acc_train)
    print >> sys.stderr, 'Test accuracy for Pegasos : %lf%%' % (100 *  acc_test)
    
    clf = SMO()
    clf.train(X_train, Y_train, linear_kernel)
    acc_train = clf.test(X_train, Y_train)
    acc_test  = clf.test(X_test, Y_test)

    print >> sys.stderr, 'Training accuracy for SMO with linear kernel : %lf%%' % (100 *  acc_train)
    print >> sys.stderr, 'Test accuracy for SMO with linear kernel : %lf%%' % (100 *  acc_test)

    clf = SMO()
    clf.train(X_train, Y_train, rbf_kernel)
    acc_train = clf.test(X_train, Y_train)
    acc_test  = clf.test(X_test, Y_test)

    print >> sys.stderr, 'Training accuracy for SMO with rbf kernel : %lf%%' % (100 *  acc_train)
    print >> sys.stderr, 'Test accuracy for SMO with rbf kernel : %lf%%' % (100 *  acc_test)



