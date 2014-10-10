# coding=utf-8

import sys
import numpy as np
'''
    All regression/classification data are stored in the same format as LIBSVM.

    Use read_sparse_data for loading sparse data
    Use read_dense_data for loading dense data

    When loading dense data, feature id must be integer.
'''

# sigmoid
sigmoid = lambda x : 1.0 / (1.0 + np.exp(-x))

def align(X0, X1):
    '''
    align X0 and X1 by column
    '''
    m0, n0 = X0.shape
    m1, n1 = X1.shape
    if n0 > n1:
        c = np.matrix(np.zeros([m1, n0 - n1]))
        X1 = np.column_stack([X1, c])
    elif n1 > n0:
        c = np.matrix(np.zeros([m0, n1 - n0]))
        X0 = np.column_stack([X0, c])
    return X0, X1    


def trace():
    '''
    print the function and line number that throws an exception
    '''
    try:
        raise Exception
    except:
        f = sys.exc_info()[2].tb_frame.f_back
    print >> sys.stderr, 'function =', f.f_code.co_name, ', line =', f.f_lineno


def read_dense_data(fp_data):
    X, Y = list(), list()
    max_len = 0

    for line in fp_data:
        line_arr = line.strip().split()
        Y.append(line_arr[0])
        x = list()
        for kv in line_arr[1 : ]:
            k, v = kv.split(':')
            k = int(k) - 1
            v = float(v)
            x.extend([0.0] * (k - len(x) + 1))
            if len(x) > max_len:
                max_len = len(x)
            x[k] = v
        X.append(x)

    for x in X:
        x.extend([0.0] * (max_len - len(x))) 

    return X, Y

def read_sparse_data(fp_data):
    X, Y = list(), list()

    for line in fp_data:
        line_arr = line.strip().split()
        Y.append(line_arr[0])
        x = list()
        for kv in line_arr[1 : ]:
            k, v = kv.split(':')
            x.append([k, float(v)])
        X.append(x)

    return X, Y

def read_sequence_data(fp_data):
    X, S = list(), list()

    for line in fp_data:
        x_list = list()
        s_list = list()
        
        for kv in line.strip().split():
            k,v = kv.split('/')
            x_list.append(k)
            s_list.append(v)

        X.append(x_list)
        S.append(s_list)

    return X, S

def plot_sequence_data(x, s, w = 128):

    x_out = list()
    t_out = list()
    s_out = list()

    for p, q in zip(x, s):

        L = max(len(p), len(q))
        
        d = L - len(q)
        m = d / 2
        n = d - m
        x_out.append(p + ' ' * (L - len(p)))
        s_out.append('-' * m + q + '-' * n)

        m = (L - 1) / 2
        n = L - 1 - m
        t_out.append(' ' * m + '|' + ' ' * n)
    
    s_line = '-'.join(s_out)
    t_line = ' '.join(t_out)
    x_line = ' '.join(x_out)

    if len(s_line) > w:
        for I in range(len(s_line) / w + 1):
            print >> sys.stderr, s_line[I * w : (I + 1) * w]
            print >> sys.stderr, t_line[I * w : (I + 1) * w]
            print >> sys.stderr, x_line[I * w : (I + 1) * w]
    else:
        print >> sys.stderr, s_line
        print >> sys.stderr, t_line
        print >> sys.stderr, x_line 


def map_label(Y):
    m = {k:v for v,k in enumerate(set(Y))}
    Y_map = [0] * len(Y)
    for i, k in enumerate(Y):
        Y_map[i] = m[k]
    return Y_map


if __name__ == '__main__':

    from random import randint
    X, S = read_sequence_data(open('data/pos_tagging.train'))

    i = randint(0, len(X) - 1)
    plot_sequence_data(X[i], S[i])

