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

def read_data(fp_data):
    return read_sparse_data(fp_data)

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

    L = list()
    I = list()

    for c, line in enumerate(fp_data):
        x, s = line.strip().split('/')
        L.append([x, s])
        if x == '###':
            I.append(c)

    for t in range(len(I) - 1):
        x_list = [x for x, s in L[I[t] + 1 : I[t + 1]]]
        s_list = [s for x, s in L[I[t] + 1 : I[t + 1]]]
        X.append(x_list)
        S.append(s_list)

    return X, S

def plot_sequence_data(x, s, line_width = 128):

    x_out = list()
    t_out = list()
    s_out = list()

    for p, q in zip(x, s):

        l = max(len(p), len(q))
        
        d = l - len(q)
        m = d / 2
        n = d - m
        x_out.append(p + ' ' * (l - len(p)))
        s_out.append('-' * m + q + '-' * n)

        m = (l - 1) / 2
        n = l - 1 - m
        t_out.append(' ' * m + '|' + ' ' * n)
    
    s_line = '-'.join(s_out)
    t_line = ' '.join(t_out)
    x_line = ' '.join(x_out)

    if len(s_line) > line_width:
        for i in range(len(s_line) / line_width + 1):
            print >> sys.stderr, s_line[i * line_width : (i + 1) * line_width]
            print >> sys.stderr, t_line[i * line_width : (i + 1) * line_width]
            print >> sys.stderr, x_line[i * line_width : (i + 1) * line_width]
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

    X, S = read_sequence_data(open('data/pos-tagging/entrain'))
    plot_sequence_data(X[1], S[1])

