# coding=utf-8

import sys

def read_data(fp_data):
    X, Y = list(), list()

    for line in fp_data:
        line_arr = line.strip().split()
        Y.append(line_arr[0])
        x = list()
        for kv in line_arr[1 : ]:
            w, c = kv.split(':')
            x.append([w, float(c)])
        X.append(x)

    return X, Y


