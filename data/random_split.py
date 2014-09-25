# coding=utf-8
'''
author: vincent.ma.zefeng@gmail.com
'''

import sys
import random

if __name__ == '__main__':
    path_in, r = sys.argv[1 : 3]
    fp_in = open(path_in)
    fp_train_out = open(path_in + '.train', 'w')
    fp_test_out = open(path_in + '.test', 'w')
    r = float(r)

    for line in fp_in:
        if random.random() <= r:
            fp_train_out.write('%s' % line)
        else:
            fp_test_out.write('%s' % line)
    
    fp_in.close()
    fp_train_out.close()
    fp_test_out.close()
    
