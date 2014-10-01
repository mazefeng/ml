# coding=utf-8

import sys
import json
from math import log
from collections import defaultdict as dict

from common import read_sequence_data
from common import plot_sequence_data

def argmax(d):
    k = max(d, key = lambda x: d[x])
    return k, d[k]

class HiddenMarkovModel:
    def __init__(self):

        self.prob_s3 = dict(float)
        self.prob_xs = dict(float)
        self.D = dict()
        self.V = dict(set)
        
    def train(self, X, S, smooth = 0.5):
        x_count  = dict(int)
        xs_count = dict(int)
        
        unigram = dict(int)
        bigram  = dict(int)
        trigram = dict(int)

        for x_list, s_list in zip(X, S):
            if len(x_list) == 0 or len(x_list) != len(s_list):
                print >> sys.stderr, 'ERROR: x_list =', x_list, ', s_list =', s_list
                continue
            bigram[('*', '*')] += 1
            bigram[('*', s_list[0])] += 1
            trigram[('*', '*', s_list[0])] += 1       
            if len(s_list) > 1:
                trigram[('*', s_list[0], s_list[1])] += 1
            
            for p in range(len(x_list)):
                x, s = x_list[p], s_list[p]
                
                self.V[x].add(s)
                x_count[x] += 1
                xs_count[(x, s)] += 1
                unigram[s] += 1

                if p < 1: continue
                s_i = s_list[p - 1]
                bigram[(s_i, s)] += 1

                if p < 2: continue
                s_i = s_list[p - 2]
                s_j = s_list[p - 1]
                trigram[(s_i, s_j, s)] += 1

            bigram[(s_list[-1], 'STOP')] += 1
        
            if len(s_list) < 2: continue
            trigram[(s_list[-2], s_list[-1], 'STOP')] += 1

        self.D = unigram

        # smoothing
        for s in self.D:
            trigram[('*', '*', s)] += smooth
        bigram[('*', '*')] += smooth * len(self.D)
        
        for s_i in self.D:
            for s_j in self.D:
                trigram[('*', s_i, s_j)] += smooth
            bigram[('*', s_i)] += smooth * len(self.D)

        for s_i in self.D:
            for s_j in self.D:
                trigram[(s_i, s_j, 'STOP')] += smooth
                bigram[(s_i, s_j)] += smooth

        for s_i in self.D:
            for s_j in self.D:
                for s_k in self.D:
                    trigram[(s_i, s_j, s_k)] += smooth
                bigram[(s_i, s_j)] += smooth * len(self.D)

        for x, s in xs_count:
            self.prob_xs[(x, s)] = 1.0 * xs_count[(x, s)] / unigram[s]

        for s_i, s_j, s_k in trigram:
            self.prob_s3[(s_i, s_j, s_k)] = 1.0 * trigram[(s_i, s_j, s_k)] / bigram[(s_i, s_j)]


    def viterbi(self, x_list):

        n = len(x_list)
        x_list = [''] + x_list        
        s_list = [''] * (n + 1)

        T = dict()        
        T[-1] = T[0] = ['*']
        for i in range(1, n + 1):
            if x_list[i] in self.V:
                T[i] = self.V[x_list[i]]
            else:
                T[i] = set(self.D)

        pi = dict(float)
        bp = dict(str)
        
        pi[(0, '*', '*')] = 1.0
        for k in range(1, n + 1):
            for u in T[k - 1]:
                for v in T[k]:
                    K = dict()
                    for w in T[k - 2]:
                        if (x_list[k], v) in self.prob_xs:
                            p = self.prob_xs[(x_list[k], v)]
                        else:
                            p = 1.0
                        K[w] = pi[(k - 1, w, u)] * self.prob_s3[(w, u, v)] * p

                    bp[(k, u, v)], pi[(k, u, v)] = argmax(K)

        K = dict()
        for u in T[n - 1]:
            for v in T[n]:
                K[(u, v)] = pi[(n, u, v)] * self.prob_s3[(u, v, 'STOP')]
        
        (s_list[n - 1], s_list[n]), val = argmax(K)

        for k in range(n - 2, 0, -1):    
            s_list[k] = bp[(k + 2, s_list[k + 1], s_list[k + 2])]
            
        return s_list[1 : ]
  
    
    def baseline(self, X, S):

        m = n = c = 0

        k, v = argmax(self.D)

        for x_list, s_list in zip(X, S):
            s_list_pred = [''] * len(s_list)

            for i in range(len(x_list)):
                x = x_list[i]
                if not x in self.V:
                    s_list_pred[i] = k  
                else:
                    d = {(x, s) : self.prob_xs[(x, s)] for s in self.V[x] if (x, s) in self.prob_xs}
                    (x_max, s_max), v_max = argmax(d)
                    s_list_pred[i] = s_max

            for p, q in zip(s_list, s_list_pred):
                if p == q: m += 1
        
            n += len(s_list)
            c += 1

            # print >> sys.stderr, 'Ground truth:'
            # plot_sequence_data(x_list, s_list)            
            # print >> sys.stderr, 'Tagging result:'
            # plot_sequence_data(x_list, s_list_pred)            

        print >> sys.stderr, 'Baseline accuracy : %lf%% (%d/%d)' % (100.0 * m / n, m, n)

        return 1.0 * m / n


 
    def test(self, X, S):

        m = n = c = 0
        for x_list, s_list in zip(X, S):
            s_list_pred = self.viterbi(x_list)
            assert(len(s_list) == len(s_list_pred))
        
            for p, q in zip(s_list, s_list_pred):
                if p == q: m += 1
        
            n += len(s_list)
            c += 1

            # print >> sys.stderr, 'Ground truth:'
            # plot_sequence_data(x_list, s_list)            
            # print >> sys.stderr, 'Tagging result:'
            # plot_sequence_data(x_list, s_list_pred)            
        
        print >> sys.stderr, 'Accuracy for HMM POS-tagger : %lf%% (%d/%d)' % (100.0 * m / n, m, n)

        return 1.0 * m / n


if __name__ == '__main__':

    train_path = 'data/pos-tagging/entrain'
    test_path = 'data/pos-tagging/entest'
    
    # train_path = 'data/pos-tagging/ictrain'
    # test_path = 'data/pos-tagging/ictest'

    X_train, S_train = read_sequence_data(open(train_path))
    X_test, S_test = read_sequence_data(open(test_path))

    tagger = HiddenMarkovModel()

    tagger.train(X_train, S_train)
    baseline_acc = tagger.baseline(X_test, S_test)
    viterbi_acc = tagger.test(X_test, S_test)

    # print >> sys.stderr, 'Baseline accuracy %lf%%' % (100.0 * baseline_acc)
    # print >> sys.stderr, 'Accuracy for HMM POS-tagger %lf%%' % (100.0 * viterbi_acc)
