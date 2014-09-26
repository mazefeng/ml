# coding=utf-8

import sys
import json
from math import log
from collections import defaultdict
from common import read_data

def entropy(p, q):
    if p == 0 or q == 0:
        return 0.0
    s = 1.0 * (p + q)
    return - (p/s) * log(p/s, 2) - (q/s) * log(q/s, 2)

def chi_square_test(p, q, thresold):
    n = p[0] + p[1] + q[0] + q[1]
    if n < 40:
        return 0
    
    s = [0.0, 0.0, 0.0, 0.0]
    s[0] = 1.0 * (p[0] + p[1]) * (p[0] + q[0]) / n
    s[1] = 1.0 * (p[0] + p[1]) * (p[1] + q[1]) / n
    s[2] = 1.0 * (q[0] + q[1]) * (p[0] + q[0]) / n
    s[3] = 1.0 * (q[0] + q[1]) * (p[1] + q[1]) / n
    for c in s: 
        if c < 5.0: return 0
    
    chi_square_score = 1.0 * (p[0] * q[1] - p[1] * q[0]) ** 2 * n / ((p[0] + p[1]) * (q[0] + q[1]) * (p[0] + q[0]) * (p[1] + q[1]))
    if chi_square_score > thresold:
        return 1
    return -1

class ID3:
        
    def __init__(self):
        
        self.model = dict()
        self.model['ROOT'] = None

        self.chi_square_thresold = None
        self.max_depth = None
        self.min_sample = None

        self.all_label = None
        self.all_feature = None
        
        self.rule_set = None

    # chi_square_thresold = 3.841459
    # max_depth = 20
    # min_sample = 10
    def train(self, X, Y, chi_square_thresold = 3.841459, max_depth = 20, min_sample = 10):
        
        self.chi_square_thresold = chi_square_thresold
        self.max_depth = max_depth
        self.min_sample = min_sample
       
        self.all_label = set(Y)
        
        self.all_feature = set()
        for x in X:
            self.all_feature.update([k for k,v in x])        
    
        I = range(0, len(Y))
        F = set()
        
        self.model['ROOT'] = self.build_dt(X, Y, I, F, 0)

    def build_dt(self, X, Y, I, F, depth):
        
        node = dict()

        S = {l : 0 for l in self.all_label}
        L = defaultdict(lambda : {l : [] for l in self.all_label})
        R = defaultdict(lambda : {l : [] for l in self.all_label}) 

        for i in I:
            x, y = X[i], Y[i]
            S[y] += 1
            
            d = {k for k,v in x}
            for f in self.all_feature - F:
                if f in d:
                    L[f][y].append(i)
                else:
                    R[f][y].append(i)

        node['TYPE'] = 'LEAF'
        node['SIZE'] = len(I)
        node['DEPTH'] = depth
        node['LABEL'] = max(S, key = lambda x : S[x])

        if depth >= self.max_depth or len(I) <= self.min_sample:
            return node
        
        if len(S) == 1 or len(F) == len(self.all_feature):
            return node

        max_info_gain = 0.0
        f_selected = None
        E = entropy(S.values()[0], S.values()[1])

        for f in self.all_feature - F:
            m = L[f].values()
            n = R[f].values()

            p = [len(m[0]), len(m[1])]
            q = [len(n[0]), len(n[1])]

            w0 = 1.0 * sum(p) / (sum(p) + sum(q))
            w1 = 1.0 * sum(q) / (sum(p) + sum(q))

            E0 = entropy(len(m[0]), len(m[1]))
            E1 = entropy(len(n[0]), len(n[1]))
        
            info_gain = E - w0 * E0 - w1 * E1
            
            if info_gain >= max_info_gain:
                max_info_gain = info_gain
                f_selected = f
        
        m = L[f_selected].values()
        n = R[f_selected].values()
        p = [len(m[0]), len(m[1])]
        q = [len(n[0]), len(n[1])]
        
        if sum(p) == 0 or sum(q) == 0:
            return node
        
        if chi_square_test(p, q, self.chi_square_thresold) < 0:
            return node

        node['TYPE'] = 'INTERNAL'
        node['SPLIT'] = f_selected
        node['CHILD'] = dict()
        
        node['CHILD']['LEFT'] = self.build_dt(X, Y, m[0] + m[1], F | {f_selected}, depth + 1)
        node['CHILD']['RIGHT'] = self.build_dt(X, Y, n[0] + n[1], F | {f_selected}, depth + 1)
        
        return node
       
    def dump_model(self, fp_model, fp_rule_set = None):
        print >> fp_model, json.dumps(self.model)
        if fp_rule_set != None:
            self.dump_rule_set(self.model['ROOT'])
            for rule in self.rule_set:
                print >> fp_rule_set, rule 
 
    def dump_rule_set(self, node):
        if node == self.model['ROOT']:
            self.rule_set = []
        
        if node['TYPE'] == 'INTERNAL':

            pre_left = ('  ' * node['DEPTH']) + 'IF (%s) THEN ' % node['SPLIT']
            self.rule_set.append(pre_left)
            self.dump_rule_set(node['CHILD']['LEFT'])
            
            pre_right = ('  ' * node['DEPTH']) + 'ELSE '
            self.rule_set.append(pre_right)
            self.dump_rule_set(node['CHILD']['RIGHT'])

        else:
            rule = ('  ' * node['DEPTH']) + 'PREDICT LABEL IS %s [branch-size : %d]' % (node['LABEL'], node['SIZE'])
            self.rule_set.append(rule)
  
    def test(self, X, Y):

        correct = 0
        for x, y in zip(X, Y):
            y_pred = None
            d = {k for k,v in x}       
            node = self.model['ROOT']
            while node['TYPE'] != 'LEAF':
                if not 'CHILD' in node:
                    y_pred = nodel['LABEL']
                    break
                if node['SPLIT'] in d:
                    node = node['CHILD']['LEFT']
                else:
                    node = node['CHILD']['RIGHT']

            if y_pred == None:
                y_pred = node['LABEL']
            if y_pred == y:
                correct += 1

        return 1.0 * correct / len(Y)

if __name__ == '__main__':
   
    train_path = 'data/2_newsgroups.train'
    test_path = 'data/2_newsgroups.test'

    X_train, Y_train = read_data(open(train_path))
    X_test, Y_test = read_data(open(test_path))

    clf = ID3()
    clf.train(X_train, Y_train)
    accuracy = clf.test(X_test, Y_test)

    print >> sys.stderr, 'Accuracy for ID3 : %f%%' % (100 * accuracy)
   
    clf.dump_model(open('data/dt.model', 'w'), open('data/dt.rule_set', 'w'))
 
