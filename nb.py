# coding=utf-8

import sys
from math import log

'''
Event model:
0 : Multi-variate Bernoulli event model
1 : Multinomial event model
'''

Bernoulli = 0
Multinomial = 1

class NaiveBayes:
    def __init__(self):
        self.priors = dict()
        self.prob = dict()
    
    def train(self, X, Y, event_model = Bernoulli, smooth = 0.5):

        if event_model != Bernoulli and event_model != Multinomial:
            print >> sys.stderr, 'event_model parameter error'
            return

        self.priors.clear()
        self.prob.clear()

        for y in Y:
            if not y in self.priors:
                self.priors[y] = 0
            self.priors[y] += 1

        D = set()
        for x, y in zip(X, Y):
            if not y in self.prob:
                self.prob[y] = dict()
            for w, c in x:
                D.add(w)
                if not w in self.prob[y]:
                    self.prob[y][w] = 0
                
                if event_model == Bernoulli:
                    self.prob[y][w] += 1
                if event_model == Multinomial:
                    self.prob[y][w] += c

        # laplace smooth
        for y in self.prob:
            for w in D:
                if not w in self.prob[y]:
                    self.prob[y][w] = 0
                self.prob[y][w] += smooth

        if event_model == Bernoulli:
            for y in self.prob:
                for w in self.prob[y]:
                    self.prob[y][w] /= 1.0 * self.priors[y]
        
        if event_model == Multinomial:
            for y in self.prob:
                n = sum(self.prob[y].values())
                for w in self.prob[y]:
                    self.prob[y][w] /= 1.0 * n
            
        n = len(Y)
        for y in self.priors:
            self.priors[y] /= 1.0 * n

        print >> sys.stderr, '-' * 128
        print >> sys.stderr, 'Top 5 strong features for each class:'
        print >> sys.stderr, '-' * 128
        
        for y in self.prob:
            ftrs = sorted(self.prob[y].items(), key = lambda x : x[1], reverse = True)
            print >> sys.stderr, y + '\t' + '\t'.join([k + ':' + str(round(v, 5)) for k,v in ftrs[0 : 5]])

        print >> sys.stderr, '-' * 128

    def test(self, X, Y):
        
        correct = 0
        
        for i, x in enumerate(X):
            d = dict()
            for y in self.priors:
                d[y] = log(self.priors[y])
                for w, c in x:
                    if not w in self.prob[y]:
                        continue
                    d[y] += log(self.prob[y][w])
            prob_list = sorted(d.items(), key = lambda x : x[1], reverse = True)
            
            y_pred = prob_list[0][0]     
    
            if y_pred == Y[i]:
                correct += 1
        
        return 1.0 * correct / len(Y)


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


if __name__ == '__main__':

    train_path = 'data/20_newsgroups.data.train'
    test_path = 'data/20_newsgroups.data.test'

    X_train, Y_train = read_data(open(train_path))
    X_test, Y_test = read_data(open(test_path))

    clf = NaiveBayes()
    clf.train(X_train, Y_train, Bernoulli)
    print >> sys.stderr, 'Accuracy for Multi-variate Bernoulli event model : %f%%' % (100 * clf.test(X_test, Y_test))

    clf.train(X_train, Y_train, Multinomial)
    print >> sys.stderr, 'Accuracy for Multinomial event model : %f%%' % (100 * clf.test(X_test, Y_test))


