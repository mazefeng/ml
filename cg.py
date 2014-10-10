
import sys
from math import isnan
from math import isinf
from math import sqrt
import numpy as np

from common import trace

# TODO
# 1. Early stop criteria, use fval to determine

def CG(f, w, max_fc, **argc):
    '''
    conjugate gradient routine for optimization
    '''

    INT = 0.1
    EXT = 3.0
    MAX = 20
    RATIO = 10
    SIG = 0.1
    RHO = SIG / 2
   
    red = 1.0
   
    # if length > 0: S = 'Linesearch'
    # else: S = 'Function evaluation'
    
    I = J = 1
    LS_failed = False
    f0, df0 = f(w, **argc)
    fval = [f0]
    print >> sys.stderr, 'Iter = %4.4i    Cost = %lf' % (I, f0)

    s = -df0
    d0 = float(-s.T * s)
    w3 = red / (1.0 - d0)
    
    while I < max_fc:
        w0, F0, dF0 = w, f0, df0
        M = min(MAX, max_fc - I)
        while True:
            w2, f2, d2, f3, df3 = 0, f0, d0, f0, df0
            success = False
            while not success and M > 0:
                try:
                    M -= 1
                    I += 1
                    f3, df3 = f(w + w3 * s, **argc)
                    fval.append(f3)
                    print >> sys.stderr, 'Iter = %4.4i    Cost = %lf' % (I, f3)
                    if isnan(f3) or isinf(f3) or np.any(np.isnan(df3) + np.isinf(df3)):
                        raise NameError, ('error')
                    success = True
                except Exception, e:
                    print >> sys.stderr, 'Exception = %s'  % e
                    trace()
                    w3 = (w2 + w3) / 2.0
            if f3 < F0:
                w0, F0, dF0 = w + w3 * s, f3, df3
            d3 = float(df3.T * s)
            if d3 > SIG * d0 or f3 > f0 + w3 * RHO * d0 or M == 0:
                break
            w1, f1, d1 = w2, f2, d2
            w2, f2, d2 = w3, f3, d3
            A = 6 * (f1 - f2) + 3 * (d2 + d1) * (w2 - w1)
            B = 3 * (f2 - f1) - (2 * d1 + d2) * (w2 - w1)
            # if not isinstance(w3, float) or isnan(w3) or isinf(w3) or w3 < 0: w3 = w2 * EXT
            try:
                w3 = w1 - d1 * (w2 - w1) ** 2 / (B + sqrt(B * B - A * d1 * (w2 - w1)))
                # add sth
            except Exception, e:
                print >> sys.stderr, 'Exception = %s' % e
                trace()
                w3 = w2 * EXT
                continue
            if w3 < 0:
                w3 = w2 * EXT
            elif w3 > w2 * EXT:
                w3 = w2 * EXT
            elif w3 < w2 + INT * (w2 - w1):
                w3 = w2 + INT * (w2 - w1)
      
        while (abs(d3) > -SIG * d0 or f3 > f0 + w3 * RHO * d0) and M > 0:
            if d3 > 0 or f3 > f0 + w3 * RHO * d0:
                w4, f4, d4 = w3, f3, d3
            else:
                w2, f2, d2 = w3, f3, d3
            try:
                if f4 > f0:
                    w3 = w2 - (0.5 * d2 * (w4 - w2) ** 2) / (f4 - f2 - d2 * (w4 - w2))
                else:
                    A = 6 * (f2 - f4) / (w4 - w2) + 3 * (d4 + d2)
                    B = 3 * (f4 - f2) - (2 * d2 + d4) * (w4 - w2)
                    w3 = w2 + (sqrt(B * B - A * d2 * (w4 - w2) ** 2) - B) / A
            except Exception, e:
                    print >> sys.stderr, 'Exception = %s' % e
                    trace()
                    w3 = float('NaN')
            if isnan(w3) or isinf(w3):
                w3 = (w2 + w4) / 2
            w3 = max(min(w3, w4 - INT * (w4 - w2)), w2 + INT * (w4 - w2))
            f3, df3 = f(w + w3 * s, **argc)
            fval.append(f3)
            if f3 < F0:
                w0, F0, dF0 = w + w3 * s, f3, df3
            M -= 1
            I += 1
            print >> sys.stderr, 'Iter = %4.4i    Cost = %lf' % (I, f3)
            d3 = float(df3.T * s)
      
        if abs(d3) < -SIG * d0 and f3 < f0 + w3 * RHO * d0:
            w, f0 = w + w3 * s, f3
            print >> sys.stderr, 'Line_search = %4.4i    Cost = %lf' % (J, f0)
            J += 1
            s = float((df3.T * df3 - df0.T * df3) / (df0.T * df0)) * s - df3
            df0 = df3
            d3, d0 = d0, float(df0.T * s)
            if d0 > 0:
                s, d0 = -df0, float(-s.T * s)
            w3 = w3 * min(RATIO, d3 / (d0 - sys.float_info.min))
            LS_failed = False
        else:
            w, f0, df0 = w0, F0, dF0
            if LS_failed or I > max_fc:
                break
            s, d0 = -df0, float(-s.T * s)
            w3 = 1.0 / (1.0 - d0)
            LS_failed = True
    print >> sys.stderr, ''
    # print >> sys.stderr, fval
    # print >> sys.stderr, ''
    return w

def f(x):
    return float(x * x - 2 * x + 1), 2 * x - 2 
    
if __name__ == '__main__':

    x = 100.0 * np.matrix(np.ones([1, 1]))
    x_opt = CG(f, x, 40)
    print x_opt
