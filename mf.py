import sys
import json
import time
import math
import random
import pickle
import threading  
import numpy as np
import matplotlib
matplotlib.use('wx')
import matplotlib.pylab as plt

random.seed(1024)

class SparseMatrix:

    kv_dict = None
    rows = None
    cols = None
    _sum = 0.0
    mean = 0.0
    
    def __init__(self, rows = None, cols = None):
        self.kv_dict = dict()
        
        if rows == None:
            self.rows = dict()
        else:
            self.rows = dict(rows)
        
        if cols == None:
            self.cols = dict()
        else:
            self.cols = dict(cols)
        
        self._sum = 0.0
        self.mean = 0.0

    def __repr__(self):
        ret_str = ''
        ret_str += '[INFO]: kv_dict size: %d mb\n' % (sys.getsizeof(self.kv_dict) / (1024 * 1024))
        ret_str += '[INFO]: rows size: %d\n' % len(self.rows)
        ret_str += '[INFO]: cols size: %d\n' % len(self.cols)
        ret_str += '[INFO]: #sparse kv_dict: %d\n' % len(self.kv_dict)
        ret_str += '[INFO]: mean: %f\n' % self.mean
        ret_str += '[INFO]: sum: %f\n' % self._sum
        return ret_str
        
    def __len__(self):
        return len(self.kv_dict)

    def __getitem__(self, index):
        r_index, c_index = index
        
        if not r_index in rows or not c_index in cols:
            return 0.0
        
        if not (rows[r_index], cols[c_index]) in kv_dict:
            return 0.0

        return 1.0 * self.kv_dict[rows[r_index], cols[c_index]]
    
    def __setitem__(self, index, value):
        r_index, c_index = index
        
        if not r_index in self.rows:
            self.rows[r_index] = len(self.rows)
        row = self.rows[r_index]

        if not c_index in self.cols:
            self.cols[c_index] = len(self.cols)
        col = self.cols[c_index]

        if not (row, col) in self.kv_dict:
            self._sum += value
        else:
            self._sum += value - self.kv_dict[row, col]

        self.kv_dict[row, col] = value

        self.mean = 1.0 * self._sum / len(self.kv_dict) if len(self.kv_dict) != 0 else 0.0
    

class ParallelSGD(threading.Thread):

    def __init__(self, name, model): 
        threading.Thread.__init__(self)  
        self.name = name
        self.model = model
        # self.v = np.matrix(np.zeros([(model.num_factor + 1) * 2, 1]))

    def run(self):

        while True:

            self.model.thread_lock.acquire()

            if self.model.current_sample >= len(self.model.L_train):
                self.model.thread_lock.release()
                break
            (u, i), r = self.model.L_train[self.model.current_sample]
            self.model.current_sample += 1

            pu = self.model.P[u]
            qi = self.model.Q[i]
            bu = self.model.bu[u]
            bi = self.model.bi[i]

            self.model.thread_lock.release()
        
            r_pred = self.model.mu + bu + bi + float(pu * qi.T)
            err = r - r_pred
        
            pu_update = pu + self.model.alpha * (err * qi - self.model._lambda * pu)
            qi += self.model.alpha * (err * pu - self.model._lambda * qi)

            bu += self.model.alpha * (err - self.model._lambda * bu)
            bi += self.model.alpha * (err - self.model._lambda * bi)

            self.model.thread_lock.acquire()
 
            self.model.P[u] = pu_update
            self.model.Q[i] = qi
            self.model.bu[u] = bu
            self.model.bi[i] = bi

            self.model.sqr_err += err ** 2
        
            self.model.thread_lock.release()

class MF:

    # num_factor = 25, _lambda = 0.005, max_iter = 10, alpha = 0.01, num_thread = 40, validate = 0
    def __init__(self, num_factor = 25, _lambda = 0.005, max_iter = 20, alpha = 0.01, num_thread = 10, validate = 0):

        self.P = None
        self.Q = None
        self.bu = None
        self.bi = None
        self.mu = 0.0

        self.rows = None   
        self.cols = None
 
        self.num_factor = num_factor
        self._lambda = _lambda
        self.max_iter = max_iter
        self.alpha = alpha
        self.num_thread = num_thread
        self.validate = validate    

        self.L_train = None
        self.L_validate = None
        self.sqr_err = 0.0
        self.current_sample = 0
        self.thread_lock = threading.Lock()
 
    def train(self, ratings, model_path):
 
        self.mu = ratings.mean
        self.P = 0.001 * np.matrix(np.random.randn(len(ratings.rows), self.num_factor))
        self.bu = 0.001 * np.matrix(np.random.randn(len(ratings.rows), 1))
        self.Q = 0.001 * np.matrix(np.random.randn(len(ratings.cols), self.num_factor))
        self.bi = 0.001 * np.matrix(np.random.randn(len(ratings.cols), 1))
        
        self.rows = dict(ratings.rows)
        self.cols = dict(ratings.cols)       

        if self.validate > 0:
            T = ratings.kv_dict.items()
            random.shuffle(T)
            k = len(T) / self.validate
            self.L_validate = T[0 : k]
            self.L_train = T[k :]
        else:
            self.L_train = ratings.kv_dict.items()

        rmse_train = [0.0] * self.max_iter
        rmse_validate = [0.0] * self.max_iter       
 
        for s in range(self.max_iter):

            random.shuffle(self.L_train)
            self.current_sample = 0
            self.sqr_err = 0.0

            self.threads = [ParallelSGD('Thread_%d' % n, self) for n in range(self.num_thread)]
            
            start = time.time()
            for t in self.threads:
                t.start()
                t.join()
            terminal = time.time()

            duration = terminal - start

            rmse_train[s] = math.sqrt(self.sqr_err / len(ratings.kv_dict))
    
            if self.validate > 0:
                m = SparseMatrix()
                m.kv_dict = {k : v for (k, v) in self.L_validate}
                rmse_validate[s] = float(self.test(m))
            
            sys.stderr.write('Iter: %4.4i' % (s + 1))
            sys.stderr.write('\t[Train RMSE] = %f' % rmse_train[s])
            if self.validate > 0:
                sys.stderr.write('\t[Validate RMSE] = %f' % rmse_validate[s])
            sys.stderr.write('\t[Duration] = %f' % duration)
            sys.stderr.write('\t[Samples] = %d\n' % len(self.L_train))

            self.dump_model(model_path + '/' + 'model_%4.4i' % (s + 1))
            self.dump_raw_model(model_path + '/' + 'model_%4.4i.raw_model' % (s + 1))

        plt.subplot(111)
        plt.plot(range(self.max_iter), rmse_train, '-og')
        plt.plot(range(self.max_iter), rmse_validate, '-xb')
        plt.show()
            
    def test(self, ratings):

        U = np.matrix(np.zeros([len(ratings), self.num_factor]))
        V = np.matrix(np.zeros([len(ratings), self.num_factor]))
        b = np.matrix(np.zeros([len(ratings), 1]))

        u_kv = dict()
        v_kv = dict()

        for s, (u, i) in enumerate(ratings.kv_dict):
            if u < len(self.P): u_kv[s] = u
            if i < len(self.Q): v_kv[s] = i

        U[u_kv.keys()] = self.P[u_kv.values()]
        V[v_kv.keys()] = self.Q[v_kv.values()]  
      
        b[u_kv.keys()] += self.bu[u_kv.values()]
        b[v_kv.keys()] += self.bi[v_kv.values()]

        R = np.matrix(ratings.kv_dict.values()).T
        err = R - (np.multiply(U, V).sum(1) + b + self.mu)
        rmse = math.sqrt(err.T * err / len(ratings))

        return rmse

    def dump_raw_model(self, path):
        fp_raw_model = open(path, 'w')

        output = dict()
        
        for k, v in self.rows.items():
            output['type'] = 'USER'
            output['id'] = k
            output['latent_factor'] = map(lambda x : round(x, 6), self.P[v].tolist()[0])
            output['bias'] = round(float(self.bu[v]), 6)
            print >> fp_raw_model, json.dumps(output)

        for k, v in self.cols.items():
            output['type'] = 'ITEM'
            output['id'] = k
            output['latent_factor'] = map(lambda x : round(x, 6), self.Q[v].tolist()[0])
            output['bias'] = round(float(self.bi[v]), 6)
            print >> fp_raw_model, json.dumps(output)
            
        fp_raw_model.close()
    

    def dump_model(self, path):

        fp_meta_data = open(path + '.meta', 'w')
        print >> fp_meta_data, self.mu
        print >> fp_meta_data, self.P.shape[0], self.P.shape[1], self.P.dtype
        print >> fp_meta_data, self.bu.shape[0], self.bu.shape[1], self.bu.dtype
        print >> fp_meta_data, self.Q.shape[0], self.Q.shape[1], self.Q.dtype
        print >> fp_meta_data, self.bi.shape[0], self.bi.shape[1], self.bi.dtype
        print >> fp_meta_data, json.dumps(self.rows)
        print >> fp_meta_data, json.dumps(self.cols)
        fp_meta_data.close()
        
        self.P.tofile(path + '.uf')
        self.bu.tofile(path + '.ub')
        self.Q.tofile(path + '.if')
        self.bi.tofile(path + '.ib')

    def load_model(self, path):
        fp_meta_data = open(path + '.meta')
        self.mu = float(fp_meta_data.readline())

        r, c, dtype = fp_meta_data.readline().split()
        self.P = np.matrix(np.fromfile(path + '.uf', dtype = np.dtype(dtype)))
        self.P.shape = (int(r), int(c))

        r, c, dtype = fp_meta_data.readline().split()
        self.bu = np.matrix(np.fromfile(path + '.ub', dtype = np.dtype(dtype)))
        self.bu.shape = (int(r), int(c))

        r, c, dtype = fp_meta_data.readline().split()
        self.Q = np.matrix(np.fromfile(path + '.if', dtype = np.dtype(dtype)))
        self.Q.shape = (int(r), int(c))

        r, c, dtype = fp_meta_data.readline().split()
        self.bi = np.matrix(np.fromfile(path + '.ib', dtype = np.dtype(dtype)))
        self.bi.shape = (int(r), int(c))

        self.rows = json.loads(fp_meta_data.readline())
        self.cols = json.loads(fp_meta_data.readline())

        fp_meta_data.close()
           
def read_sparse_matrix(fp_data, rows = None, cols = None):
    m = SparseMatrix(rows, cols)
    for line in fp_data:
        user_id, item_id, rating, timestamp = line.strip().split('::')
        user_id = int(user_id)
        item_id = int(item_id)
        rating = float(rating)
        m[user_id, item_id] = rating
    return m
 
if __name__ == '__main__':
 
    train_data = 'data/movielens.1m.train'
    # train_data = '/home/mazefeng/movielens/ra.train'
    test_data = 'data/movielens.1m.test'
    # test_data = '/home/mazefeng/movielens/ra.test'

    train_ratings = read_sparse_matrix(open(train_data))
    print >> sys.stderr, 'read training sparse matrix done.'
    test_ratings = read_sparse_matrix(open(test_data), train_ratings.rows, train_ratings.cols)
    print >> sys.stderr, 'read test sparse matrix done.'

    print >> sys.stderr, train_ratings
    print >> sys.stderr, test_ratings

    # pickle.dump(train_ratings, open('train_ratings.pkl', 'w'))
    # pickle.dump(test_ratings, open('test_ratings.pkl', 'w'))
    # ratings = pickle.load(open('ratings.pkl'))

    mf = MF()
    mf.train(train_ratings, 'mf_model')

    rmse_train = mf.test(train_ratings)
    rmse_test = mf.test(test_ratings)
    
    print >> sys.stderr, 'Training RMSE for MovieLens 1m dataset: %lf' % rmse_train
    print >> sys.stderr, 'Test RMSE for MovieLens 1m dataset: %lf' % rmse_test

