import sys
import random
import math
import pickle
import numpy as np

random.seed(1024)
np.seterr(all = 'raise')

class SparseMatrix:

    kv_dict = None
    rows = None
    cols = None
    sum = 0.0
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
        
        self.sum = 0.0
        self.mean = 0.0

    def __repr__(self):
        ret_str = ''
        ret_str += '[INFO]: kv_dict size: %d mb\n' % (sys.getsizeof(self.kv_dict) / (1024 * 1024))
        ret_str += '[INFO]: rows size: %d\n' % len(self.rows)
        ret_str += '[INFO]: cols size: %d\n' % len(self.cols)
        ret_str += '[INFO]: #sparse kv_dict: %d\n' % len(self.kv_dict)
        ret_str += '[INFO]: mean: %f\n' % self.mean
        ret_str += '[INFO]: sum: %f\n' % self.sum
        return ret_str
        
    def __len__(self):
        return len(self.kv_dict)

    def __getitem__(self, index):
        r_index, c_index = index
        if not r_index in rows or not c_index in cols: return 0.0
        if not (rows[r_index], cols[c_index]) in kv_dict: return 0.0
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
            self.sum = self.sum + value
        else:
            self.sum = self.sum - self.kv_dict[row, col] + value

        self.kv_dict[row, col] = value
        self.mean = 1.0 * self.sum / len(self.kv_dict) if len(self.kv_dict) != 0 else 0.0
    
    
class MF:

    user_factor = None
    item_factor = None
    user_bias = None
    item_bias = None
    mu = 0.0

    factor = 25
    max_iter = 5
    batch_size = 100000
    rmse_thresold = 1e-6
    alpha = 0.1
    L1 = 0.0
    L2 = 0.001
    
    def train(self, ratings):
        
        self.mu = ratings.mean

        self.user_factor = 0.001 * np.matrix(np.random.randn(len(ratings.rows), self.factor))
        self.user_bias = 0.001 * np.matrix(np.random.randn(len(ratings.rows), 1))
        
        self.item_factor = 0.001 * np.matrix(np.random.randn(len(ratings.cols), self.factor))
        self.item_bias = 0.001 * np.matrix(np.random.randn(len(ratings.cols), 1))

        L = ratings.kv_dict.items()
        for i in range(self.max_iter):

            lamb = self.alpha * (1.0 / math.sqrt(i + 1))
           
            random.shuffle(L)
            R = np.matrix([I[1] for I in L]).T 
            sqr_err = list()
            
            for s in range(0, len(L), self.batch_size):
                mini_batch = L[s : s + self.batch_size]
                r = R[s : s + self.batch_size]

                uid = [I[0][0] for I in mini_batch]
                iid = [I[0][1] for I in mini_batch]
 
                base_line = self.mu + self.user_bias[uid] + self.item_bias[iid]
                r_pred = base_line + np.sum(np.multiply(self.user_factor[uid], self.item_factor[iid]), 1)

                err = r - r_pred
                
                p = self.user_factor[uid]
                q = self.item_factor[iid]
                bu = self.user_bias[uid]
                bi = self.item_bias[iid]
                
                p_grad = np.multiply(err, q) - self.L2 * p
                p = p + lamb * p_grad
                
                q_grad = np.multiply(err, p) - self.L2 * q
                q = q + lamb * q_grad
                
                bu_grad = err - self.L2 * bu
                bi_grad = err - self.L2 * bi
                
                bu = bu + lamb * bu_grad
                bi = bi + lamb * bi_grad

                self.user_factor[uid] = p
                self.item_factor[iid] = q
                self.user_bias[uid] = bu
                self.item_bias[iid] = bi
                
                sqr_err.append(float(err.T * err) / self.batch_size)

            rmse = math.sqrt(np.mean(np.matrix(sqr_err)))
            sys.stderr.write('Iter: %4.4i    RMSE: %f\n' % (i + 1, rmse))

            if i > 0 and i % 10 ==0: self.dump_model('model_%d' % (i))
            
            if rmse < self.rmse_thresold: break
    
    def test(self, ratings):
        L = ratings.kv_dict.items()
        R = np.matrix([I[1] for I in L]).T 
        sqr_err = list()
       
        au = len(ratings.rows) - len(self.user_factor)
        self.user_factor = np.row_stack([self.user_factor, np.matrix(np.zeros([au, self.factor]))])
        self.user_bias = np.row_stack([self.user_bias, np.matrix(np.zeros([au, 1]))])
 
        ai = len(ratings.cols) - len(self.item_factor)
        self.item_factor = np.row_stack([self.item_factor, np.matrix(np.zeros([ai, self.factor]))])
        self.item_bias = np.row_stack([self.item_bias, np.matrix(np.zeros([ai, 1]))])
 
        for s in range(0, len(L), self.batch_size):
            mini_batch = L[s : s + self.batch_size]
            r = R[s : s + self.batch_size]
            
            uid = [I[0][0] for I in mini_batch]
            iid = [I[0][1] for I in mini_batch]
 
            base_line = self.mu + self.user_bias[uid] + self.item_bias[iid]
            r_pred = base_line + np.sum(np.multiply(self.user_factor[uid], self.item_factor[iid]), 1)

            err = r - r_pred
            sqr_err.append(float(err.T * err) / self.batch_size)
        
        rmse = math.sqrt(np.mean(np.matrix(sqr_err)))
        # sys.stderr.write('RMSE: %f\n' % (rmse))
        
        return rmse
    

    def dump_model(self, path):

        fp_meta_data = open(path + '.meta', 'w')
        print >> fp_meta_data, self.mu
        print >> fp_meta_data, self.user_factor.shape[0], self.user_factor.shape[1], self.user_factor.dtype
        print >> fp_meta_data, self.user_bias.shape[0], self.user_bias.shape[1], self.user_bias.dtype
        print >> fp_meta_data, self.item_factor.shape[0], self.item_factor.shape[1], self.item_factor.dtype
        print >> fp_meta_data, self.item_bias.shape[0], self.item_bias.shape[1], self.item_bias.dtype
        fp_meta_data.close()
        
        self.user_factor.tofile(path + '.uf')
        self.user_bias.tofile(path + '.ub')
        self.item_factor.tofile(path + '.if')
        self.item_bias.tofile(path + '.ib')

    def load_model(self, path):
        fp_meta_data = open(path + '.meta')
        self.mu = float(fp_meta_data.readline())

        r, c, dtype = fp_meta_data.readline().split()
        self.user_factor = np.matrix(np.fromfile(path + '.uf', dtype = np.dtype(dtype)))
        self.user_factor.shape = (int(r), int(c))

        r, c, dtype = fp_meta_data.readline().split()
        self.user_bias = np.matrix(np.fromfile(path + '.ub', dtype = np.dtype(dtype)))
        self.user_bias.shape = (int(r), int(c))

        r, c, dtype = fp_meta_data.readline().split()
        self.item_factor = np.matrix(np.fromfile(path + '.if', dtype = np.dtype(dtype)))
        self.item_factor.shape = (int(r), int(c))

        r, c, dtype = fp_meta_data.readline().split()
        self.item_bias = np.matrix(np.fromfile(path + '.ib', dtype = np.dtype(dtype)))
        self.item_bias.shape = (int(r), int(c))

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
    test_data = 'data/movielens.1m.test'

    train_ratings = read_sparse_matrix(open(train_data))
    print >> sys.stderr, 'read training sparse matrix done.'
    test_ratings = read_sparse_matrix(open(test_data), train_ratings.rows, train_ratings.cols)
    print >> sys.stderr, 'read test sparse matrix done.'

    # pickle.dump(train_ratings, open('train_ratings.pkl', 'w'))
    # pickle.dump(test_ratings, open('test_ratings.pkl', 'w'))
    # ratings = pickle.load(open('ratings.pkl'))

    mf = MF()
    mf.train(train_ratings)

    rmse_train = mf.test(train_ratings)
    rmse_test = mf.test(test_ratings)
    
    print >> sys.stderr, 'Training RMSE for MovieLens 1m dataset: %lf' % rmse_train
    print >> sys.stderr, 'Test RMSE for MovieLens 1m dataset: %lf' % rmse_test


