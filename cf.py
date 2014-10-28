import sys
import json
import math
import heapq
import time

def line_parser(line):
    uid, iid, rating, log_timestamp = line.strip().split('::')
    rating = float(rating)
    log_timestamp = int(log_timestamp)
    return uid, iid, rating, log_timestamp

def sparse_dot(L1, L2):
    dot = 0.0
    p1 = p2 = 0
    while p1 < len(L1) and p2 < len(L2):
        a1 = L1[p1][0]
        a2 = L2[p2][0]
        if a1 == a2:
            dot += L1[p1][1] * L2[p2][1]
            p1 += 1
            p2 += 1
        elif a1 > a2: p2 += 1
        else: p1 += 1
    return dot

def pair_key(id_p, id_q):
    K = id_p + ':' + id_q if id_p < id_q else id_q + ':' + id_p
    return K

def read_movie_index(fp_movie):
    L = ['unknown', 'Action', 'Adventure', 'Animation', 'Children\'s', \
         'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', \
         'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', \
         'Sci-Fi', 'Thriller', 'War', 'Western']

    movie_index = dict()
    for line in fp_movie:
        line_arr = line.strip().split('|')
        iid, title, release_date, video_release_date, url = line_arr[0 : 5]
        T = list()
        for id, tag in zip(line_arr[5 : ], L):
            if id == '1': T.append(tag)
        
        movie_index[iid] = [title, '|'.join(T)]
    return movie_index

class CF:
    def __init__(self):
        self.U = None
        self.V = None
        self.S = None
        self.C = None
        self.b = None
        self.n = None
        self.r_mean = 0.0

    def train(self, fp_ratings):
        self.V = dict()
        self.U = dict()
        self.b = dict()
    
        for c, line in enumerate(fp_ratings):
            
            # print >> sys.stderr, 'Line : %8.8i' % c,
            # print >> sys.stderr, '\r',

            uid, iid, rating, log_timestamp = line_parser(line)

            if not uid in self.U:
                self.U[uid] = dict()
            if not iid in self.U[uid] or log_timestamp > self.U[uid][iid][1]:
                self.U[uid][iid] = [rating, log_timestamp]
            
            if not iid in self.V: 
                self.V[iid] = dict()
            if not uid in self.V[iid] or log_timestamp > self.V[iid][uid][1]:
                self.V[iid][uid] = [rating, log_timestamp]

            if not iid in self.b:
                self.b[iid] = [0.0, 0]
            self.b[iid][0] += rating
            self.b[iid][1] += 1


        self.r_mean = 1.0 * sum([s for s, c in self.b.values()]) / sum([c for s, c in self.b.values()])
        
        print >> sys.stderr, 'UserId: %d' % len(self.U)
        print >> sys.stderr, 'ItemId: %d' % len(self.V)
        print >> sys.stderr, 'Mean rating: %lf' % self.r_mean

        for iid, (s, c) in self.b.items():
            self.b[iid] = 1.0 * s / c

        for uid, item_dict in self.U.items():
            L = [(iid, rating) for iid, (rating, log_timestamp) in item_dict.items()]
            L.sort(key = lambda x : x[0], reverse = False)
            self.U[uid] = L

        for iid, user_dict in self.V.items():
            L = [(uid, rating) for uid, (rating, log_timestamp) in user_dict.items()]
            L.sort(key = lambda x : x[0], reverse = False)
            self.V[iid] = L

    def test(self, fp_ratings):
        rmse = 0.0
        for c, line in enumerate(fp_ratings):
            uid, iid, rating, log_timestamp = line_parser(line)
            r_pred = self.predict(uid, iid)
            rmse += (rating - r_pred) ** 2
    
        rmse = math.sqrt(1.0 * rmse / (c + 1))
        return rmse

    def rec(self, uid, K = 10):
        if not uid in self.U:
            print >> sys.stderr, 'uid %s not exist' % uid
            return None

        # TODO: O(n*log(K)) for heap, O(n*log(n)) for sorting L
        # When K is small, heap is faster
        # When K is large, sort L directly is faster due to the extra cost of heap

        L = list()
        for iid in self.V:
            L.append((self.predict(uid, iid), iid))

        T_heap_s = time.time()
        rec_list = list()
        for I in L:
            if len(rec_list) < K:
                heapq.heappush(rec_list, I)
            elif I > rec_list[0]:
                heapq.heapreplace(rec_list, I)
        rec_list = reversed([heapq.heappop(rec_list) for x in range(len(rec_list))])
        T_heap_t = time.time()

        T_list_s = time.time()
        # L.sort(key = lambda x : x[0], reverse = True)
        # rec_list = L[0 : K]
        T_list_t = time.time()
        
        print >> sys.stderr, 'Time duration for HEAP: %lf' % (T_heap_t - T_heap_s)
        print >> sys.stderr, 'Time duration for LIST: %lf' % (T_list_t - T_list_s)

        rec_list = [(iid, score) for score, iid in rec_list]
        return rec_list

class ItemCF(CF):

    def __init__(self):
        CF.__init__(self)

    def train(self, fp_ratings):
        CF.train(self, fp_ratings)

        self.n = dict()
        for iid, L in self.V.items():
            self.n[iid] = math.sqrt(sum([r*r for u, r in L]))

        c = 0
        self.S = dict()
        for iid_p in self.V:
            for iid_q in self.V:
                c += 1
                K = pair_key(iid_p, iid_q)
                if K in self.S: continue
                self.S[K] = self.sim(iid_p, iid_q)
        print >> sys.stderr, 'Pair-wise ItemSim: %d' % len(self.S)

    def sim(self, iid_p, iid_q):
        if not iid_p in self.V or not iid_q in self.V:
            return 0.0
        Vp = self.V[iid_p]
        Vq = self.V[iid_q]
        return 1.0 * sparse_dot(Vp, Vq) / (self.n[iid_p] * self.n[iid_q])

    def predict(self, uid, iid_p):
        if not uid in self.U:
            if iid_p in self.b: return self.b[iid_p]
            return self.r_mean
        if not iid_p in self.V:
            return self.r_mean

        s = n = 0.0
        for iid_q, rating in self.U[uid]:
            K = pair_key(iid_p, iid_q)
            if not K in self.S: continue
            w = self.S[K]
            s += w * rating
            n += w

        if n == 0:
            if iid_p in self.b:
                s = self.b[iid_p]
            else:
                s = self.r_mean
        else:
            s = s / n
        return s


class UserCF(CF):

    def __init__(self):
        CF.__init__(self)

    def train(self, fp_ratings):
        CF.train(self, fp_ratings)

        self.n = dict()
        for uid, L in self.U.items():
            self.n[uid] = math.sqrt(sum([r*r for u, r in L]))

        c = 0
        self.S = dict()
        for uid_p in self.U:
            for uid_q in self.U:
                c += 1
                K = pair_key(uid_p, uid_q)
                if K in self.S: continue
                self.S[K] = self.sim(uid_p, uid_q)
        print >> sys.stderr, 'Pair-wise UserSim: %d' % len(self.S)
        
    def sim(self, uid_p, uid_q):
        if not uid_p in self.U or not uid_q in self.U:
            return 0.0
        Up = self.U[uid_p]
        Uq = self.U[uid_q]
        return 1.0 * sparse_dot(Up, Uq) / (self.n[uid_p] * self.n[uid_q])

    def predict(self, uid_p, iid):
        if not uid_p in self.U:
            if iid in self.b: return self.b[iid]
            return self.r_mean
        if not iid in self.V:
            return self.r_mean

        s = n = 0.0
        for uid_q, rating in self.V[iid]:
            K = pair_key(uid_p, uid_q)
            if not K in self.S: continue
            w = self.S[K]
            s += w * rating
            n += w

        if n == 0:
            if iid in self.b:
                s = self.b[iid]
            else:
                s = self.r_mean
        else:
            s = s / n
        return s

class SlopeOneCF(ItemCF):
    def __init__(self):
        ItemCF.__init__(self)

    def train(self, fp_ratings):
        ItemCF.train(self, fp_ratings)
      
        self.C = dict() 
        for iid_p in self.V:
            for iid_q in self.V: 
                K = pair_key(iid_p, iid_q)
                if K in self.C: continue
                self.C[K] = (iid_p, self.b[iid_q] - self.b[iid_p])
        print >> sys.stderr, 'Pair-wise SlopeOne: %d' % len(self.C)


    def predict(self, uid, iid_p):
        if not uid in self.U:
            if iid_p in self.b: return self.b[iid_p]
            return self.r_mean
        if not iid_p in self.V:
            return self.r_mean

        s = n = 0.0
        for iid_q, rating in self.U[uid]:
            K = pair_key(iid_p, iid_q)
            if not K in self.S or not K in self.C: continue

            w = self.S[K]

            if self.C[K][0] == iid_p:
                r = rating - self.C[K][1]
            elif self.C[K][0] == iid_q:
                r = rating + self.C[K][1]
            s += w * r
            n += w

        if n == 0:
            if iid_p in self.b:
                s = self.b[iid_p]
            else:
                s = self.r_mean
        else:
            s = s / n
        return s



def print_item_list(L, movie_index):
    for iid, score in L:
        title, label = movie_index[iid]
        print >> sys.stderr, iid + '\t' + title + '\t' + str(score) + '\t' + label

if __name__ == '__main__':

    uid = '310'

    movie_index = read_movie_index(open('data/movielens.100k.index'))

    cf = ItemCF()
    cf.train(open('data/movielens.100k.train'))
    rmse = cf.test(open('data/movielens.100k.test'))
    print >> sys.stderr, 'RMSE for ItemCF on MovieLens 100K dataset: %lf' % rmse
    rec_list = cf.rec(uid)
    user_profile = cf.U[uid]

    print >> sys.stderr, '=' * 100
    print >> sys.stderr, 'user profile for %s: ' % uid
    print_item_list(user_profile, movie_index)
    print >> sys.stderr, '=' * 100
    print >> sys.stderr, 'rec list: '
    print_item_list(rec_list, movie_index)
    print >> sys.stderr, '=' * 100

    cf = UserCF()
    cf.train(open('data/movielens.100k.train'))
    rmse = cf.test(open('data/movielens.100k.test'))
    print >> sys.stderr, 'RMSE for UserCF on MovieLens 100K dataset: %lf' % rmse
    rec_list = cf.rec(uid)
    user_profile = cf.U[uid]

    print >> sys.stderr, '=' * 100
    print >> sys.stderr, 'user profile for %s: ' % uid
    print_item_list(user_profile, movie_index)
    print >> sys.stderr, '=' * 100
    print >> sys.stderr, 'rec list: '
    print_item_list(rec_list, movie_index)
    print >> sys.stderr, '=' * 100

    cf = SlopeOneCF()
    cf.train(open('data/movielens.100k.train'))
    rmse = cf.test(open('data/movielens.100k.test'))
    print >> sys.stderr, 'RMSE for SlopeOneCF on MovieLens 100K dataset: %lf' % rmse
    rec_list = cf.rec(uid)
    user_profile = cf.U[uid]

    print >> sys.stderr, '=' * 100
    print >> sys.stderr, 'user profile for %s: ' % uid
    print_item_list(user_profile, movie_index)
    print >> sys.stderr, '=' * 100
    print >> sys.stderr, 'rec list: '
    print_item_list(rec_list, movie_index)
    print >> sys.stderr, '=' * 100
    
