import linecache
import tensorflow as tf
import numpy as np
import random
import math

class Dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

class Dataset(object):

    def __init__(self):
        self.V1, self.V2 = self._load_data()
        self.num_nodes=self.V1.shape[0]
        self._order = np.arange(self.num_nodes)
        self._index_in_epoch = 0
        self.is_epoch_end = False

    def _load_data(self):
        V1=np.ones([1000,100])
        V2=np.ones([1000,100])
        return V1, V2
        
    def generate_samples(self,do_shuffle=True):
        
        order = np.arange(self.num_nodes)
        if(do_shuffle):
            np.random.shuffle(order)
        return order

    def sample(self, batch_size, do_shuffle=True, with_label=True):
        # //每一次迭代结束，重新初始化列表
        if self.is_epoch_end: 
            # 是否需要重新初始化
            if do_shuffle:
                np.random.shuffle(self._order)
            else:
                self._order = np.sort(self._order)
            self.is_epoch_end = False
            self._index_in_epoch = 0

        mini_batch = Dotdict()
        end_index = min(self.num_nodes, self._index_in_epoch + batch_size)
        cur_index = self._order[self._index_in_epoch:end_index]
        mini_batch.V1 = self.V1[cur_index]
        mini_batch.V2 = self.V2[cur_index]

        # 返回当前索引
        mini_batch.cur_index=cur_index

        # if with_label:
        #     mini_batch.Y = self.Y[cur_index]

        if end_index == self.num_nodes:
            end_index = 0
            self.is_epoch_end = True

        self._index_in_epoch = end_index

        return mini_batch

    def sample_by_idx(self, idx):
        mini_batch = Dotdict()
        mini_batch.V1 = self.V1[idx]
        mini_batch.V2 = self.V2[idx]

        mini_batch.idx=idx
        return mini_batch




class DatasetDUg():
    def __init__(self, train_x=None, train_y=None, test_x=None, test_y=None):
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
    
    def gen_next_batch(self, batch_size, is_train_set, epoch=None, iteration=None):
        if is_train_set==True:
            x = self.train_x
            y = self.train_y
        else:
            x = self.test_x
            y = self.test_y
            
        assert len(x)>=batch_size, "batch size must be smaller than data size {}.".format(len(x))
        
        if epoch != None:
            until = math.ceil(float(epoch*len(x))/float(batch_size))
        elif iteration != None:
            until = iteration
        else:
            assert False, "epoch or iteration must be set."
        
        iter_ = 0
        index_list = [i for i in range(len(x))]
        while iter_ <= until:
            idxs = random.sample(index_list, batch_size)
            iter_ += 1
            yield (x[idxs], y[idxs], idxs)


# class MNIST(Dataset):
#     def __init__(self):
#         super().__init__()
#         (self.train_x, self.train_y), (self.test_x, self.test_y) = tf.keras.datasets.mnist.load_data()
#         self.train_x = self.train_x.reshape(-1, self.train_x.shape[1]*self.train_x.shape[2])
#         self.train_x = self.train_x*0.02
#         self.test_x = self.test_x.reshape(-1, self.test_x.shape[1]*self.test_x.shape[2])
#         self.test_x = self.test_x*0.02
#         self.num_classes = 10
#         self.feature_dim = 784


