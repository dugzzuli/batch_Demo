import numpy as np
import linecache

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



