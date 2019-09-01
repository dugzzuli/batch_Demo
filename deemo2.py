import tensorflow as tf
import numpy as np
import random
from dataset import DatasetDUg

class MNIST(DatasetDUg):
    def __init__(self):
        super().__init__()
        (self.train_x, self.train_y), (self.test_x, self.test_y) = tf.keras.datasets.mnist.load_data()
        self.train_x = self.train_x.reshape(-1, self.train_x.shape[1]*self.train_x.shape[2])
        self.train_x = self.train_x*0.02
        self.test_x = self.test_x.reshape(-1, self.test_x.shape[1]*self.test_x.shape[2])
        self.test_x = self.test_x*0.02
        self.num_classes = 10
        self.feature_dim = 784

batch_size=10000
data=MNIST()
index=0
for iter_, (batch_x, batch_y, batch_idxs) in enumerate(data.gen_next_batch(batch_size=batch_size,is_train_set=True, epoch=3)):

    print(index)
    index=index+1


