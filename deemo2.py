import tensorflow as tf
import numpy as np
import random
def shuffle_set(train_image, train_label, test_image, test_label):
    train_row = range(len(train_label))
    random.shuffle(train_row)
    train_image = train_image[train_row]
    train_label = train_label[train_row]
    
    test_row = range(len(test_label))
    random.shuffle(test_row)
    test_image = test_image[test_row]
    test_label = test_label[test_row]
    return train_image, train_label, test_image, test_label

train_image, train_label, test_image, test_label=np.ones([1000,10]),np.ones([1000,1]),np.ones([1000,10]),np.ones([1000,1])

train_image2, train_label2, test_image2, test_label2=shuffle_set(train_image, train_label, test_image, test_label)