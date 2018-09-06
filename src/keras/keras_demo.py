"""
 * Created with PyCharm.
 * User: 彭诗杰
 * Date: 2018/9/7
 * Time: 0:11
 * Description: keras 示例程序
"""
import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.utils import np_utils

# (X_train, y_train), (X_test, y_test) = mnist.load_data('mnist.npz')
f = np.load('mnist.npz')
x_train, y_train = f['x_train'], f['y_train']
x_test, y_test = f['x_test'], f['y_test']
f.close()

print(x_train.shape)
print(x_test.shape)


