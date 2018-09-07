"""
 * Created with PyCharm.
 * User: 彭诗杰
 * Date: 2018/9/7
 * Time: 0:11
 * Description: keras 示例程序
"""
import numpy as np
from keras.layers.core import Dense
from keras.models import Sequential
from keras.utils import np_utils


def load_data():
    f = np.load('mnist.npz')
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']
    f.close()

    number = 10000
    x_train = x_train[0:number]
    y_train = y_train[0:number]
    x_train = x_train.reshape(number, 28 * 28)
    x_test = x_test.reshape(x_test.shape[0], 28 * 28)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)
    x_train = x_train
    x_test = x_test
    x_train = x_train / 255
    x_test = x_test / 255
    return (x_train, y_train), (x_test, y_test)


(X_train, y_train), (X_test, y_test) = load_data()

model = Sequential()
model.add(Dense(units=512, activation='relu', input_dim=28 * 28))
model.add(Dense(units=512, activation='relu'))
model.add(Dense(units=512, activation='relu'))

model.add(Dense(units=10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
# x_train and y_train are Numpy arrays --just like in the Scikit-Learn API.
model.fit(X_train, y_train, epochs=5)

loss_and_metrics = model.evaluate(X_test, y_test, batch_size=128)

print(loss_and_metrics)
