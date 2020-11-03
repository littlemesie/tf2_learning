import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from utils.titanic_util import preprocessing
tf.compat.v1.disable_eager_execution()

dftrain_raw = pd.read_csv('../data/titanic/train.csv')
dftest_raw = pd.read_csv('../data/titanic/test.csv')

x_train = preprocessing(dftrain_raw).values
y_train = dftrain_raw['Survived'].values

x_test = preprocessing(dftest_raw)
# y_test = dftest_raw['Survived'].values

class WideDeepModel(keras.models.Model):
    def __init__(self):
        super(WideDeepModel, self).__init__()
        """定义模型的层次"""
        self.hidden1_layer = keras.layers.Dense(32, activation='relu')
        self.hidden2_layer = keras.layers.Dense(16, activation='relu')
        self.output_layer = keras.layers.Dense(1)

    def call(self, inputs, training=None, mask=None):
        """完成模型的正向计算"""
        hidden1 = self.hidden1_layer(inputs)
        hidden2 = self.hidden2_layer(hidden1)
        concat = keras.layers.concatenate([inputs, hidden2])
        output = self.output_layer(concat)
        return output

model = WideDeepModel()
model.build(input_shape=(None, 15))

model.summary()
optimizer=tf.keras.optimizers.RMSprop(0.001)
model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['AUC'])
history = model.fit(x_train, y_train, batch_size=32, epochs=30)
