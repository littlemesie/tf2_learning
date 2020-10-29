import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import models, layers
from utils.titanic_util import preprocessing

dftrain_raw = pd.read_csv('../data/titanic/train.csv')
dftest_raw = pd.read_csv('../data/titanic/test.csv')

x_train = preprocessing(dftrain_raw)
y_train = dftrain_raw['Survived'].values

x_test = preprocessing(dftest_raw)
# y_test = dftest_raw['Survived'].values

# 定义模型
model = models.Sequential()
model.add(layers.Dense(20, activation='relu', input_shape=(15,)))
model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='adam',
            loss='binary_crossentropy',
            metrics=['AUC'])

# 模型训练
history = model.fit(x_train, y_train,
                    batch_size=64,
                    epochs=30,
                    validation_split=0.2 # 分割一部分训练数据用于验证
                   )

# 评估模型可视化
def plot_metric(history, metric):
    train_metrics = history.history[metric]
    val_metrics = history.history['val_'+metric]
    epochs = range(1, len(train_metrics) + 1)
    plt.plot(epochs, train_metrics, 'bo--')
    plt.plot(epochs, val_metrics, 'ro-')
    plt.title('Training and validation ' + metric)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend(["train_"+metric, 'val_'+metric])
    plt.show()

print(history.history)
plot_metric(history, "loss")
plot_metric(history, "auc")

# # 测试集上的结果
# test_ret = model.evaluate(x=x_test, y=y_test)
# print(test_ret)
#
# # 测试集上预测结果
# #预测概率
# y_pro = model.predict(x_test[0:10])
# print(y_pro)
# #预测类别
# y_label = model.predict_classes(x_test[0:10])
# print(y_label)