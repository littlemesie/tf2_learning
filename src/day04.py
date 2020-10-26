import numpy as np
from tensorflow import keras
from tensorflow.keras import models, layers
from matplotlib import pyplot as plt
from utils.fashion_mnist import load_data

y_train_path = '../data/fashion-mnist/train-labels-idx1-ubyte.gz'
x_train_path = '../data/fashion-mnist/train-images-idx3-ubyte.gz'
y_test_path = '../data/fashion-mnist/t10k-labels-idx1-ubyte.gz'
x_test_path = '../data/fashion-mnist/t10k-images-idx3-ubyte.gz'

(train_images, train_labels), (test_images, test_labels) = \
    load_data(y_train_path, x_train_path, y_test_path, x_test_path)
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# plt.figure()
# plt.imshow(train_images[0])
# plt.colorbar()
# plt.grid(False)
# plt.show()
#
# train_images = train_images / 255.0
#
# test_images = test_images / 255.0
# plt.figure(figsize=(10, 10))
# for i in range(25):
#     plt.subplot(5, 5, i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i], cmap=plt.cm.binary)
#     plt.xlabel(class_names[train_labels[i]])
# plt.show()

model = keras.Sequential(
[
    layers.Flatten(input_shape=[28, 28]),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5)

predictions = model.predict(test_images)
for i, pred in enumerate(predictions):
    print(test_labels[i], np.argmax(pred))