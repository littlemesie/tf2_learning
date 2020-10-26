import gzip
import numpy as np
# 'train-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz',
#       't10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz'
def load_data(y_train_path, x_train_path, y_test_path, x_test_path):
    with gzip.open(y_train_path, 'rb') as lbpath:
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)
    with gzip.open(x_train_path, 'rb') as imgpath:
        x_train = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28)

    with gzip.open(y_test_path, 'rb') as lbpath:
        y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(x_test_path, 'rb') as imgpath:
        x_test = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_test), 28, 28)

    return (x_train, y_train), (x_test, y_test)

# load_data('../../data/fashion-mnist/train-labels-idx1-ubyte.gz')