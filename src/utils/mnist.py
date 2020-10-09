import numpy as np

def load_data(data_path='mnist.npz'):

    with np.load(data_path, allow_pickle=True) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']

        return (x_train, y_train), (x_test, y_test)

# (x_train, y_train), (x_test, y_test) = load_data('../../data/mnist/mnist.npz')
# print(x_train)