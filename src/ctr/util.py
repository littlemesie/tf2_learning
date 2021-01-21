import pandas as pd
import numpy as np
import tensorflow as tf

# load sample as tf dataset
def get_dataset(file_path):

    df = pd.read_csv(file_path, sep=',')
    str_column = ['userGenre1', 'userGenre2', 'userGenre3', 'userGenre4', 'userGenre5', 'movieGenre1', 'movieGenre2', 'movieGenre3']
    for c in str_column:

        df[c].fillna('', inplace=True)
    data_feature = {}
    columns = df.columns.values
    labels = []
    for rating in df['rating'].values:
        if int(rating) > 3:
            labels.append(1)
        else:
            labels.append(0)
    for column in columns:
        if column == 'rating':
            continue
        dt = np.array(df[column].values).dtype

        if dt == 'object':
            dt = 'string'
        elif dt == 'int64':
            dt = 'int32'
        elif dt == 'float64':
            dt = 'float32'
        else:
            dt = 'int32'
        # print(dt, column)
        # print(list(df[column].values)[:10])
        data_feature[column] = tf.constant(df[column].values, dtype=dt)
        # data_feature[column] = np.array(df[column].values, dtype=dt)

    return data_feature, np.array(labels)