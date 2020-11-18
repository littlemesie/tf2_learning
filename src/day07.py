import tensorflow as tf
import tensorflow.keras as keras
"""
embedding操作
"""
embedding = tf.constant(
    [
        [0.21, 0.41, 0.51, 0.11],
        [0.22, 0.42, 0.52, 0.12],
        [0.23, 0.43, 0.53, 0.13],
        [0.24, 0.44, 0.54, 0.14]
    ], dtype=tf.float32)

feature_batch = tf.constant([[2, 3, 1, 0]])

get_embedding1 = tf.nn.embedding_lookup(embedding, feature_batch)

feature_batch_one_hot = tf.one_hot(feature_batch, depth=4)

get_embedding2 = tf.matmul(feature_batch_one_hot, embedding)


input_x = tf.keras.Input(shape=(None,), )

embedding_x = keras.layers.Embedding(10, 10)(input_x)
hidden1 = keras.layers.Dense(50, activation='relu')(embedding_x)
output = keras.layers.Dense(2, activation='softmax')(hidden1)

x_train = [2, 3, 4, 5, 8, 1, 6, 7, 2, 3, 4, 5, 8, 1, 6, 7, 2, 3, 4, 5, 8, 1, 6, 7, 2, 3, 4, 5, 8, 1, 6, 7, 2, 3, 4, 5,
           8, 1, 6, 7, 2, 3, 4, 5, 8, 1, 6, 7, 2, 3, 4, 5, 8, 1, 6, 7, 2, 3, 4, 5, 8, 1, 6, 7]
y_train = [0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1,
           1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1]

model2 = tf.keras.Model(inputs=input_x, outputs=output)
model2.compile(optimizer=tf.keras.optimizers.Adam(0.001),
               loss='sparse_categorical_crossentropy',
               metrics=['accuracy'])

history = model2.fit(x_train, y_train)
print(history.history)

