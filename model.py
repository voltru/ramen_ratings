from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


#tf.keras.backend.set_floatx('float64')

def input_fn(features, labels, shuffle, num_epochs, batch_size):
    
    if labels is None:
        inputs = features
    else:
        inputs = (features, labels)
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(features))
    
    
    
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)
    return dataset


def create_keras_model(input_dim, learning_rate):
    
    Dense = tf.keras.layers.Dense
    model = tf.keras.Sequential(
        [
            Dense(100, activation=tf.nn.relu, kernel_initializer='uniform',
                  input_shape=(input_dim,)),
            Dense(75, activation=tf.nn.relu),
            Dense(50, activation=tf.nn.relu),
            Dense(25, activation=tf.nn.relu),
            Dense(5, activation=tf.nn.softmax)
        ])

    
    optimizer = tf.keras.optimizers.RMSprop(lr=learning_rate)

    
    model.compile(
        loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model
