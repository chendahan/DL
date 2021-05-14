import os
import random

import numpy as np
import tensorflow.keras.backend as K
from matplotlib import pyplot as plt
from sklearn.model_selection import ParameterGrid
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Input, Lambda
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.regularizers import l2
from tensorflow.python.keras.layers import BatchNormalization

IMAGES_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'lfw2', 'lfw2')


class LFW2DataLoader:
    def __init__(self, input_shape):
        self.input_shape = input_shape

    @staticmethod
    def _pad_image_number(image_number: str) -> str:
        return '0' * (4 - len(image_number)) + image_number

    def _join_image_path(self, name, number):
        return os.path.join(IMAGES_PATH,
                            name,
                            f'{name}_{self._pad_image_number(number)}.jpg')

    def _read_dataset_file(self, file_path):
        first_images_paths = []
        second_images_paths = []
        labels = []
        with open(file_path) as f:
            for line in f.readlines()[1:]:
                splitted_line = line.split()
                if len(splitted_line) == 3:
                    name = splitted_line[0]
                    first_images_paths.append(self._join_image_path(name, splitted_line[1]))
                    second_images_paths.append(self._join_image_path(name, splitted_line[2]))
                    labels.append(1)
                else:
                    first_images_paths.append(self._join_image_path(splitted_line[0], splitted_line[1]))
                    second_images_paths.append(self._join_image_path(splitted_line[2], splitted_line[3]))
                    labels.append(0)

        return first_images_paths, second_images_paths, labels

    def _open_images(self, image_paths):
        return np.array([img_to_array(load_img(f_path, color_mode='grayscale').resize(self.input_shape))
                         for f_path in image_paths])

    def load_images_from_path(self, path, normalize=True):
        first_images_paths, second_images_paths, labels = self._read_dataset_file(path)
        first_images = self._open_images(first_images_paths)
        second_images = self._open_images(second_images_paths)

        if normalize:
            first_images /= 255
            second_images /= 255

        return [first_images, second_images], np.array(labels)


def _add_convolutional_layer(convolutional_net, filters, kernel_size, kernel_regularizer, add_pool):
    convolutional_net.add(Conv2D(filters=filters,
                                 kernel_size=kernel_size,
                                 activation='relu',
                                 kernel_initializer=RandomNormal(mean=0, stddev=0.01),
                                 bias_initializer=RandomNormal(mean=0.5, stddev=0.01),
                                 kernel_regularizer=l2(kernel_regularizer)))
    convolutional_net.add(BatchNormalization())

    if add_pool:
        convolutional_net.add(MaxPool2D())


def _get_convolutional_network(kernel_regularizer_conv, kernel_regularizer_dense):
    convolutional_net = Sequential()
    _add_convolutional_layer(convolutional_net, 64, (10, 10), kernel_regularizer_conv, True)
    # convolutional_net.add(SpatialDropout2D(0.2))
    # convolutional_net.add(MaxPool2D(pool_size=(3, 3), strides=(3, 3)))

    _add_convolutional_layer(convolutional_net, 128, (7, 7), kernel_regularizer_conv, True)
    # convolutional_net.add(SpatialDropout2D(0.5))
    # convolutional_net.add(MaxPool2D(pool_size=(3, 3), strides=(3, 3)))

    _add_convolutional_layer(convolutional_net, 128, (4, 4), kernel_regularizer_conv, True)
    # convolutional_net.add(SpatialDropout2D(0.5))
    # convolutional_net.add(MaxPool2D())

    _add_convolutional_layer(convolutional_net, 256, (4, 4), kernel_regularizer_conv, False)

    convolutional_net.add(Flatten())
    convolutional_net.add(Dense(units=4096,
                                activation='sigmoid',
                                kernel_initializer=RandomNormal(mean=0, stddev=0.01),
                                bias_initializer=RandomNormal(mean=0.5, stddev=0.01),
                                kernel_regularizer=l2(kernel_regularizer_dense)))
    # convolutional_net.add(BatchNormalization())

    return convolutional_net


def get_network(learning_rate=0.001,
                momentum=0.5,
                decay_rate=0.98,
                use_sgd=True,
                kernel_regularizer_conv=0.01,
                kernel_regularizer_dense=0.0001,
                **kwargs):
    convolutional_network = _get_convolutional_network(kernel_regularizer_conv,
                                                       kernel_regularizer_dense)

    input_image_1 = Input(input_shape)
    input_image_2 = Input(input_shape)

    output_image_1 = convolutional_network(input_image_1)
    output_image_2 = convolutional_network(input_image_2)

    l1_distance_layer = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))
    l1_distance = l1_distance_layer([output_image_1, output_image_2])

    prediction = Dense(units=1,
                       activation='sigmoid',
                       kernel_initializer=RandomNormal(mean=0, stddev=0.01),
                       bias_initializer=RandomNormal(mean=0.5, stddev=0.01))(l1_distance)
    model = Model(inputs=[input_image_1, input_image_2], outputs=prediction)

    if use_sgd:
        optimizer = SGD(learning_rate=ExponentialDecay(learning_rate, 100000, decay_rate),
                        momentum=momentum)
    else:
        optimizer = Adam(learning_rate=ExponentialDecay(learning_rate, 100000, decay_rate))
    model.compile(loss='binary_crossentropy',
                  metrics=['binary_accuracy'],
                  optimizer=optimizer)

    return model


param_grid = {'learning_rate': [0.001, 0.0001, 0.01],
              'decay_rate': [0.99, 0.95, 0.9],
              'use_sgd': [False],
              'kernel_regularizer_conv': [1e-4, 0.01],
              'kernel_regularizer_dense': [1e-3, 0.0001],
              'batch_size': [16, 32, 64]
              # 'conv_kernel_initializer': [RandomNormal(mean=0, stddev=0.01)],
              # 'conv_bias_initializer': [RandomNormal(mean=0.5, stddev=0.01)],
              # 'fc_kernel_initializer': [RandomNormal(mean=0, stddev=0.2)],
              # 'fc_bias_initializer': [RandomNormal(mean=0.5, stddev=0.01)]
              }

param_grid_use_sgd = {'learning_rate': [0.001, 0.0001, 0.1],
                      'momentum': [0.5, 0.6, 0.7, 0.8, 0.9],
                      'decay_rate': [0.99, 0.95, 0.9],
                      'use_sgd': [True],
                      'kernel_regularizer_conv': [1e-4, 0.01],
                      'kernel_regularizer_dense': [1e-3, 0.0001],
                      'batch_size': [16, 32, 64]
                      }


def get_parameter_grid():
    adam_param_grid = ParameterGrid(param_grid)
    sgd_param_grid = ParameterGrid(param_grid_use_sgd)
    parameter_grid = list(adam_param_grid) + list(sgd_param_grid)
    random.shuffle(parameter_grid)

    return parameter_grid


def run_hp_tuning():
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    parameter_grid = get_parameter_grid()
    for parameters in parameter_grid:
        model = get_network(**parameters)
        model.fit(train_X,
                  train_y,
                  batch_size=parameters['batch_size'],
                  validation_split=0.2,
                  verbose=0,
                  epochs=200,
                  callbacks=[early_stopping])
        train_eval = model.evaluate(train_X, train_y, verbose=False)
        test_eval = model.evaluate(test_X, test_y, verbose=False)
        results = {'test_evaluation': test_eval,
                   'train_evaluation': train_eval,
                   **parameters}
        print(results)


def plot_losses(history, save=False, show=False):
    plt.plot(history.history['binary_accuracy'])
    plt.plot(history.history['val_binary_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'val'], loc='upper left')
    if save:
        plt.savefig('accuracy.jpg')
    if show:
        plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'val'], loc='upper left')
    if save:
        plt.savefig('loss.jpg')
    if show:
        plt.show()


if __name__ == '__main__':
    train_file_path = 'pairsDevTrain.txt'
    test_file_path = 'pairsDevTest.txt'

    input_size = 105
    input_shape = (input_size, input_size, 1)
    data_loader = LFW2DataLoader((input_size, input_size))
    train_X, train_y = data_loader.load_images_from_path(train_file_path)
    test_X, test_y = data_loader.load_images_from_path(test_file_path)

    model = get_network(learning_rate=0.0001, decay_rate=1, kernel_regularizer_conv=2e-4, use_sgd=False)
    early_stopping = EarlyStopping(monitor='val_loss', verbose=1, patience=20, restore_best_weights=True)
    history = model.fit(train_X,
                        train_y.astype(float),
                        batch_size=64,
                        validation_split=0.15,
                        epochs=200,
                        verbose=1,
                        callbacks=[early_stopping])

    train_eval = model.evaluate(train_X, train_y, verbose=False)
    test_eval = model.evaluate(test_X, test_y, verbose=False)

    print('Test evaluation - ', test_eval)
    print('Train evaluation - ', train_eval)
    plot_losses(history)

    print()
