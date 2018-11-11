import argparse
from datetime import datetime
from os import environ
import keras
import numpy as np
import pandas as pd
import tensorflow as tf
import Layers
from keras.callbacks import (EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard)
from keras.metrics import binary_accuracy, binary_crossentropy
from keras.models import Model
from keras.optimizers import SGD, Adam
from keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import class_weight
from keras.layers import Input, Dense, Activation, Conv2D, Flatten
from keras.datasets import cifar100, cifar10

environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Shut up tensorflow!

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
keras.backend.tensorflow_backend.set_session(session)


class Fitness(object):

    def __init__(self, list_layers=[], father=None, folder_name='models/'):
        self.classes = 100
        self.epochs = 100
        self.workers = 16
        self.batch_size = 8
        self.learning_rate = 1e-3
        self.lr_wait = 10
        self.decay = 1e-4
        self.momentum = 0.9
        self.resume = ''
        self.seed = 1337
        self.img_channels = 3
        self.img_size = 32
        self.early_stop = 10
        self.list_layers = list_layers
        self.father = father
        self.folder_name = folder_name

    def build_net_with_layer(self):
        if self.list_layers == []:
            input = Input(shape=self.img_shape)
            x = Conv2D(32, kernel_size=4, strides=4)(input)
            x = Flatten()(x)
            # x = Dense(self.classes, activation='sigmoid')(x)
            x = Dense(self.classes, activation='softmax')(x)
            self.model = Model(input, x)
        else:
            input = Input(shape=self.img_shape, name='Input')
            x = input
            x1 = x
            for layer in self.list_layers:
                if layer.layer_type == 'concatenate':
                    x = layer.get_layer()([x, x1])
                else:
                    x = layer.get_layer()(x)
                if layer.concatenate:
                    x1 = x
            x = Flatten(name='Flatten')(x)
            x = Dense(self.classes, activation='softmax', name='output')(x)
            self.model = Model(input, x)

    def load_weights(self):
        if 'base' in str(self.father):
            file_path = 'seed/' + str(self.father) + '/weights.hdf5'
            self.model.load_weights(file_path, by_name=True, skip_mismatch=True, reshape=True)
            print('load net ' + str(self.father) + ' weights')
        elif self.father == 'DN169':
            file_path = 'seed/densenet169.hdf5'
            self.model.load_weights(file_path, by_name=True, skip_mismatch=True, reshape=True)
            print('load net ' + str(self.father) + ' weights')
        elif self.father is not None:
            file_path = 'population/' + str(self.father) + '/weights.hdf5'
            self.model.load_weights(file_path, by_name=True, skip_mismatch=True, reshape=True)
            print('load net ' + str(self.father) + ' weights')

    def train(self):
        self.img_shape = (self.img_size, self.img_size, self.img_channels)  # blame theano
        now_iso = datetime.now().strftime('%Y-%m-%dT%H:%M:%S%z')

        (x_train, y_train), (x_test, y_test) = cifar100.load_data()
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255
        y_train = y_train.reshape(y_train.shape[0])
        y_test = y_test.reshape(y_test.shape[0])
        y_train = keras.utils.to_categorical(y_train, self.classes)
        y_test = keras.utils.to_categorical(y_test, self.classes)

        try:
            self.build_net_with_layer()
        except:
            print('can not build net!')
            return 0

        try:
            self.load_weights()
            pass
        except:
            print('can not load weights!')

        print(self.model.summary())

        file_path = self.folder_name + 'weights.hdf5'
        checkpoint = ModelCheckpoint(filepath=file_path, verbose=1, save_best_only=True)
        early_stop = EarlyStopping(monitor='val_loss', patience=self.early_stop, mode='auto')
        tensorboard = TensorBoard(log_dir=self.folder_name)
        reduce_lr = ReduceLROnPlateau(factor=0.03, cooldown=0, patience=self.lr_wait, min_lr=0.1e-6)
        callbacks = [early_stop, checkpoint]

        self.model.compile(
            # optimizer=Adam(lr=self.learning_rate, decay=self.decay),
            optimizer=SGD(lr=self.learning_rate, decay=self.decay, momentum=self.momentum, nesterov=True),
            loss='categorical_crossentropy',
            metrics=['accuracy'], )

        aug = ImageDataGenerator(width_shift_range=0.125, height_shift_range=0.125, horizontal_flip=True)
        aug.fit(x_train)
        gen = aug.flow(x_train, y_train, batch_size=self.batch_size)
        model_out = self.model.fit_generator(generator=gen,
                                             # steps_per_epoch=50000 / self.batch_size,
                                             epochs=self.epochs,
                                             validation_data=(x_test, y_test),
                                             workers=self.workers,
                                             use_multiprocessing=True,
                                             callbacks=callbacks)
        '''
        model_out = self.model.fit(x_train, y_train, batch_size=self.batch_size, epochs=self.epochs, verbose=1,
                                   callbacks=callbacks,
                                   validation_data=(x_test, y_test))
        # score = self.model.evaluate(x_test, y_test, batch_size=self.batch_size)
        # print(score)
        '''
        '''
        weights_path = self.folder_name + 'weights.hdf5'
        self.model.save_weights(filepath=weights_path, overwrite=True)
        print('weights save to ' + weights_path)
        '''
        q = max(model_out.history["val_acc"])
        return q


if __name__ == '__main__':
    f = Fitness()
    f.train()
