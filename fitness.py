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

environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Shut up tensorflow!


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
keras.backend.tensorflow_backend.set_session(session)


class Fitness(object):

    def __init__(self, list_layers=[], father=None, folder_name='models/'):
        self.classes = 1
        self.workers = 4
        self.epochs = 120
        self.batch_size = 16
        self.learning_rate = 1e-4
        self.lr_wait = 10
        self.decay = 1e-4
        self.momentum = 0.9
        self.resume = ''
        self.seed = 1337
        self.img_channels = 3
        self.img_size = 224
        self.early_stop = 10
        self.list_layers = list_layers
        self.father = father
        self.folder_name = folder_name

        self.layer_num = 0

    def build_net_with_layer(self):
        if self.list_layers == []:
            input = Input(shape=self.img_shape)
            x = Conv2D(32, kernel_size=4, strides=4)(input)
            x = Flatten()(x)
            x = Dense(self.classes, activation='sigmoid')(x)
            self.model = Model(input, x)
        else:
            input = Input(shape=self.img_shape, name='Input')
            x = input
            x1 = x
            for layer in self.list_layers:
                if layer.layer_type == 'Concatenate':
                    x = layer.get_layer()([x, x1])
                else:
                    x = layer.get_layer()(x)
                if layer.concatenate:
                    x1 = x
            x = Flatten(name='Flatten')(x)
            x = Dense(self.classes, name='output')(x)
            self.model = Model(input, x)

    def load_weights(self):
        if self.father == 'base':
            file_path = 'seed/base/weights.hdf5'
            self.model.load_weights(file_path, by_name=True, skip_mismatch=True, reshape=True)
            print('load net ' + str(self.father) + ' weights')
        elif self.father == 'DN169':
            file_path = 'seed/densenet169.hdf5'
            self.model.load_weights(file_path, by_name=True, skip_mismatch=True, reshape=True)
            print('load net ' + str(self.father) + ' weights')
        elif self.father != None:
            file_path = 'population/' + str(self.father) + '/weights.hdf5'
            self.model.load_weights(file_path, by_name=True, skip_mismatch=True, reshape=True)
            print('load net ' + str(self.father) + ' weights')

    def train(self):
        self.img_shape = (self.img_size, self.img_size, self.img_channels)  # blame theano
        now_iso = datetime.now().strftime('%Y-%m-%dT%H:%M:%S%z')
        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            rotation_range=45,
            # width_shift_range=0.2,
            # height_shift_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)

        train_generator = train_datagen.flow_from_directory(
            '/home/fish/MRACdata/train',
            shuffle=True,
            target_size=(self.img_size, self.img_size),
            class_mode='binary',
            batch_size=self.batch_size, )

        val_datagen = ImageDataGenerator(rescale=1. / 255)
        val_generator = val_datagen.flow_from_directory(
            '/home/fish/MRACdata/val',
            shuffle=True,  # otherwise we get distorted batch-wise metrics
            class_mode='binary',
            target_size=(self.img_size, self.img_size),
            batch_size=self.batch_size, )

        classes = len(train_generator.class_indices)
        assert classes > 0
        assert classes is len(val_generator.class_indices)
        n_of_train_samples = train_generator.samples
        n_of_val_samples = val_generator.samples

        try:
            self.build_net_with_layer()
        except:
            print('can not build net!')
            return 0

        try:
            self.load_weights()
        except:
            print('can not load weights!')

        file_path = self.folder_name + 'checkpoint.hdf5'
        checkpoint = ModelCheckpoint(filepath=file_path, verbose=1, save_best_only=True)
        early_stop = EarlyStopping(patience=self.early_stop)
        tensorboard = TensorBoard(log_dir=self.folder_name)
        reduce_lr = ReduceLROnPlateau(factor=0.03, cooldown=0, patience=self.lr_wait, min_lr=0.1e-6)
        callbacks = [early_stop, reduce_lr]

        weights = class_weight.compute_class_weight('balanced', np.unique(train_generator.classes),
                                                    train_generator.classes)
        weights = {0: weights[0], 1: weights[1]}

        print(self.model.summary())

        if self.resume:
            self.model.load_weights(self.resume)
            for layer in self.model.layers:
                layer.set_trainable = True

        self.model.compile(
            optimizer=Adam(lr=self.learning_rate, decay=self.decay),
            # optimizer=SGD(lr=args.learning_rate, decay=args.decay,momentum=args.momentum, nesterov=True),
            loss=binary_crossentropy,
            metrics=[binary_accuracy], )

        model_out = self.model.fit_generator(
            train_generator,
            steps_per_epoch=n_of_train_samples // self.batch_size,
            epochs=self.epochs,
            validation_data=val_generator,
            validation_steps=n_of_val_samples // self.batch_size,
            class_weight=weights,
            workers=self.workers,
            use_multiprocessing=False,
            callbacks=callbacks)

        weights_path = self.folder_name + 'weights.hdf5'
        self.model.save_weights(filepath=weights_path, overwrite=True)
        print('weights save to ' + weights_path)

        q = max(model_out.history["val_binary_accuracy"])
        return q

    def append_layer(self, layer_in):
        self.layer_num = self.layer_num + 1
        layer_in.set_name(0, self.layer_num)
        self.list_layers.append(layer_in)

    def dense_block(self, blocks):
        for i in range(blocks):
            self.append_layer(Layers.BatchNormalizationLayer())
            self.append_layer(Layers.ActivationLayer('relu'))
            self.append_layer(Layers.ConvLayer(4 * 32, 1, 1, use_bias_in=False))
            self.append_layer(Layers.BatchNormalizationLayer())
            self.append_layer(Layers.ActivationLayer('relu'))
            self.append_layer(Layers.ConvLayer(32, 3, 1, padding_in='same', use_bias_in=False))
            self.append_layer(Layers.ConcatenateLayer(concatenate=True))

    def transition_block(self, conv_in):
        self.append_layer(Layers.BatchNormalizationLayer())
        self.append_layer(Layers.ActivationLayer('relu'))
        self.append_layer(Layers.ConvLayer(conv_in, 1, 1, use_bias_in=False))
        self.append_layer(Layers.PoolLayer(2, 2, type_in='average', concatenate=True))

    def test_layer(self):
        self.append_layer(Layers.ZeroPaddingLayer(3))
        self.append_layer(Layers.ConvLayer(64, 7, 2, use_bias_in=False))
        self.append_layer(Layers.BatchNormalizationLayer())
        self.append_layer(Layers.ActivationLayer('relu'))
        self.append_layer(Layers.ZeroPaddingLayer(1))
        self.append_layer(Layers.PoolLayer(3, 2, type_in='max', concatenate=True))
        self.dense_block(6)
        self.transition_block(128)
        self.dense_block(12)
        self.transition_block(256)
        self.dense_block(32)
        self.transition_block(640)
        self.dense_block(32)
        self.append_layer(Layers.BatchNormalizationLayer())
        self.append_layer(Layers.PoolLayer(2, 2, pool_type_in='globalaverage'))
        self.append_layer(Layers.ActivationLayer('sigmoid'))


if __name__ == '__main__':
    f = Fitness()
    f.test_layer()
    f.train()
