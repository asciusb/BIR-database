# Classification experiment for style detection (bold, italic, regular) in individual word images.

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.xception import Xception

from tensorflow.keras.applications.mobilenet import preprocess_input as preproc_mobilenet
from tensorflow.keras.applications.xception import preprocess_input as preproc_xception

from tensorflow.keras.callbacks import CSVLogger
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import random as python_random


def classify(name_experiment, path_train, path_valid, n_epochs):

    # settings

    train_images = []
    for wclass in ["0", "1", "2"]:
        dir_wclass = os.path.join(path_train, wclass)
        for file in os.listdir(dir_wclass):
            if file.endswith(".png"):
                train_images.append(os.path.join(dir_wclass, file))
    n_train = len(train_images)

    valid_images = []
    for wclass in ["0", "1", "2"]:
        dir_wclass = os.path.join(path_valid, wclass)
        for file in os.listdir(dir_wclass):
            if file.endswith(".png"):
                valid_images.append(os.path.join(dir_wclass, file))
    n_valid = len(valid_images)

    print(n_train, n_valid)
    i_shape = (160, 160, 3)
    i_size = (160, 160)
    n_classes = 3
    B = 32

    # base model

    pt_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=i_shape)
    #pt_model = Xception(weights='imagenet', include_top=False, input_shape=i_shape)

    print('model loaded')
    pt_model.trainable = True

    # model

    model = tf.keras.Sequential([
        pt_model,
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(100),
        keras.layers.Dropout(0.5),
        keras.layers.Activation('relu'),
        keras.layers.Dense(n_classes, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    # log

    dir_log = os.path.join("log", name_experiment)
    if not os.path.exists(dir_log):
        os.makedirs(dir_log)
    checkpoint_filepath = os.path.join(dir_log, "weigths.best")
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)
    file_logger = os.path.join(dir_log, "log.csv")
    csv_logger = CSVLogger(file_logger, append=True, separator=';')

    # data

    data_gen = ImageDataGenerator(preprocessing_function=preproc_mobilenet)
    #data_gen = ImageDataGenerator(preprocessing_function=preproc_xception)
    batches = data_gen.flow_from_directory(
        path_train,
        target_size=i_size,
        batch_size=B,
        shuffle=True)
    val_batches = data_gen.flow_from_directory(
        path_valid,
        target_size=i_size,
        batch_size=B,
        shuffle=False)

    # train

    model.fit(
        batches,
        steps_per_epoch=n_train / B,
        epochs=n_epochs,
        validation_data=val_batches,
        validation_steps=n_valid / B,
        use_multiprocessing=False,
        callbacks=[model_checkpoint_callback, csv_logger])

    # evaluation report

    model.load_weights(checkpoint_filepath)
    Y_pred = model.predict(val_batches, n_valid / B)
    y_pred = np.argmax(Y_pred, axis=1)
    print(confusion_matrix(val_batches.classes, y_pred))
    target_names = ['Regular', 'Bold', 'Italic']
    print(classification_report(val_batches.classes, y_pred, target_names=target_names))