#!/usr/bin/env python
# coding: utf-8


import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras import backend as K
from keras.layers import Dropout,Flatten,Conv2D,MaxPooling2D,BatchNormalization,Activation
from keras import regularizers
from keras.optimizers import SGD, Adadelta, Adam, RMSprop
import time
from sklearn.model_selection import KFold
import keras.backend as K
from keras.models import model_from_json
from keras.preprocessing import image


# データのインポート
import dataset

X, Y = dataset.load_data()

# type1のみ使用
y=Y[0]


# ラベル名を用意
type_labels = np.array([
    "Normal",
    "Fire",
    "Water",
    "Electric",
    "Grass",
    "Ice",
    "Fighting",
    "Poison",
    "Ground",
    "Flying",
    "Psychic",
    "Bug",
    "Rock",
    "Ghost",
    "Dragon",
    "Dark",
    "Steel",
    "Fairy"])


img_rows, img_cols = 64, 64
img_channels = 4 # RGBA
nb_classes = 18 # 正解のパターン数


def get_model():
    
    weight_decay = 1e-4
    model = Sequential()

    model.add(Conv2D(32, (2,2), padding='same', activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.4))


    model.add(Conv2D(64, (2,2), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.4))


    model.add(Conv2D(128, (2,2), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.4))

    model.add(Conv2D(256, (2,2), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.4))


    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(18, activation='softmax'))
    
    return model


# 損失関数
def categorical_loss(y_true, y_pred):
    return K.categorical_crossentropy(y_true, y_pred)

# 評価関数
def total_acc(y_true, y_pred):
    pred = K.cast(K.greater_equal(y_pred, 0.5), "float")
    flag = K.cast(K.equal(y_true, pred), "float")
    return K.prod(flag, axis=-1)

def binary_acc(y_true, y_pred):
    pred = K.cast(K.greater_equal(y_pred, 0.5), "float")
    flag = K.cast(K.equal(y_true, pred), "float")
    return K.mean(flag, axis=-1)

def train(X, y):
    kf = KFold(n_splits=5, shuffle=False)

    score_average = []
    accuracy_average = []

    for train_index, test_index in kf.split(X,y):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = y[train_index], y[test_index]

        X_train = X_train.astype(np.float32) / 255.
        X_test = X_test.astype(np.float32) / 255.

        Y_train = np_utils.to_categorical(Y_train, nb_classes)
        Y_test = np_utils.to_categorical(Y_test, nb_classes)

        if K.image_data_format() == 'channels_first':
            X_train = X_train.reshape(X_train.shape[0], img_channels, img_rows, img_cols)
            X_test = X_test.reshape(xX_test.shape[0], img_channels, img_rows, img_cols)
            input_shape = (img_channels, img_rows, img_cols)
        else:
            X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, img_channels)
            X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, img_channels)
            input_shape = (img_rows, img_cols, img_channels)

        model = get_model()

        model.compile(loss=categorical_loss, # 誤差(損失)関数
                  optimizer=optimizers.RMSprop(lr=1e-4), # 最適化関数
                  metrics=[total_acc, binary_acc] # 評価指標
                 )

        history = model.fit(X_train, Y_train,
                        batch_size=32,
                        nb_epoch=5,
                        verbose=1,
                        validation_data=(X_test, Y_test)
                       )

        score = model.evaluate(X_test, Y_test, verbose=0)
        print('Test score:', score[0]) # 損失関数の値
        score_average.append(score[0])
        print('Test accuracy:', score[1]) # 精度
        accuracy_average.append(score[1])
        print('-----------------------')


    print("test score average:", np.mean(score_average))
    print("test accuracy average:", np.mean(accuracy_average))


def save():
    #モデルの保存
    json_string = model.model.to_json()
    open('predict.json', 'w').write(json_string)

    #重みの保存
    hdf5_file = "predict.hdf5"
    model.model.save_weights(hdf5_file)



def predict(image_pass):

    #保存したモデルの読み込み
    model = model_from_json(open('predict.json').read())
    #保存した重みの読み込み
    model.load_weights('predict.hdf5')


    #画像を読み込む
    img = Image.open(image_pass)
    img = img.resize((64,64))
    img = img.convert('RGB')
    x = np.array(img)
    x = np.expand_dims(x, axis=0)

    #予測
    features = model.predict(x)

    return type_labels[np.argmax(features)]
