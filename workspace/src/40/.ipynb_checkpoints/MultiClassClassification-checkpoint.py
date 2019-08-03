#!/usr/bin/env python
# coding: utf-8

# In[1]:


## development environment
# keras=2.2.0
# tensorflow=1.8.0
# dask=0.18.1
import keras
import tensorflow as tf
import dask

import warnings
warnings.filterwarnings('ignore')

print("keras={}".format(keras.__version__))
print("tensorflow={}".format(tf.__version__))
print("dask={}".format(dask.__version__))


# In[2]:


from keras.models import Sequential
from keras.layers import Dense


# In[4]:


# 可視化用ライブラリの読み込み
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Keras関連
from keras.utils import np_utils
from keras import backend as K
from keras.layers import Dropout,Flatten,Conv2D,MaxPooling2D,BatchNormalization,Activation
from keras import regularizers
from keras.optimizers import SGD, Adadelta, Adam, RMSprop# 最適化手法
import time


# In[109]:


# データのインポート
import dataset

X, Y = dataset.load_data()

# type1のみ使用
y=Y[0]


# In[110]:


# x_train=X[0:700]
# x_test=X[700:]

# y_train=y[0:700]
# y_test=y[700:]


# In[6]:


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


# In[113]:


img_rows, img_cols = 64, 64
img_channels = 4
nb_classes = 18 # 正解のパターン数


# In[114]:


# # 各ラベルごとに画像を10枚格納
# img_list = []
# for for_1 in range(3):
#     choice_idx = np.random.choice(np.where(y_test == for_1)[0], 10)
#     img_list.append(x_test[choice_idx])


# In[115]:


# # データの可視化
# for for_1 in range(3):
#     fig, ax = plt.subplots(1, 10, figsize=(18, 8))
#     for for_2 in range(10):
#         ax[for_2].imshow(img_list[for_1][for_2].reshape(img_rows, img_cols, img_channels)) #for_2の値+nでn番目以降のテストデータを出力する．
#         ax[for_2].set_title(type_labels[for_1])
#         ax[for_2].axis('off')


# In[116]:


# 正規化
x_train = x_train.astype(np.float32) / 255.
x_test = x_test.astype(np.float32) / 255.

nb_classes = 18 # 正解のパターン数


# In[117]:


# from keras.utils import np_utils
# # クラスベクトルをバイナリクラスの行列に変換する
# y_train = np_utils.to_categorical(y_train, nb_classes)
# y_test = np_utils.to_categorical(y_test, nb_classes)


# In[118]:


# # backendの違いによる次元数の入力型の調整(おまじない)
# from keras import backend as K

# if K.image_data_format() == 'channels_first':
#     x_train = x_train.reshape(x_train.shape[0], img_channels, img_rows, img_cols)
#     x_test = x_test.reshape(x_test.shape[0], img_channels, img_rows, img_cols)
#     input_shape = (img_channels, img_rows, img_cols)
# else:
#     x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, img_channels)
#     x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, img_channels)
#     input_shape = (img_rows, img_cols, img_channels)


# In[13]:


from keras.layers import Dropout,Flatten,Conv2D,MaxPooling2D,BatchNormalization,Activation
from keras import regularizers


# In[106]:


# """
# model.add()の中にConv2DやMaxPooling2Dをいれてモデルを作ってみよう

# サンプルで使用している関数一覧
# ---
# # データの一次元配列化
#     model.add(Flatten()) # 全結合層につなげる直前に使おう
# # 全結合層
#     model.add(Dense(次元数, activation=活性化関数)) 
# # 畳み込み層
#     model.add(Conv2D(次元数, kernel_size=フィルターのサイズ,activation=活性化関数,input_shape=input_shape))
# # プーリング層
#     model.add(MaxPooling2D(pool_size=プーリングするサイズ))
# # ドロップアウト
# model.add(Dropout(0から1までの数値)) # 学習するパーセプトロンのうち使用しない割合を設定
# ---

# その他、調べてみて便利な関数があればぜひ追加してみよう
# """

# weight_decay = 1e-4
# model = Sequential()

# model.add(Conv2D(32, (2,2), padding='same', activation='relu', input_shape=input_shape))
# model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Dropout(0.4))


# model.add(Conv2D(64, (2,2), padding='same', activation='relu'))
# model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Dropout(0.4))


# model.add(Conv2D(128, (2,2), padding='same', activation='relu'))
# model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Dropout(0.4))

# model.add(Conv2D(256, (2,2), padding='same', activation='relu'))
# model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Dropout(0.4))


# model.add(Flatten())
# model.add(Dropout(0.4))
# model.add(Dense(512, activation='relu'))
# model.add(Dense(18, activation='softmax'))


# In[ ]:


# model.summary()


# In[21]:


# from keras.optimizers import SGD, Adadelta, Adam, RMSprop# 最適化手法
# from keras import optimizers

# """
# モデルを評価する関数をmodel.compile()で定義しよう

# 実際にmodel.compileの中にはこのようにします

# model.compile(loss=誤差関数,
#              optimizer=最適化関数,
#              metrics=['accuracy']
#              )
             
# 誤差関数　モデルの精度の悪さを表す指標　誤差逆伝播時にパラメータの更新方向を決定する数値
# ・categorical_crossentropy

# 最適化関数(好きなものを選ぼう)　誤差逆伝播時にパラメータを更新する手法
# ・SGD
# ・Adadelta
# ・Adam
#   Adamは最近主流になっている最適化関数。一般的にSGDよりも優秀(例外はある)。
# ・RMSprop

# 評価指標
# ・accuracy
# """
# # 損失関数
# # def categorical_loss(y_true, y_pred):
# #     return K.categorical_crossentropy(y_true, y_pred)

# # 評価関数
# def total_acc(y_true, y_pred):
#     pred = K.cast(K.greater_equal(y_pred, 0.5), "float")
#     flag = K.cast(K.equal(y_true, pred), "float")
#     return K.prod(flag, axis=-1)

# def binary_acc(y_true, y_pred):
#     pred = K.cast(K.greater_equal(y_pred, 0.5), "float")
#     flag = K.cast(K.equal(y_true, pred), "float")
#     return K.mean(flag, axis=-1)


# model.compile(loss="categorical_crossentropy", # 誤差(損失)関数
#               optimizer="Adam", # 最適化関数
#               metrics=[total_acc, binary_acc] # 評価指標
#              )


# In[111]:


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


# In[ ]:


from sklearn.model_selection import KFold
import keras.backend as K


kf = KFold(n_splits=2, shuffle=False)

score_average = []
accuracy_average = []

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

for train_index, test_index in kf.split(X,y):
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = y[train_index], y[test_index]
    
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


# In[16]:


#モデルの保存
json_string = model.model.to_json()
open('predict.json', 'w').write(json_string)

#重みの保存
hdf5_file = "predict.hdf5"
model.model.save_weights(hdf5_file)


# In[29]:


from keras import models
from keras.models import model_from_json
from keras.preprocessing import image


#保存したモデルの読み込み
model = model_from_json(open('predict.json').read())
#保存した重みの読み込み
model.load_weights('predict.hdf5')


#画像を読み込む
img = Image.open("/workspace/data/images/0801.png")
img = img.resize((64,64))
x = np.array(img)
x = np.expand_dims(x, axis=0)

#予測
features = model.predict(x)

print(type_labels[np.argmax(features)])


# In[92]:


np.argmax(y_test)


# In[93]:


# for i in range(10):
#     # 予測値
#     # 各ラベルごとに画像を18枚格納
#     New_test = []
#     for for_1 in range(18):
#         choice_idx = np.random.choice(np.where(y_test == for_1)[0])
#         New_test.append(x_test[choice_idx])
#     New_test = np.array(New_test)
#     # 画像のラベルを推定する
#     y_test_pred = model.predict(New_test)

#     # データの可視化
#     # 上段は正しいラベル　下段は推測ラベル
#     fig, ax = plt.subplots(1, 18, figsize=(18, 8))
#     for for_1 in range(18):
#         ax[for_1].imshow(New_test[for_1].reshape(64, 64, 4)) #for_2の値+nでn番目以降のテストデータを出力する．
#         ax[for_1].set_title(type_labels[for_1]+"\n"+type_labels[np.argmax(y_test_pred[for_1])])
#         ax[for_1].axis('off')


# In[94]:


# # データの可視化
# # 上段は正しいラベル　下段は推測ラベル
# fig, ax = plt.subplots(1, 18, figsize=(18, 8))
# for for_1 in range(18):
#     ax[for_1].imshow(New_test[for_1].reshape(64, 64, 4)) #for_2の値+nでn番目以降のテストデータを出力する．
#     ax[for_1].set_title(type_labels[for_1]+"\n"+type_labels[np.argmax(y_test_pred[for_1])])
#     ax[for_1].axis('off')


# In[98]:


get_ipython().system('jupyter nbconvert --to python MultiClassClassification.ipynb')


# In[ ]:




