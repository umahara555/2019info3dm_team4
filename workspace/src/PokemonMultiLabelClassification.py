from tensorflow.keras import regularizers, optimizers
from tensorflow.keras import utils
from tensorflow.keras.models import Sequential, model_from_json, model_from_yaml
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.callbacks import History, ModelCheckpoint, EarlyStopping
import pandas as pd
import numpy as np
from PIL import Image
import glob
import sys
import dataset


# 画像の形状情報
IMG_ROWS = 64
IMG_COLS = 64
IMG_CHANNELS = 3
IMG_CLASSES = 18

# fit parameters
BATCH_SIZE = 32
EPOCHS = 10

# 学習済みモデルの保存ファイル
MODEL_FILE_PATH = './pokemon_multi_label_classification_cnn_model.json'
INIT_WEIGHT_FILE_PATH = './pokemon_multi_label_classification_cnn_init_weight.hdf5'
BEST_WEIGHT_FILE_PATH = './pokemon_multi_label_classification_cnn_best_weight.hdf5'


def data_augmentation(images, labels):
    '''
    9倍
    '''
    new_images = []
    new_labels = []
    images_length = len(images)
    for image,label,i in zip(images,labels,range(images_length)):
        progress = f'{int((i/images_length)*100)}/100%'
        print(f'\rrunning data augmentation ... {progress}', end='')
        for x in [-16, 0, 16]:
            for y in [-16, 0, 16]:
                img = image

                # ずらし
                img = np.roll(img, x, axis=1)
                img = np.roll(img, y, axis=0)

                # ずらし後のはみ出し削除
                if y > 0:
                    img[:y] = 0
                elif y < 0:
                    img[y:] = 0
                if x > 0:
                    img[:, :x] = 0
                elif x < 0:
                    img[:, x:] = 0

                new_images.append(img)
                new_labels.append(label)

    print(f'\rrunning data augmentation ... 100/100% done')
    return np.array(new_images),np.array(new_labels)


def data_shuffle(images,types):
    indices = np.arange(images.shape[0])
    x,y = [], []
    for i in indices:
        x.append(images[i])
        y.append(types[i])

    return np.array(x),np.array(y)


def generate_model():
    model = Sequential()
    model.add(Conv2D(32, (5, 5),
                     padding='same',
                     activation='relu',
                     kernel_regularizer=regularizers.l2(0.001),
                     input_shape=(IMG_ROWS,IMG_COLS,IMG_CHANNELS)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Conv2D(64, (5, 5),
                     padding='same',
                     activation='relu',
                     kernel_regularizer=regularizers.l2(0.001)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(512,
                    activation='relu',
                    kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dropout(0.5))
    model.add(Dense(IMG_CLASSES,
                    kernel_regularizer=regularizers.l2(0.001),
                    activation='sigmoid'))

    model.compile(optimizers.Adam(lr=0.0001),
                  loss='binary_crossentropy',
                  metrics=["accuracy"])

    return model


def generator(x, y, batch_size=32):
    indices = np.arange(x.shape[0])
    while True:
        img_cahce, label_cache = [], []
        np.random.shuffle(indices)
        for i in indices:
            img_cahce.append(x[i])
            label_cache.append(y[i])
            if len(img_cahce) == batch_size:
                X_batch = np.array(img_cahce)
                Y_batch = np.array(label_cache)
                img_cahce, label_cache = [], []
                yield X_batch, Y_batch


def fit(raw_images, type1, type2):
    # 前処理
    images = raw_images.astype('float32') / 255.0
    type1_c = utils.to_categorical(type1, IMG_CLASSES)
    type2_c = utils.to_categorical(type2, IMG_CLASSES)
    types = type1_c + type2_c
    types = np.where(types > 1., 1., types)

    imgs, ts = data_shuffle(images, types)
    img_train, type_train = data_augmentation(imgs[:750], types[:750])
    img_valid, type_valid = data_augmentation(imgs[750:], types[750:])
    img_train, type_train = imgs[:750], types[:750]
    img_valid, type_valid = imgs[750:], types[750:]

    # TODO : class weight 算出処理
    # w_data = pd.concat([types_df['Type1'][:750],types_df['Type2'][:750]]).value_counts(normalize=True,sort=True)
    # class_weight = pd.Series(w_data.values[0] / w_data.values, index=w_data.index.map(type_dict)).to_dict()

    model = generate_model()

    # JSON形式でモデルを保存
    json_string = model.to_json()
    open(MODEL_FILE_PATH, 'w').write(json_string)
    #　初期ウエイトの保存
    model.save_weights(INIT_WEIGHT_FILE_PATH, overwrite=True)

    # バリデーションロスが下がれば、エポックごとにモデルを保存
    cp_cb = ModelCheckpoint(filepath=BEST_WEIGHT_FILE_PATH,
                            monitor='val_loss',
                            verbose=1,
                            save_best_only=True,
                            mode='auto')

    # バリデーションロスが５エポック連続で上がったら、ランを打ち切る
    es_cb = EarlyStopping(monitor='val_loss',
                          patience=5,
                          verbose=0,
                          mode='auto')

    history = History()

    model.fit_generator(generator=generator(x=img_train,
                                            y=type_train,
                                            batch_size=BATCH_SIZE),
                        steps_per_epoch=len(img_train)//BATCH_SIZE,
                        validation_data=generator(x=img_valid,
                                                  y=type_valid,
                                                  batch_size=BATCH_SIZE),
                        validation_steps=len(img_valid)//BATCH_SIZE,
#                         class_weight=class_weight,
                        verbose=1,
                        epochs=EPOCHS,
                        callbacks=[history,cp_cb,es_cb])


def predict(image):
    # 学習済みモデルとパラメータの呼び出し
    json_string = open(MODEL_FILE_PATH, 'r').read()
    model = model_from_json(json_string)
    model.load_weights(BEST_WEIGHT_FILE_PATH)

    img =  np.array([image]).astype('float32') / 255.0

    type_dict = dataset.type_dict
    type_labels = {v: k for k, v in type_dict.items()}

    # 予測
    pred=model.predict(img, batch_size=1)

    # 予測結果表示
    pred_first_type = type_labels[np.argmax(pred[0])]
    pred[0][np.argmax(pred[0])] = 0
    pred_seconds_type = type_labels[np.argmax(pred[0])]
    print(pred_first_type, pred_seconds_type)


if __name__ == '__main__':
    import os
    import tensorflow as tf
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['KMP_WARNINGS'] = 'off'
    tf.logging.set_verbosity(tf.logging.ERROR)

    args = sys.argv
    if len(args) >= 2:
        if args[1] == 'fit':
            # Load data.
            raw_images, (type1, type2) = dataset.load_data()

            # モデルに合う様に変換
            images = []
            for _img in raw_images:
                img = Image.fromarray(_img, 'RGBA')
                img = img.convert('RGB')
                img = img.resize((64,64))
                images.append(np.array(img))
            images = np.array(images)

            # Run fit.
            fit(images, type1, type2)

        elif args[1] == 'predict':
            # Load image.
            image_path = args[2]
            img = Image.open(image_path)
            img = img.convert('RGB')
            img = img.resize((64,64))
            image = np.array(img)

            # Run predict.
            predict(image)
    else:
        print('引数を指定してください．')
