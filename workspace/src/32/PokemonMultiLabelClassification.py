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

type_dict = {
    "Normal":0, "Fire":1, "Water":2, "Electric":3, "Grass":4, "Ice":5,
    "Fighting":6, "Poison":7, "Ground":8, "Flying":9, "Psychic":10, "Bug":11,
    "Rock":12, "Ghost":13, "Dragon":14, "Dark":15,"Steel":16, "Fairy":17
}

data_dir = '/workspace/data/'

def load_images():
    images_path = data_dir + 'images2/*'
    image_paths = glob.glob(images_path)
    image_paths.sort()
    images_length = len(image_paths)
    raw_images = []
    for image_path,i in zip(image_paths, range(images_length)):
        img = Image.open(image_path)
        raw_images.append(np.array(img))
        print(f'\rloading images ... {i}/{images_length}', end='')
    print(f'\rloading images ... {images_length}/{images_length} done')
    raw_images = np.array(raw_images)
    return raw_images


def load_types():
    types_file_path = data_dir + 'Pokemon.csv'

    print(f'\rloading types data ... ', end='')
    types_df = pd.read_csv(types_file_path, sep=',')
    types_df.drop_duplicates(subset='Number', inplace=True)
    types_df.reset_index(inplace=True, drop=True)

    # Type2がない場合,Type1で補完する
    df = types_df.copy(deep=True)
    ind = df[df['Type2'].isnull()]['Type2'].index
    df.iloc[ind, 3] = df.iloc[ind, 2]

    df_1 = df["Type1"][:801]
    df_1 = df_1.map(type_dict)
    type1 = df_1.values

    df_2 = df["Type2"][:801]
    df_2 = df_2.map(type_dict)
    type2 = df_2.values

    print(f'\rloading types data ... done')

    return type1, type2


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


def generate_model(img_rows,img_cols,img_channels,nb_classes):
    model = Sequential()
    model.add(Conv2D(32, (5, 5),
                     padding='same',
                     activation='relu',
                     kernel_regularizer=regularizers.l2(0.001),
                     input_shape=(img_rows,img_cols,img_channels)))
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
    model.add(Dense(nb_classes,
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
    # 画像の形状情報
    img_rows = 64
    img_cols = 64
    img_channels = 3
    nb_classes = 18

    # fit parameter
    batch_size = 32
    epochs = 2

    # 前処理
    images = raw_images.astype('float32') / 255.0
    type1_c = utils.to_categorical(type1, nb_classes)
    type2_c = utils.to_categorical(type2, nb_classes)
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

    model = generate_model(img_rows, img_cols, img_channels, nb_classes)

    # JSON形式でモデルを保存
    json_string = model.to_json()
    open('./pokemon_cnn.json', 'w').write(json_string)
    #　初期ウエイトの保存
    model.save_weights('./pokemon_cnn_init_weight.hdf5', overwrite=True)

    # バリデーションロスが下がれば、エポックごとにモデルを保存
    cp_cb = ModelCheckpoint(filepath='./pokemon_cnn_best_weight1.hdf5',
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
                                            batch_size=batch_size),
                        steps_per_epoch=len(img_train)//batch_size,
                        validation_data=generator(x=img_valid,
                                                  y=type_valid,
                                                  batch_size=batch_size),
                        validation_steps=len(img_valid)//batch_size,
#                         class_weight=class_weight,
                        verbose=1,
                        epochs=epochs,
                        callbacks=[history,cp_cb,es_cb])


def predict(image):
    # 画像の形状情報
    img_rows = 64
    img_cols = 64
    img_channels = 3
    nb_classes = 18

    # 学習済みモデルとパラメータの呼び出し
    json_string = open('./pokemon_cnn.json', 'r').read()
    model = model_from_json(json_string)
    model.load_weights('./pokemon_cnn_best_weight1.hdf5')

    img =  np.array([image]).astype('float32') / 255.0

    type_labels = {v: k for k, v in type_dict.items()}

    # 予測
    pred=model.predict(img, batch_size=1)

    print(type_labels[np.argmax(pred[0])])


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
            raw_images = load_images()
            type1, type2 = load_types()
            # Run fit.
            fit(raw_images, type1, type2)
        elif args[1] == 'predict':
            # Load image.
            image_path = args[2]
            img = Image.open(image_path)
            image = np.array(img)
            # Run predict.
            predict(image)
    else:
        print('引数を指定してください．')
