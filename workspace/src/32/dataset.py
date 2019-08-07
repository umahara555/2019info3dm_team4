import pandas as pd
from PIL import Image
import numpy as np
import glob
from matplotlib import pylab as plt
from multiprocessing import Pool
import time

type_dict = {
    "Normal":0,
    "Fire":1,
    "Water":2,
    "Electric":3,
    "Grass":4,
    "Ice":5,
    "Fighting":6,
    "Poison":7,
    "Ground":8,
    "Flying":9,
    "Psychic":10,
    "Bug":11,
    "Rock":12,
    "Ghost":13,
    "Dragon":14,
    "Dark":15,
    "Steel":16,
    "Fairy":17
}

def _read_image(path):
    img = Image.open(path)
    img = img.resize((64,64))
    return np.array(img)

def _read_images(path):
    image_paths = glob.glob(path+'images/*')
    image_paths.sort()

    images_length = len(image_paths)
    images = []
    with Pool() as p:
        for image, i in zip(p.imap(_read_image, image_paths), range(images_length)):
            print(f'\rloading images ... {i}/{images_length}', end='')
            images.append(image)
        print(f'\rloading images ... {images_length}/{images_length} done')
    return np.array(images)

def _read_types(path):
    print(f'\rloading types data ... ', end='')
    df = pd.read_csv(path+'Pokemon.csv', sep=',')
    df.drop_duplicates(subset='Number', inplace=True)
    df.reset_index(inplace=True, drop=True)
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

def load_data(path='/workspace/data/'):
    # load images
    images = _read_images(path)

    # load type
    type1, type2 = _read_types(path)
    # df = pd.read_csv(path+'Pokemon.csv', sep=',')
    # df.drop_duplicates(subset='Number', inplace=True)
    # df.reset_index(inplace=True, drop=True)
    # ind = df[df['Type2'].isnull()]['Type2'].index
    # df.iloc[ind, 3] = df.iloc[ind, 2]
    #
    # df_1 = df["Type1"][:801]
    # df_1 = df_1.map(type_dict)
    # type1 = df_1.values
    #
    # df_2 = df["Type2"][:801]
    # df_2 = df_2.map(type_dict)
    # type2 = df_2.values

    # X = images
    # Y = (type1, type2)
    return images, (type1, type2)
