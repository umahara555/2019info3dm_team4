import pandas as pd
from PIL import Image
import numpy as np
import glob
from matplotlib import pylab as plt
from multiprocessing import Pool

type_dict = {
    "Normal": 0,
    "Fire": 1,
    "Water": 2,
    "Electric": 3,
    "Grass": 4,
    "Ice": 5,
    "Fighting": 6,
    "Poison": 7,
    "Ground": 8,
    "Flying": 9,
    "Psychic": 10,
    "Bug": 11,
    "Rock": 12,
    "Ghost": 13,
    "Dragon": 14,
    "Dark": 15,
    "Steel": 16,
    "Fairy": 17
}


def _read_image(path):
    img = Image.open(path)
    img = img.resize((64, 64))
    return np.array(img)


def _read_images():
    image_paths = glob.glob("../../data/images/*")
    image_paths.sort()

    with Pool() as p:
        arr = p.map(_read_image, image_paths)

    return arr


def load_data():
    # load images
    images = np.array(_read_images())

    # load type
    df = pd.read_csv('../../data/Pokemon.csv', sep=',')
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

    X = images
    Y = (type1, type2)
    return X, Y
