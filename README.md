# ポケモン画像のタイプ予測

## 概要

知能情報実験3 Group4 の成果物である．  
ポケモンの画像データからそのポケモンのタイプの予測を行う．

## 使用したデータセット

- [Pokemon Images](https://www.kaggle.com/dollarakshay/pokemon-images)
- [pokemonData](https://github.com/lgreski/pokemonData)

## 動作環境

- Python 3.x
- Tensorflow
- Keras

## セットアップ

ソースのクローン
```
$ git clone https://github.com/umahara555/2019info3dm_team4.git
```

上記からダウンロードしたデータセットを以下の様に配置する．
```
data
├── Pokemon.csv
└── images
```

## 実行方法

```
# 多ラベル分類
$ python PokemonMultiLabelClassification.py fit
$ python PokemonMultiLabelClassification.py predict <filepath>
```

```
# 多クラス分類
$ python MultiClass.py
```

```
# 多クラス分類
$ python MultiClassClassification.py
```

```
# ランダムフォレスト
$ python RandomForest.py
```

## Author

氏名 : 上原由宇駆  
連絡先 : e175732@ie.u-ryukyu.ac.jp

氏名 : 上原一真  
連絡先 : e175740@ie.u-ryukyu.ac.jp

氏名 : 白石貴祥  
連絡先 : e175761@ie.u-ryukyu.ac.jp

## Licence

[MIT Licence](https://github.com/umahara555/2019info3dm_team4/blob/master/LICENSE)
