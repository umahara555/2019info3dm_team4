# 2019info3dm_team4

## Docker関係の操作

基本的に`2019info3dm_team4`ディレクトリ内で行う

### 起動

```
$ docker-compose up -d
```

### 停止

```
$ docker-compose down
```

### JupyterLabのアドレス確認

```
$ docker-compose logs
# 中略
http://(hogehoge or 127.0.0.1):8888/?token=XXXXXXX
```

表示されたアドレスを次の様な形にして，ブラウザでアクセスする
`http://127.0.0.1:8888/?token=XXXXXXX`


### データセットの配置

データセットを`workspace/data/`下に置く

### ソースコード

ソースコードは`workspace/src/`下に置く
