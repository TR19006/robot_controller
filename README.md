# 遠隔・ロボットコントローラ
遠隔でロボットを制御するツール
<br>
物体検出機能を導入しているため，ロボットの自立的な制御を組み込むことが可能です．

## 必要な物
1. カメラ内臓のPCまたは，PCおよびPCに接続可能なWebカメラ（ロボットを想定）<br>
Windows 10およびUbuntu 18.04 LTSにて動作確認済みです．<br>

2. Webブラウザを使用可能な端末<br>
WebブラウザはGoogle Chromeを推奨します．<br>

3. ネットワーク環境<br>
VPNを使用・ファイアウォールの特定のポート番号の通信を許可等により，IP接続が可能な環境を構築する必要があります．

## 実行手順
ロボットの認識システムとして，顔検出システムを用いるか，M2Det（物体検出システム）を用いるかで実行方法が異なります．
<br>
また，M2Detを用いる場合は，GPU（CUDA）が使用可能かどうかで実行方法が異なります．
<br>
※ M2Detは最新かつ高精度な物体検出技術を用いているため，GPUを用いないとリアルタイムに検出ができません．

### 環境設定
環境の設定およびコードの実行は，カメラの使用可能なPC上で行います．
1. Pythonの使用可能な環境を用意してください．<br>
Python 3.7.4 にて動作確認済みです．<br>

2. 以下のコマンドで必要なパッケージをインストールしてください．<br>
torch のインストールは，[こちらのリンク](https://pytorch.org/get-started/locally/)に従ってください．

```sh
pip3 install opencv-python
pip3 install flask
pip3 install addict
pip3 install torch
pip3 install torchvision
```

### 顔検出システムを用いたコントローラの実行
1. 以下のコマンドで実行します．
 > python3 run.py

2. ifconfig もしくは ipconfig コマンドで**IPアドレス**を確認し，Webブラウザを使用可能な別の端末のブラウザに以下を入力することで動作します．
```sh
[IPアドレス]:9999
```

### M2Det用いたコントローラの実行
1. [こちらのリンク](https://drive.google.com/file/d/1NM1UDdZnwHwiNDxhcP-nndaWj24m-90L/)より，*m2det512_vgg.pth*をダウンロードし，*./configs*以下に配置してください．

2. 以下のコマンドで実行できます．<br>
GPU（CUDA）が使えない環境の場合は，*./configs/m2det512_vgg.py*のをFalseに変更してください．
 > python3 run.py -m

3. ifconfig もしくは ipconfig コマンドで**IPアドレス**を確認し，Webブラウザを使用可能な別の端末のブラウザに以下を入力することで動作します．
```sh
[IPアドレス]:9999
```
