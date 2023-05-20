# onnx_vc6

## インストール方法
```
pip3 install numpy
pip3 install scipy
apt update && apt install python3-opencv

# onnx
git clone https://github.com/onnx/onnx
cd onnx
git checkout v1.1.0
pip3 install -e .

#onnxruntime
git clone https://github.com/nknytk/built-onnxruntime-for-raspberrypi-linux
pip3 install wheels/{自分の環境に合うwheelファイル}
```

## 実行
[こちら](https://drive.google.com/drive/u/2/folders/1LGD9YBfyXGgZmZcXCYYNFqIRvcbwPytl)からVitのonnxをダウンロード・models/に配置。  
このモデルは[OpenMMLab](https://github.com/open-mmlab/mmpretrain/tree/main/configs/vision_transformer)の`vit-base-p32_in21k-pre_3rdparty_in1k-384px`をonnxに変換＆onnx-simplifierで最適化したものです。  

```
python3 test_run.py
```



