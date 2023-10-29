# vit_for_vc6

## インストール方法
```
pip3 install -r requirements.txt
apt update && apt install python3-opencv

```

## 実行
[こちら](https://drive.google.com/drive/u/2/folders/1LGD9YBfyXGgZmZcXCYYNFqIRvcbwPytl)からVitのonnxをダウンロード・models/に配置。  
このモデルは[OpenMMLab](https://github.com/open-mmlab/mmpretrain/tree/main/configs/vision_transformer)の`vit-base-p32_in21k-pre_3rdparty_in1k-384px`をonnxに変換＆onnx-simplifierで最適化したものです。  

```
python3 run.py
```



