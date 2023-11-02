import vit_for_vc6
import onnxruntime
import onnx
import numpy as np
import cv2
import time
import sys
import json
import urllib.request
import os
def download_imagenet_labels(filename='imagenet_labels.json'):
    url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            labels = json.load(f)
    else:
        with urllib.request.urlopen(url) as response:
            labels = json.loads(response.read())
            # ラベルをローカルファイルに保存
            with open(filename, 'w') as f:
                json.dump(labels, f)
                
    return labels

def crop_center_square(image):
    h,w=image.shape[:2]
    min_dim = min(h,w)
    top = (h-min_dim)//2
    bottom=(h+min_dim)//2
    left=(w-min_dim)//2
    right=(w+min_dim)//2
    cropped_image=image[top:bottom,left:right]
    return cropped_image


def preprocess(img):
    #前処理
    mean=np.array([0.5,0.5,0.5])
    std=np.array([0.5,0.5,0.5])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = crop_center_square(img)
    img = cv2.resize(img,(384,384))
    img = img.astype(np.float32) / 255.0
    img = (img-mean)/std
    img = np.transpose(img, (2, 0, 1))
    img = img[np.newaxis,:,:,:]
    img=img.astype(np.float32)
    return img

print("Loading model...",end='')
model_path = 'models/vision_transformerb32.onnx'
model = onnx.load(model_path)
print("Done")


img = cv2.imread('Parrots.jpg')
img = preprocess(img)

#vit_for_vc6
print("Initializing...",end='')
sys.stdout.flush()
vfv_sess = vit_for_vc6.Inf(model,use_gpu=True)

#onnxruntime
ort_sess = onnxruntime.InferenceSession(model_path)
input_name = ort_sess.get_inputs()[0].name
print("Done")



print("Inference start")
#onnxruntimeでの推論
st = time.time()
result_ort = ort_sess.run(None, {input_name: img})[0]
ed = time.time()
ort_time=ed-st

#vit_for_vc6での推論
st = time.time()
result_vfv = vfv_sess.run(img,profile=True)
ed = time.time()
vfv_time=ed-st

imagenet_labels = download_imagenet_labels()
pred = imagenet_labels[np.argmax(result_vfv)]
print("Predicted label : {}".format(pred))
print("onnxruntime time:{:.2f}sec".format(ort_time))
print("vc6 time:{:.2f}sec".format(vfv_time))

#計算誤差
print('maximum absolute error : {:.4e}'.format(float(np.max(np.abs(result_vfv-result_ort)))))
print('maximum relative error : {:.4e}'.format(float(np.max(np.abs(result_vfv-result_ort)/result_vfv))))
