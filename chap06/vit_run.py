import vit_for_vc6
import onnxruntime
import onnx
import numpy as np
import cv2
import time
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
model_path = 'models/vision_transformerb32.onnx'
model = onnx.load(model_path)
img = cv2.imread('Parrots.jpg')
img = preprocess(img)

#vit_for_vc6
ov_engine = vit_for_vc6.Inf(model,use_gpu=True)

#onnxruntime
session = onnxruntime.InferenceSession(model_path)
input_name = session.get_inputs()[0].name

#onnxruntimeでの推論
st = time.time()
result_ort = session.run(None, {input_name: img})[0]
ed = time.time()
ort_time=ed-st

#vit_for_vc6での推論
st = time.time()
result_ov = ov_engine.run(img,profile=True)
ed = time.time()
numpy_time=ed-st
print("Maximum output value of the model : {}".format(np.argmax(result_ov)))
print("ort time:{}".format(ort_time))
print("numpy time:{}".format(numpy_time))

#計算誤差
print('maximum absolute error : {:.4e}'.format(float(np.max(np.abs(result_ov-result_ort)))))
print('maximum relative error : {:.4e}'.format(float(np.max(np.abs(result_ov-result_ort)/result_ov))))
