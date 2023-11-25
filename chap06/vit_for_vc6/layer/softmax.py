import onnx
import numpy as np
from .layer_base import Layer
class Softmax(Layer):
    def __init__(self,model,node,tensor_data,use_gpu=False,drv=None):
        super().__init__(model,node,tensor_data,use_gpu)

    def run(self):
        input_data = self.tensor_data[self.input_name[0]]
        e = np.exp(input_data - np.max(input_data, axis=self.attrs['axis'], keepdims=True))
        s = np.sum(e, axis=self.attrs['axis'], keepdims=True)
        output = e /s
        if self.use_gpu:
            self.tensor_data[self.output_name][:]=output
        else:
            self.tensor_data[self.output_name]=output



