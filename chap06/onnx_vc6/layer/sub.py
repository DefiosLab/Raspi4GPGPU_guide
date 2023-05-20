import onnx
import numpy as np
from .layer_base import Layer
class Sub(Layer):
    def __init__(self,model,node,tensor_data,use_gpu=False,drv=None):
        super().__init__(model,node,tensor_data,use_gpu)
    def run(self):
        input_data1 = self.tensor_data[self.input_name[0]] 
        input_data2 = self.tensor_data[self.input_name[1]] 
        output = input_data1-input_data2
        if self.use_gpu:
            self.tensor_data[self.output_name][:]=output
        else:
            self.tensor_data[self.output_name]=output



