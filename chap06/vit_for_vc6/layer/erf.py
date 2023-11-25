import onnx
import numpy as np
from scipy.special import erf
from .layer_base import Layer
class Erf(Layer):
    def __init__(self,model,node,tensor_data,use_gpu=False,drv=None):
        super().__init__(model,node,tensor_data,use_gpu)
    def run(self):
        x = self.tensor_data[self.input_name[0]]
        output = erf(x)
        
        if self.use_gpu:
            self.tensor_data[self.output_name][:]=output
        else:
            self.tensor_data[self.output_name]=output
