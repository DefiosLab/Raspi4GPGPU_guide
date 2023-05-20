import onnx
import numpy as np
from .layer_base import Layer
class Shape():
    def __init__(self,model,node,tensor_data,use_gpu=False,drv=None):
        super().__init__(model,node,tensor_data,use_gpu)
    def run(self):
        input_data = self.tensor_data[self.input_name]
        output = np.array(input_data.shape,dtype=np.int64)
        if self.use_gpu:
            self.tensor_data[self.output_name][:]=output
        else:
            self.tensor_data[self.output_name]=output



