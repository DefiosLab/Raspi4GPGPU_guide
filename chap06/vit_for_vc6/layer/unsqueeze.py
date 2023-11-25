import onnx
import numpy as np
from .layer_base import Layer
class Unsqueeze(Layer):
    def __init__(self,model,node,tensor_data,use_gpu=False,drv=None):
        super().__init__(model,node,tensor_data,use_gpu)
    def run(self):
        input_data = self.tensor_data[self.input_name[0]]    
        output = np.expand_dims(input_data, axis=self.attrs['axes'])
        if self.use_gpu:
            self.tensor_data[self.output_name][:]=output
        else:
            self.tensor_data[self.output_name]=output



