import onnx
import numpy as np
from .layer_base import Layer
class Concat(Layer):
    def __init__(self,model,node,tensor_data,use_gpu=False,drv=None):
        super().__init__(model,node,tensor_data,use_gpu)
    def run(self):
        input_data =[]
        for n in self.input_name:
            data = self.tensor_data[n]
            if data.ndim==2:
                data=data.flatten()                
            input_data.append(data)
        output = np.concatenate(input_data, axis=self.attrs['axis'])
        if self.use_gpu:
            self.tensor_data[self.output_name][:]=output
        else:
            self.tensor_data[self.output_name]=output


