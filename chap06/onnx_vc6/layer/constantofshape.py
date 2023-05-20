import onnx
from onnx import numpy_helper
import numpy as np
from .layer_base import Layer
class ConstantOfShape():
    def __init__(self,model,node,tensor_data,use_gpu=False,drv=None):
        super().__init__(model,node,tensor_data,use_gpu)
    def run(self):
        input_data = self.tensor_data[self.input_name]
        value=numpy_helper.to_array(self.attrs['value'])
        output = np.full(input_data,value)
        self.tensor_data[self.output_name]=output


