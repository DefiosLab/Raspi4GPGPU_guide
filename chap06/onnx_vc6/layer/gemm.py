import onnx
import numpy as np
from .layer_base import Layer
class Gemm(Layer):
    def __init__(self,model,node,tensor_data,use_gpu=False,drv=None):
        super().__init__(model,node,tensor_data,use_gpu)
    def run(self):
        input_data1 = self.tensor_data[self.input_name[0]] 
        input_data2 = self.tensor_data[self.input_name[1]]
        if 'transA' in self.attrs:
            transA = self.attrs['transA']
        else:
            transA = 0
            
        if 'transB' in self.attrs:
            transB = self.attrs['transB']
        else:
            transB = 0
        if 'alpha' in self.attrs:
            alpha = self.attrs['alpha']
        else:
            alpha = 1
            
        if 'beta' in self.attrs:
            beta = self.attrs['beta']
        else:
            beta = 1
        
        input_data1 = input_data1.T if transA else input_data1
        input_data2 = input_data2.T if transB else input_data2
        output = alpha * np.dot(input_data1,input_data2)
        if len(self.input_name)==3:
            input_data3 = self.tensor_data[self.input_name[2]]
            output += beta * input_data3
            
        if self.use_gpu:
            self.tensor_data[self.output_name][:]=output
        else:
            self.tensor_data[self.output_name]=output
