from onnx import numpy_helper
from .layer_dict import *
import time
import numpy as np
from videocore6.driver import Driver
class Inf:
    def __init__(self,model,use_gpu=False,input_shape=None):
        self.model=model
        self.tensor_data={}
        self.layer=[]
        self.time_dict={}

        #Weightを格納
        self.tensor_data.update({initializer.name: numpy_helper.to_array(initializer) for initializer in self.model.graph.initializer})
        self.create_layer()
        
        if use_gpu:
            assert input_shape is not None, "Error:Please input the shape of the dummy input."
            self.alloc_gpu(input_shape)
            self.create_layer(use_gpu)
        
    def create_layer(self,use_gpu=False):
        self.layer=[]
        for node in self.model.graph.node:
            #対応する層のインスタンスを作成
            if use_gpu:
                self.layer.append(layer_dict_gpu[node.op_type](self.model,node,self.tensor_data,use_gpu,self.drv))
            else:
                for output_name in node.output:
                    self.tensor_data[output_name]=None                
                self.layer.append(layer_dict[node.op_type](self.model,node,self.tensor_data))
            self.time_dict[node.op_type]=0
        

    def alloc_gpu(self,input_shape):
        self.drv = Driver(data_area_size=1024*1024*1024+1024*1024*512)
        dammy_inp = np.random.rand(*input_shape).astype(np.float32)
        #推論で得たShapeをもとにalloc
        self.run(dammy_inp)
        for node in self.model.graph.node:
            for output_name in node.output:
                cpu_out = self.tensor_data[output_name]
                self.tensor_data[output_name] = self.drv.alloc(cpu_out.shape,dtype=cpu_out.dtype)
        cpu_inp = self.tensor_data[self.model.graph.input[0].name]
        self.tensor_data[self.model.graph.input[0].name]=self.drv.alloc(cpu_inp.shape,dtype=cpu_inp.dtype)
        
        # for k,v in self.tensor_data.items():
        #     self.tensor_data[k] = self.drv.alloc(v.shape,dtype=v.dtype)
        # for initializer in self.model.graph.initializer:
        #     if numpy_helper.to_array(initializer).shape==():
        #         self.tensor_data[initializer.name] = numpy_helper.to_array(initializer)
        #     else:
        #         self.tensor_data[initializer.name][:] = numpy_helper.to_array(initializer)

    def set_input(self,input_data):
        if self.model.graph.input[0].name in self.tensor_data:
            self.tensor_data[self.model.graph.input[0].name][:]=input_data
        else:

            self.tensor_data[self.model.graph.input[0].name]=input_data
            
    def run(self,input_data,profile=False):
        self.set_input(input_data)
        for i,A in enumerate(self.layer):
            st = time.time()
            A.run()
            ed = time.time()
            self.time_dict[self.model.graph.node[i].op_type]+=ed-st
        if profile:
            for k,v in  self.time_dict.items():
                print("{}:{}msec".format(k,v*1000))
        return self.tensor_data[self.model.graph.output[0].name]
