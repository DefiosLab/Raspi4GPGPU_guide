from onnx import numpy_helper,shape_inference
from .layer_dict import *
import time
import numpy as np
from videocore6.driver import Driver
class Inf:
    def __init__(self,model,use_gpu=False):
        # self.model=model
        self.model = shape_inference.infer_shapes(model)
        self.tensor_data={}
        self.layer=[]
        self.time_dict={}

        #Weightを格納
        self.tensor_data.update({initializer.name: numpy_helper.to_array(initializer) for initializer in self.model.graph.initializer})
        self.create_layer(use_gpu)
        
    def create_layer(self,use_gpu=False):
        self.layer=[]
        if use_gpu:
            self.alloc_gpu()        
        for node in self.model.graph.node:
            #対応する層のインスタンスを作成
            if use_gpu:
                self.layer.append(layer_dict_gpu[node.op_type](self.model,node,self.tensor_data,use_gpu,self.drv))
            else:
                for output_name in node.output:
                    self.tensor_data[output_name]=None                
                self.layer.append(layer_dict[node.op_type](self.model,node,self.tensor_data))
            self.time_dict[node.op_type]=0
        
    def elem_type2numpy(self,dtype_onnx):
        if dtype_onnx == 1:
            dtype_numpy = np.float32
        elif dtype_onnx == 2:
            dtype_numpy = np.uint8
        elif dtype_onnx == 3:
            dtype_numpy = np.int8
        elif dtype_onnx == 4:
            dtype_numpy = np.uint16
        elif dtype_onnx == 5:
            dtype_numpy = np.int16
        elif dtype_onnx == 6:
            dtype_numpy = np.int32
        elif dtype_onnx == 7:
            dtype_numpy = np.int64
        elif dtype_onnx == 9:
            dtype_numpy = np.bool
        elif dtype_onnx == 10:
            dtype_numpy = np.float16
        elif dtype_onnx == 11:
            dtype_numpy = np.double
        else:
            raise ValueError(f"Unknown ONNX data type: {dtype_onnx}")
        return dtype_numpy

    def search_tensorinfo(self,name):
        for value_info in self.model.graph.value_info:            
            if value_info.name == name:
                shape_onnx = value_info.type.tensor_type.shape
                shape_tuple = tuple(dim.dim_value if dim.HasField("dim_value") else None for dim in shape_onnx.dim)             
                dtype = self.elem_type2numpy(value_info.type.tensor_type.elem_type)
                break
        else:
            for model_output in self.model.graph.output:
                if model_output.name == name:
                    shape_onnx = model_output.type.tensor_type.shape
                    shape_tuple = tuple(dim.dim_value if dim.HasField("dim_value") else None for dim in shape_onnx.dim)             
                    dtype = self.elem_type2numpy(model_output.type.tensor_type.elem_type)
            for model_input in self.model.graph.input:
                if model_input.name == name:
                    shape_onnx = model_input.type.tensor_type.shape
                    shape_tuple = tuple(dim.dim_value if dim.HasField("dim_value") else None for dim in shape_onnx.dim)             
                    dtype = self.elem_type2numpy(model_input.type.tensor_type.elem_type)
                    
        return shape_tuple,dtype
    
    def alloc_gpu(self):
        self.drv = Driver(data_area_size=1024*1024*1024+1024*1024*512)
        for node in self.model.graph.node:
            for output_name in node.output:
                shape,dtype = self.search_tensorinfo(output_name)
                self.tensor_data[output_name] = self.drv.alloc(shape,dtype=dtype)
        shape,dtype = self.search_tensorinfo(self.model.graph.input[0].name)
        self.tensor_data[self.model.graph.input[0].name]=self.drv.alloc(shape,dtype=dtype)
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
            max_length = max(len(key) for key in self.time_dict.keys())
            print("Processing time for each layer")
            for k,v in  self.time_dict.items():
                print("{}:{:.2f}msec".format(k.ljust(max_length),v*1000))
        return self.tensor_data[self.model.graph.output[0].name]
