from onnx import numpy_helper
import onnx
class Layer:
    def __init__(self,model,node,tensor_data,use_gpu=False):
        self.tensor_data=tensor_data
        self.node=node
        self.use_gpu=use_gpu
        self.input_name=node.input
        self.attrs = {attr.name: onnx.helper.get_attribute_value(attr) for attr in self.node.attribute}
        self.output_name = self.node.output[0]
        self.init_names = {tensor.name for tensor in model.graph.initializer}
    def alloc_gpu_for_initializer(self,tensor_name):
        if tensor_name in self.init_names:
            cpu_w = self.tensor_data[tensor_name]
            self.tensor_data[tensor_name] = self.drv.alloc(cpu_w.shape,cpu_w.dtype)
            self.tensor_data[tensor_name][:]=cpu_w
        return self.tensor_data[tensor_name]
        

class Eltwise_Layer(Layer):
    def __init__(self,model,node,tensor_data,use_gpu,drv):
        super().__init__(model,node,tensor_data,use_gpu)
        self.drv=drv        
        self.input_data1 = self.tensor_data[self.input_name[0]]
        if len(self.input_name) >=2:
            self.input_data2 = self.tensor_data[self.input_name[1]]        
        self.output_data = self.tensor_data[self.output_name]
        self.copy_flg1=False
        self.copy_flg2=False        
    def run(self):
        if self.copy_flg1:
            self.input_data1[:] = self.tensor_data[self.input_name[0]]
        if self.copy_flg2:
            self.input_data2[:] = self.tensor_data[self.input_name[1]]
        self.drv.execute(self.code, self.unif.addresses()[0], thread=self.num_qpus)
        
    def set_1input(self,kernel):
        data_size=1

        for i in self.input_data1.shape:
            data_size*=i
        if data_size<=128:
            self.num_qpus=1
        else:
            self.num_qpus=8
        simd_width=16
        qpu_mod = data_size%self.num_qpus
        proc_size=int(data_size/self.num_qpus)
        proc_size_lth = qpu_mod+proc_size
        loop_num_lth = int(proc_size_lth/simd_width)
        loop_num=int(proc_size/simd_width)
        edge_mod_lth = simd_width-proc_size_lth%simd_width
        edge_mod = simd_width-proc_size%simd_width
        # uniform setting
        self.unif = self.drv.alloc(16, dtype='uint32')
        self.unif[0] = self.input_data1.address
        self.unif[1] = self.output_data.address
        self.unif[2] = proc_size
        self.unif[3] = loop_num
        self.unif[4] = edge_mod
        self.unif[5] = loop_num_lth
        self.unif[6] = edge_mod_lth        
        self.code = self.drv.program(kernel, num_qpus=self.num_qpus)
            
    def set_2input(self,kernel):
        #shapeを合わせる
        if self.input_data1.size != self.input_data2.size:
            if self.input_data1.size > self.input_data2.size:
                new_data=self.drv.alloc(self.input_data1.shape,dtype=self.input_data2.dtype)
                if self.input_name[1] in self.init_names:
                    new_data[:]=self.input_data2
                else:
                    #weightではないデータは推論時にコピー
                    self.copy_flg2=True
                self.input_data2=new_data
            else:
                new_data=self.drv.alloc(self.input_data2.shape,dtype=self.input_data1.dtype)
                if self.input_name[0] in self.init_names:
                    new_data[:]=self.input_data1
                else:
                    #weightではないデータは推論時にコピー
                    self.copy_flg1=True
                self.input_data1=new_data
        else:
            for iname in self.input_name:
                if iname in self.init_names:
                    cpu_w = self.tensor_data[iname]
                    self.tensor_data[iname] = self.drv.alloc(cpu_w.shape,cpu_w.dtype)
                    self.tensor_data[iname][:]=cpu_w
            self.input_data1 = self.tensor_data[self.input_name[0]]
            self.input_data2 = self.tensor_data[self.input_name[1]]        
        self.output_data = self.tensor_data[self.output_name]
        data_size=1
        for i in self.input_data1.shape:
            data_size*=i
        if data_size<=128:
            self.num_qpus=1
        else:
            self.num_qpus=8
        simd_width=16
        qpu_mod = data_size%self.num_qpus
        proc_size=int(data_size/self.num_qpus)
        proc_size_lth = qpu_mod+proc_size
        loop_num_lth = int(proc_size_lth/simd_width)
        loop_num=int(proc_size/simd_width)
        edge_mod_lth = simd_width-proc_size_lth%simd_width
        edge_mod = simd_width-proc_size%simd_width
        # uniform setting
        self.unif = self.drv.alloc(16, dtype='uint32')
        self.unif[0] = self.input_data1.address
        self.unif[1] = self.input_data2.address
        self.unif[2] = self.output_data.address
        self.unif[3] = proc_size
        self.unif[4] = loop_num
        self.unif[5] = edge_mod
        self.unif[6] = loop_num_lth
        self.unif[7] = edge_mod_lth        
        self.code = self.drv.program(kernel, num_qpus=self.num_qpus)        
