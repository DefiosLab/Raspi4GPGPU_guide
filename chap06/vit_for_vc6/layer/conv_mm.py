import onnx
import numpy as np
from .layer_base import Layer
class Conv_mm(Layer):
    def __init__(self,model,node,tensor_data,use_gpu=False,drv=None):
        super().__init__(model,node,tensor_data,use_gpu)

        self.im2col_w()
    def im2col_w(self):
        weights = self.tensor_data[self.input_name[1]]
        n,c,h,w=weights.shape
        self.col_w=weights.reshape(n,-1)
    def im2col_i(self,inp,inp_c,inp_h,inp_w,out_h,out_w,str_h,str_w,ker_h,ker_w):
        col = np.zeros((inp_c*ker_h*ker_w,out_h*out_w),dtype=np.float32)
        for h in range(0,inp_h,str_h):
            for w in range(0,inp_w,str_w):
                
                ch=int(h/str_h)
                cw=int(w/str_w)
                col[:,ch*out_w+cw]=inp[0,:,h:h+str_h,w:w+str_w].reshape(-1)
        
        return col
    def col2im(self,col,batch,out_ch,out_h,out_w):
        return col.reshape(batch,out_ch,out_h,out_w)
        
    def run(self):
        input_data = self.tensor_data[self.input_name[0]]
        weights = self.tensor_data[self.input_name[1]]
        if len(self.input_name) >=3:
            bias = self.tensor_data[self.input_name[2]]
        else:
            bias = None
        if 'pads' in self.attrs:
            padding = self.attrs['pads']
        else:
            padding = [0,0,0,0]
        if 'strides' in self.attrs:
            stride = self.attrs['strides']
        else:
            stride = [1,1]
        # assert input_data.ndim == 4, "Input data must be 4-dimensional"
        # assert weights.ndim == 4, "Weights must be 4-dimensional"
        if bias is not None:
            assert bias.ndim == 1, "Bias must be 1-dimensional"
        batch_size, in_channels, in_height, in_width = input_data.shape
        out_channels, _, kernel_height, kernel_width = weights.shape
        out_height = int(in_height/kernel_height)
        out_width = int(in_width/kernel_width)
        _,_,pad_height, pad_width = padding
        stride_height, stride_width = stride
        col_i = self.im2col_i(input_data,in_channels,in_height,in_width,
                      out_height,out_width,
                      stride_height,stride_width,kernel_height,kernel_width)
        col_o = self.col_w @col_i

        output = self.col2im(col_o,batch_size,out_channels,out_height,out_width)
        if bias is not None:
            output += bias.reshape(1, out_channels, 1, 1)
        if self.use_gpu:
            self.tensor_data[self.output_name][:]=output
        else:
            self.tensor_data[self.output_name]=output


