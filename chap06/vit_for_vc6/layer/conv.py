import onnx
import numpy as np
from .layer_base import Layer
class Conv(Layer):
    def __init__(self,model,node,tensor_data,use_gpu=False,drv=None):
        super().__init__(model,node,tensor_data,use_gpu)
        # self.input_name = node.input[0]
        # self.weight_name = node.input[1]
        # self.bias_name = None
        # if len(node.input) >=3:
        #     self.bias_name = node.input[2]
        # self.attrs = {attr.name: onnx.helper.get_attribute_value(attr) for attr in node.attribute}
        # self.output_name = node.output[0]
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
        assert input_data.ndim == 4, "Input data must be 4-dimensional"
        assert weights.ndim == 4, "Weights must be 4-dimensional"
        if bias is not None:
            assert bias.ndim == 1, "Bias must be 1-dimensional"
        batch_size, in_channels, in_height, in_width = input_data.shape
        out_channels, _, kernel_height, kernel_width = weights.shape
        
        _,_,pad_height, pad_width = padding
        stride_height, stride_width = stride

        # パディングを適用します。
        if pad_height > 0 or pad_width > 0:
            input_data = np.pad(
                input_data,
                pad_width=((0, 0), (0, 0), (pad_height, pad_height), (pad_width, pad_width)),
                mode='constant',
                constant_values=0
            )

        # 畳み込みの出力サイズを計算します。
        out_height = (in_height - kernel_height + 2 * pad_height) // stride_height + 1
        out_width = (in_width - kernel_width + 2 * pad_width) // stride_width + 1
        # 出力テンソルを初期化します。
        output = np.zeros((batch_size, out_channels, out_height, out_width))
        # 入力データの各要
        for b in range(batch_size):
            for c in range(out_channels):
                for h in range(0, in_height, stride_height):
                    for w in range(0, in_width, stride_width):
                        h_idx = int(h / stride_height)
                        w_idx = int(w / stride_width)
                        output[b, c, h_idx, w_idx] = np.sum(
                            input_data[b, :, h : h + kernel_height,
                                       w : w + kernel_width] * weights[c]
                        )
        # バイアスがある場合はそれを加えます。
        if bias is not None:
            output += bias.reshape(1, out_channels, 1, 1)
        if self.use_gpu:
            self.tensor_data[self.output_name][:]=output
        else:
            self.tensor_data[self.output_name]=output

