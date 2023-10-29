from .layer import * 
layer_dict = {"Conv":Conv_mm,"Shape":Shape, "Gather":Gather, "Unsqueeze":Unsqueeze, "Concat":Concat,
              "Reshape":Reshape, "ConstantOfShape":ConstantOfShape, "Transpose":Transpose, "Add":Add,
              "Sub":Sub, "ReduceMean":ReduceMean, "Pow":Pow, "Sqrt":Sqrt, "Div":Div, "Mul":Mul, "MatMul":MatMul,
              "Softmax":Softmax,"Erf":Erf,"Gemm":Gemm} 

layer_dict_gpu = {"Conv":Conv_mm,"Shape":Shape, "Gather":Gather, "Unsqueeze":Unsqueeze, "Concat":Concat,
              "Reshape":Reshape, "ConstantOfShape":ConstantOfShape, "Transpose":Transpose, "Add":Add_gpu,
              "Sub":Sub_gpu, "ReduceMean":ReduceMean, "Pow":Pow_gpu, "Sqrt":Sqrt_gpu, "Div":Div_gpu, "Mul":Mul_gpu, "MatMul":MatMul_gpu,
              "Softmax":Softmax,"Erf":Erf,"Gemm":Gemm} 

# layer_dict_gpu = {"Conv":Conv_mm,"Shape":Shape, "Gather":Gather, "Unsqueeze":Unsqueeze, "Concat":Concat,
#               "Reshape":Reshape, "ConstantOfShape":ConstantOfShape, "Transpose":Transpose, "Add":Add_gpu,
#               "Sub":Sub_gpu, "ReduceMean":ReduceMean, "Pow":Pow_gpu, "Sqrt":Sqrt_gpu, "Div":Div_gpu, "Mul":Mul_gpu, "MatMul":MatMul,
#               "Softmax":Softmax,"Erf":Erf,"Gemm":Gemm} 
