from conv_layer import imageToColumn
import numpy as np

def max_pooling(inp_layer, kernel_size, stride = 1, type = "max"):
    inp_col, out_height, out_width = imageToColumn(inp_layer, kernel_size)
    

