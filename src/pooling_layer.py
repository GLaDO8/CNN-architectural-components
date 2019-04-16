from conv_layer import conv_indices
import numpy as np

#pooling function uses the same index generator 'conv_indices' from conv_layer
def pooling(inp_layer, kernel_size, stride = 1, type = "max"):
    inp_layer, col_index, row_index, out_height, out_width, C = conv_indices(inp_layer, kernel_size, stride = stride, padding = padding)
    pools = inp_layer[:, row_index, col_index]
    if(type == "avg"):
        pool_out = np.average(pools, axis = 2)
    elif(type == "min"):
        pool_out = np.min(pools, axis = 2)
    else:
        pool_out = np.max(pools, axis = 2)
    
    pool_out = pool_out.reshape(-1, out_height, out_width)
    return pool_out


