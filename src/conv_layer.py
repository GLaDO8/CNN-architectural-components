import numpy as np

def index_generator(inp_layer, kernel_size, stride = 1,  padding = 0):
    H, W, C = inp_layer.shape[1], inp_layer.shape[0], inp_layer.shape[2]
    (kernel2d, kernel2d, kernel_depth) = kernel_size
    out_height = int((H + 2*padding - kernel2d)/stride) + 1
    out_width = int((W + 2*padding - kernel2d)/stride) + 1

    #base vectors for index generation
    colm = np.tile(np.arange(kernel2d), kernel2d)
    coli = stride * np.tile(np.arange(out_width), out_height)
    rowm = np.repeat(np.arange(kernel2d), kernel2d)
    rowi = stride * np.repeat(np.arange(out_width), out_height)

    #index generation
    col_index = colm.reshape(1, -1) + coli.reshape(-1, 1)
    row_index = rowm.reshape(1, -1) + rowi.reshape(-1, 1)

    return col_index, row_index, out_height, out_width

def convolve(inp_layer, col_index, row_index, kernel_size, out_height, out_width):
    (kernel2d, kernel2d, kernel_depth) = kernel_size
    conv_out = np.matmul(kernel.reshape(1, -1), np.transpose(inp_layer[row_index, col_index]/(kernel2d*kernel2d))).reshape(out_height, out_width)
    return conv_out
