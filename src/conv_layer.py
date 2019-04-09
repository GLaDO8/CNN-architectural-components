import numpy as np

#kernel_size is the flat2d size of the kernel. No need to pass the number of kernel layers. Eg - (3x3), (11x11) so pass 3, 11
def imageToColumn(inp_layer, kernel_size, stride = 1, padding = 0):
    H, W, C = inp_layer.shape[1], inp_layer.shape[0], inp_layer.shape[2]
    out_height = int((H + 2*padding - kernel_size)/stride) + 1
    out_width = int((W + 2*padding - kernel_size)/stride) + 1

    #base vectors for index generation
    colm = np.tile(np.arange(kernel_size), kernel_size)
    coli = stride * np.tile(np.arange(out_width), out_height)
    rowm = np.repeat(np.arange(kernel_size), kernel_size)
    rowi = stride * np.repeat(np.arange(out_width), out_height)

    col_index = colm.reshape(1, -1) + coli.reshape(-1, 1)
    row_index = rowm.reshape(1, -1) + rowi.reshape(-1, 1)

    #image_2_col generation
    inp_col = np.concatenate((np.vsplit(inp_layer[:, row_index, col_index], C)), axis = 2)
    return inp_col

def kernel_initialise(kernel_size):
    Knum, Kdepth, Kx, Ky = kernel_size
    return np.random.randint(2, size = (Knum, Kdepth, Kx, Ky))

#kernel is to be passed with the following shape params - (Knum, Kdepth, Kx, Ky)
def kernelToRow(inp_layer, kernel, stride = 1):
    Knum, Kdepth, Kx, Ky = kernel.shape[0], kernel.shape[1], kernel.shape[2], kernel.shape[3] 
    kernel.reshape(-1, Kdepth*Kx*Ky)
    conv_out = np.matmul(kernel, inp_layer)/(Kx*Ky)
    return conv_out
