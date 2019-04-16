import numpy as np

#utility functions for padding
def utility_pad(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 10)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value
    return vector

def apply_padding(inp_layer, pad_width = 1):
    return np.pad(inp_layer, pad_width, utility_pad, padder = 0)

#kernel_size is the flat2d size of the kernel. No need to pass the number of kernel layers. Eg - (3x3), (11x11) so pass 3, 11
#inp_layer shape is (batch, row, col, channel)
def conv_indices(inp_layer, kernel_size, stride = 1, padding = 0):
    #check for padding
    C, H, W  = inp_layer.shape[0], inp_layer.shape[1], inp_layer.shape[2]
    
    if(padding != 0):
        inp_layer = apply_padding(inp_layer, padding)
        inp_layer = inp_layer[1:inp_layer.shape[0]-1]

    out_height = int((H + 2*padding - kernel_size)/stride) + 1
    out_width = int((W + 2*padding - kernel_size)/stride) + 1

    #base vectors for index generation
    colm = np.tile(np.arange(kernel_size), kernel_size)
    coli = stride * np.tile(np.arange(out_width), out_height)
    rowm = np.repeat(np.arange(kernel_size), kernel_size)
    rowi = stride * np.repeat(np.arange(out_width), out_height)

    #indices
    col_index = colm.reshape(1, -1) + coli.reshape(-1, 1)
    row_index = rowm.reshape(1, -1) + rowi.reshape(-1, 1)
    
    return inp_layer, col_index, row_index, out_height, out_width, C

#image transformed into columns for matrix multiplication with kernel
def imageToColumn(inp_layer, col_index, row_index, C):
    inp_col = np.transpose(np.squeeze(np.concatenate((np.vsplit(inp_layer[:, row_index, col_index], C)), axis = 2)))
    return inp_col

#kernel is to be passed with the following shape params - (Kx, Ky, Kdepth, Knum)
#internally it works with (Knum, Kdepth, Kx, Ky) size for kernel
def kernelToRow(inp_layer, kernel, bias, out_height, out_width):
    Knum, Kdepth, Kx, Ky = kernel.shape[0], kernel.shape[1], kernel.shape[2], kernel.shape[3] 
    kernel = np.reshape(kernel, (-1, Kdepth*Kx*Ky))
    conv_out = (np.matmul(kernel, inp_layer) + bias)
    conv_out = conv_out.reshape(-1, out_height, out_width)
    return conv_out

def activation_func(conv_out, activation_type = "relu"):
    if(activation_type == "relu"):
        conv_out = np.where(conv_out < 0, 0, conv_out)
    else:
        conv_out = 1/(1 + np.exp(-conv_out))
    return conv_out

def convolution2d(inp_layer, kernel, bias, stride = 1, padding = 0, activation_type = "relu"):
    #important to convert keras filter shape to a meaningful numpy shape
    kernel = np.transpose(kernel, axes  = (3,2,0,1))
    padded_inp, col_index, row_index, out_height, out_width, C = conv_indices(inp_layer, kernel.shape[2])
    image_col = imageToColumn(padded_inp, col_index, row_index, C)
    conv_out = kernelToRow(image_col, kernel, bias, out_height, out_width)
    conv_out = activation_func(conv_out, activation_type)
    return conv_out

if __name__ == "__main__":
    kernel = np.load("layer_weights.npy")
    inp_layer = np.random.randint(3, size=(2, 4, 4))
    print(inp_layer)
    kernel = np.random.randint(2, size = (4, 2, 3, 3))
    print(kernel)
    padded_inp, col_index, row_index, out_height, out_width, C = conv_indices(inp_layer, kernel.shape[2])
    image_col = imageToColumn(padded_inp, col_index, row_index, C)
    print(image_col)
    conv_out = kernelToRow(image_col, kernel, out_height, out_width)
    print(conv_out)