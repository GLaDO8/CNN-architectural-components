import numpy as np

def upsampling_layer(input, size = (2,2), upsampling_type = "nearest_neighbour"):
    return np.repeat(np.repeat(input, size[0], axis = 1), size[1], axis = 2)


