# loopless CNN

Numpy implementation of the convolution layer and pooling layer(max, min, avg etc). Works with multi-channel input with padding and n-kernel layers. 

**Pending improvements**
* implement batch image convolutions
* implement padded input path

**Note**

This implementation is inspired from the matrix multiplication techniques laid out in [Stanford's CS231n, CNNs for VR](http://cs231n.github.io/convolutional-networks/) website. One should note that there is always a compromise in terms of either speed or memory when optimising algorithms, atleast in most of the cases. iterating through a matrix with special constraints always requires tiling up the indices, which is a memory intensive process if the matrix is huge. But we benefit from the optimised matrix multiplication interface numpy uses, which is [ATLAS](https://en.wikipedia.org/wiki/Automatically_Tuned_Linear_Algebra_Software), an open-source implementation of [BLAS](https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms) in C.


