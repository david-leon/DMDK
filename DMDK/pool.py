# coding:utf-8
# <Descriptions>
# Created   :  mm, dd, yyyy
# Revised   :  mm, dd, yyyy
# All rights reserved
#------------------------------------------------------------------------------------------------
__author__ = 'dawei.leng'

from base import *
import backend
from utils import *

def pool_output_length(input_length, pool_size, stride, pad, ignore_border):
    """
    Compute the output length of a pooling operator
    along a single dimension.

    Parameters
    ----------
    input_length : integer
        The length of the input in the pooling dimension
    pool_size : integer
        The length of the pooling region
    stride : integer
        The stride between successive pooling regions
    pad : integer
        The number of elements to be added to the input on each side.
    ignore_border: bool
        If ``True``, partial pooling regions will be ignored.
        Must be ``True`` if ``pad != 0``.

    Returns
    -------
    output_length
        * None if either input is None.
        * Computed length of the pooling operator otherwise.

    Notes
    -----
    When ``ignore_border == True``, this is given by the number of full
    pooling regions that fit in the padded input length,
    divided by the stride (rounding down).

    If ``ignore_border == False``, a single partial pooling region is
    appended if at least one input element would be left uncovered otherwise.
    """
    if input_length is None or pool_size is None:
        return None

    if ignore_border:
        output_length = input_length + 2 * pad - pool_size + 1
        output_length = (output_length + stride - 1) // stride

    # output length calculation taken from:
    # https://github.com/Theano/Theano/blob/master/theano/tensor/signal/downsample.py
    else:
        assert pad == 0

        if stride >= pool_size:
            output_length = (input_length + stride - 1) // stride
        else:
            output_length = max(
                0, (input_length - pool_size + stride - 1) // stride) + 1

    return output_length

class Pool2DLayer(Module):
    """
    2D pooling layer

    Performs 2D mean or max-pooling over the two trailing axes
    of a 4D input tensor.

    Parameters
    ----------
    incoming : a :class:`Layer` instance or tuple
        The layer feeding into this layer, or the expected input shape.

    pool_size : integer or iterable
        The length of the pooling region in each dimension.  If an integer, it
        is promoted to a square pooling region. If an iterable, it should have
        two elements.

    stride : integer, iterable or ``None``
        The strides between sucessive pooling regions in each dimension.
        If ``None`` then ``stride = pool_size``.

    pad : integer or iterable
        Number of elements to be added on each side of the input
        in each dimension. Each value must be less than
        the corresponding stride.

    ignore_border : bool
        If ``True``, partial pooling regions will be ignored.
        Must be ``True`` if ``pad != (0, 0)``.

    mode : {'max', 'average_inc_pad', 'average_exc_pad'}
        Pooling mode: max-pooling or mean-pooling including/excluding zeros
        from partially padded pooling regions. Default is 'max'.

    **kwargs
        Any additional keyword arguments are passed to the :class:`Layer`
        superclass.

    See Also
    --------
    MaxPool2DLayer : Shortcut for max pooling layer.

    Notes
    -----
    The value used to pad the input is chosen to be less than
    the minimum of the input, so that the output of each pooling region
    always corresponds to some element in the unpadded input region.

    Using ``ignore_border=False`` prevents Theano from using cuDNN for the
    operation, so it will fall back to a slower implementation.
    """

    def __init__(self, incoming, pool_size, stride=None, pad=(0, 0),
                 ignore_border=True, mode='max', op=None,
                 **kwargs):
        super().__init__(incoming, **kwargs)

        self.pool_size = as_tuple(pool_size, 2)

        if len(self.input_shape) != 4:
            raise ValueError("Tried to create a 2D pooling layer with "
                             "input shape %r. Expected 4 input dimensions "
                             "(batchsize, channels, 2 spatial dimensions)."
                             % (self.input_shape,))

        if stride is None:
            self.stride = self.pool_size
        else:
            self.stride = as_tuple(stride, 2)

        self.pad = as_tuple(pad, 2)

        self.ignore_border = ignore_border
        self.mode = mode
        if op is None:
            op = backend.Pool(ws=self.pool_size, ignore_border=self.ignore_border, stride=self.stride,
                              pad=self.pad, mode=self.mode, ndim=2)
        self.op = op

    def get_output_shape_for(self, input_shape):
        output_shape = list(input_shape)  # copy / convert to mutable list

        output_shape[2] = pool_output_length(input_shape[2],
                                             pool_size=self.pool_size[0],
                                             stride=self.stride[0],
                                             pad=self.pad[0],
                                             ignore_border=self.ignore_border,
                                             )

        output_shape[3] = pool_output_length(input_shape[3],
                                             pool_size=self.pool_size[1],
                                             stride=self.stride[1],
                                             pad=self.pad[1],
                                             ignore_border=self.ignore_border,
                                             )

        return tuple(output_shape)

    def forward(self, input):
        pooled = self.op(input)
        return pooled



if __name__ == '__main__':
    import os
    floatX = 'float32'
    os.environ['THEANO_FLAGS'] = "floatX=%s, mode=FAST_RUN" % floatX
    os.environ['MODEL_PRECISION'] = '%s' % floatX

    import numpy as np, time
    import base
    import nonlinearities
    import scipy

    np.random.seed(2017)

    import lasagne.layers as lasa_layer
    import lasagne.nonlinearities as lasa_nonlinear

    import theano
    import theano.tensor as tensor
    from lasagne_ext.utils import freeze_layer, get_layer_by_name

    # import torch
    # import torch.nn as nn
    # import torch.nn.functional as F
    # from torch.autograd import Variable
    #
    #
    # class build_model_torch(nn.Module):
    #     def __init__(self, kernel_size=3, border_mode='same', W=None, b=None):
    #         super().__init__()
    #         self.channel = 1
    #         self.Nclass = Nclass
    #         self.kernel_size = kernel_size
    #         self.border_mode = border_mode
    #         if border_mode == 'same':
    #             pad = kernel_size // 2
    #         else:
    #             pad = 0
    #         self.conv0 = nn.Conv2d(channel, 1, kernel_size, padding=pad)
    #         self.conv0.weight.data = torch.Tensor(W)
    #         self.conv0.bias.data = torch.Tensor(b)
    #
    #     def forward(self, x):
    #         x = self.conv0(x)
    #         return x


    #--- lasagne-theano ---#
    def compile_lasagne_model(model):
        """
        Compile the model train & predict functions
        :param model:
        :param fn_tags: list of function tags
        :return:
        """
        X_var = get_layer_by_name(model, 'input0').input_var
        Y_var = lasa_layer.get_output(model, deterministic=True)
        predict_fn = theano.function([X_var], Y_var, no_default_updates=True)
        return predict_fn

    def build_lasagne_model(feadim, input_length=None, pool_size=(2,2)):
        input0 = lasa_layer.InputLayer(shape=(None, 1, feadim, input_length), name='input0')
        pool0  = lasa_layer.Pool2DLayer(input0,pool_size=pool_size)
        return pool0

    #--- model_numpy ---#
    class build_model_np(base.Module):
        def __init__(self, feadim, input_length=None, pool_size=(2,2)):
            super().__init__((None, 1, feadim, input_length))
            self.pool0 = Pool2DLayer((None, 1, feadim, input_length), pool_size=pool_size)
        def forward(self, x):
            x = self.pool0(x)
            return x


    batch_size, channel, feadim, input_length = 1, 1, 101, 101
    pool_size = (5, 5)

    model = build_lasagne_model(feadim=feadim, pool_size=pool_size)
    predict_fn = compile_lasagne_model(model)

    model2 = build_model_np(feadim=feadim, pool_size=pool_size)


    run_time0, run_time2, run_time3, run_time4 = 0, 0, 0, 0
    loop = 10
    for i in range(loop):
        X = np.random.rand(batch_size, channel, feadim, input_length).astype(floatX)
        X2 = X.copy()
        X3 = X[0,0,:,:]

        time0 = time.time()
        Y = predict_fn(X)
        # Y = Y[0, 0, :, :]
        run_time0 += (time.time() - time0)


        time0 = time.time()
        Y2 = model2(X2)
        # Y2 = Y2[0, 0, :, :]
        run_time2 += (time.time() - time0)

        # time0 = time.time()
        # Y3 = scipy.signal.convolve2d(X3, W3, mode='same', )
        # Y3 = scipy.signal.fftconvolve(X3, W3, mode='same', )
        # run_time3 += (time.time() - time0)

        # time0 = time.time()
        # X4 = Variable(torch.Tensor(X.copy()))
        # Y4 = model_torch(X4).data.numpy()
        # run_time4 += (time.time() - time0)




    print('time0 =', run_time0/loop)
    print('time2 =', run_time2/loop)
    # print('time3 =', run_time3/loop)
    # print('time4 =', run_time4/loop)
    print('acceleration =', run_time0/run_time2)

    dY12 = Y - Y2
    # dY13 = Y - Y3
    # dY23 = Y2 - Y3
    # dY14 = Y - Y4
    # dY24 = Y2 - Y4
    print('max diff(1,2) = ', np.abs(dY12).max())
    # print('max diff(1,3) = ', np.abs(dY13).max())
    # print('max diff(2,3) = ', np.abs(dY23).max())
    # print('max diff(1,4) = ', np.abs(dY14).max())
    # print('max diff(2,4) = ', np.abs(dY24).max())

    print(Y2.dtype)

    # print(Y4)