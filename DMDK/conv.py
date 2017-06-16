# coding:utf-8
"""
# Numpy-version of Lasagne's CNN
# Result:
        time0 = 0.21010231971740723
        time2 = 0.47037267684936523
        time4 = 0.28623151779174805
        acceleration = 0.44667203274796413
        max diff(1,2) =  7.15256e-07
        max diff(1,4) =  0.0
        max diff(2,4) =  7.15256e-07
    So, for float32
        1) pytorch '0.1.12_2' is a bit slower than Theano, both much faster than scipy
        2) pytorch's result exactly match Theano's
    for float64, pytorch is not working well with double precision.
# Created   :   6, 12, 2017
# Revised   :   6, 16, 2017
# All rights reserved
#------------------------------------------------------------------------------------------------
"""
__author__ = 'dawei.leng'

from base import *
import backend
import utils, nonlinearities, init

def conv_output_length(input_length, filter_size, stride, pad=0):
    """Helper function to compute the output size of a convolution operation

    This function computes the length along a single axis, which corresponds
    to a 1D convolution. It can also be used for convolutions with higher
    dimensionalities by using it individually for each axis.

    Parameters
    ----------
    input_length : int or None
        The size of the input.

    filter_size : int
        The size of the filter.

    stride : int
        The stride of the convolution operation.

    pad : int, 'full' or 'same' (default: 0)
        By default, the convolution is only computed where the input and the
        filter fully overlap (a valid convolution). When ``stride=1``, this
        yields an output that is smaller than the input by ``filter_size - 1``.
        The `pad` argument allows you to implicitly pad the input with zeros,
        extending the output size.

        A single integer results in symmetric zero-padding of the given size on
        both borders.

        ``'full'`` pads with one less than the filter size on both sides. This
        is equivalent to computing the convolution wherever the input and the
        filter overlap by at least one position.

        ``'same'`` pads with half the filter size on both sides (one less on
        the second side for an even filter size). When ``stride=1``, this
        results in an output size equal to the input size.

    Returns
    -------
    int or None
        The output size corresponding to the given convolution parameters, or
        ``None`` if `input_size` is ``None``.

    Raises
    ------
    ValueError
        When an invalid padding is specified, a `ValueError` is raised.
    """
    if input_length is None:
        return None
    if pad == 'valid':
        output_length = input_length - filter_size + 1
    elif pad == 'full':
        output_length = input_length + filter_size - 1
    elif pad == 'same':
        output_length = input_length
    elif isinstance(pad, int):
        output_length = input_length + 2 * pad - filter_size + 1
    else:
        raise ValueError('Invalid pad: {0}'.format(pad))

    # This is the integer arithmetic equivalent to
    # np.ceil(output_length / stride)
    output_length = (output_length + stride - 1) // stride

    return output_length

def conv_input_length(output_length, filter_size, stride, pad=0):
    """Helper function to compute the input size of a convolution operation

    This function computes the length along a single axis, which corresponds
    to a 1D convolution. It can also be used for convolutions with higher
    dimensionalities by using it individually for each axis.

    Parameters
    ----------
    output_length : int or None
        The size of the output.

    filter_size : int
        The size of the filter.

    stride : int
        The stride of the convolution operation.

    pad : int, 'full' or 'same' (default: 0)
        By default, the convolution is only computed where the input and the
        filter fully overlap (a valid convolution). When ``stride=1``, this
        yields an output that is smaller than the input by ``filter_size - 1``.
        The `pad` argument allows you to implicitly pad the input with zeros,
        extending the output size.

        A single integer results in symmetric zero-padding of the given size on
        both borders.

        ``'full'`` pads with one less than the filter size on both sides. This
        is equivalent to computing the convolution wherever the input and the
        filter overlap by at least one position.

        ``'same'`` pads with half the filter size on both sides (one less on
        the second side for an even filter size). When ``stride=1``, this
        results in an output size equal to the input size.

    Returns
    -------
    int or None
        The smallest input size corresponding to the given convolution
        parameters for the given output size, or ``None`` if `output_size` is
        ``None``. For a strided convolution, any input size of up to
        ``stride - 1`` elements larger than returned will still give the same
        output size.

    Raises
    ------
    ValueError
        When an invalid padding is specified, a `ValueError` is raised.

    Notes
    -----
    This can be used to compute the output size of a convolution backward pass,
    also called transposed convolution, fractionally-strided convolution or
    (wrongly) deconvolution in the literature.
    """
    if output_length is None:
        return None
    if pad == 'valid':
        pad = 0
    elif pad == 'full':
        pad = filter_size - 1
    elif pad == 'same':
        pad = filter_size // 2
    if not isinstance(pad, int):
        raise ValueError('Invalid pad: {0}'.format(pad))
    return (output_length - 1) * stride - 2 * pad + filter_size

class BaseConvLayer(Module):
    """
    lasagne.layers.BaseConvLayer(incoming, num_filters, filter_size,
    stride=1, pad=0, untie_biases=False,
    W=lasagne.init.GlorotUniform(), b=lasagne.init.Constant(0.),
    nonlinearity=lasagne.nonlinearities.rectify, flip_filters=True,
    conv_dim=None, **kwargs)

    Convolutional layer base class

    Base class for performing an `conv_dim`-dimensional convolution on its input,
    optionally adding a bias and applying an elementwise nonlinearity. Note
    that this class cannot be used in a Lasagne network, only its subclasses
    can (e.g., :class:`Conv1DLayer`, :class:`Conv2DLayer`).

    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape. Must
        be a tensor of 2+`conv_dim` dimensions:
        ``(batch_size, num_input_channels, <conv_dim spatial dimensions>)``.

    num_filters : int
        The number of learnable convolutional filters this layer has.

    filter_size : int or iterable of int
        An integer or an `conv_dim`-element tuple specifying the size of the filters.

    stride : int or iterable of int
        An integer or an `conv_dim`-element tuple specifying the stride of the
        convolution operation.

    pad : int, iterable of int, 'full', 'same' or 'valid' (default: 0)
        By default, the convolution is only computed where the input and the
        filter fully overlap (a valid convolution). When ``stride=1``, this
        yields an output that is smaller than the input by ``filter_size - 1``.
        The `pad` argument allows you to implicitly pad the input with zeros,
        extending the output size.

        A single integer results in symmetric zero-padding of the given size on
        all borders, a tuple of `conv_dim` integers allows different symmetric padding
        per dimension.

        ``'full'`` pads with one less than the filter size on both sides. This
        is equivalent to computing the convolution wherever the input and the
        filter overlap by at least one position.

        ``'same'`` pads with half the filter size (rounded down) on both sides.
        When ``stride=1`` this results in an output size equal to the input
        size. Even filter size is not supported.

        ``'valid'`` is an alias for ``0`` (no padding / a valid convolution).

        Note that ``'full'`` and ``'same'`` can be faster than equivalent
        integer values due to optimizations by Theano.

    untie_biases : bool (default: False)
        If ``False``, the layer will have a bias parameter for each channel,
        which is shared across all positions in this channel. As a result, the
        `b` attribute will be a vector (1D).

        If ``True``, the layer will have separate bias parameters for each
        position in each channel. As a result, the `b` attribute will be an
        `conv_dim`-dimensional tensor.

    W : Theano shared variable, expression, numpy array or callable
        Initial value, expression or initializer for the weights.
        These should be a tensor of 2+`conv_dim` dimensions with shape
        ``(num_filters, num_input_channels, <conv_dim spatial dimensions>)``.
        See :func:`lasagne.utils.create_param` for more information.

    b : Theano shared variable, expression, numpy array, callable or ``None``
        Initial value, expression or initializer for the biases. If set to
        ``None``, the layer will have no biases. Otherwise, biases should be
        a 1D array with shape ``(num_filters,)`` if `untied_biases` is set to
        ``False``. If it is set to ``True``, its shape should be
        ``(num_filters, <conv_dim spatial dimensions>)`` instead.
        See :func:`lasagne.utils.create_param` for more information.

    nonlinearity : callable or None
        The nonlinearity that is applied to the layer activations. If None
        is provided, the layer will be linear.

    flip_filters : bool (default: True)
        Whether to flip the filters before sliding them over the input,
        performing a convolution (this is the default), or not to flip them and
        perform a correlation. Note that for some other convolutional layers in
        Lasagne, flipping incurs an overhead and is disabled by default --
        check the documentation when using learned weights from another layer.

    conv_dim : int or None
        The dimensionality of the convolution (i.e., the number of spatial
        dimensions of each feature map and each convolutional filter). If
        ``None``, will be inferred from the input shape.

    **kwargs
        Any additional keyword arguments are passed to the `Layer` superclass.

    Attributes
    ----------
    W : Theano shared variable or expression
        Variable or expression representing the filter weights.

    b : Theano shared variable or expression
        Variable or expression representing the biases.
    """
    def __init__(self, incoming, num_filters, filter_size, stride=1, pad=0,
                 untie_biases=False, W=init.Empty(), b=init.Empty(), 
                 nonlinearity=nonlinearities.rectify, flip_filters=True,
                 conv_dim=None, **kwargs):
        super().__init__(incoming, **kwargs)
        if nonlinearity is None:
            self.nonlinearity = nonlinearities.identity
        else:
            self.nonlinearity = nonlinearity

        if conv_dim is None:    # convolution dimension
            conv_dim = len(self.input_shape) - 2
        elif conv_dim != len(self.input_shape) - 2:
            raise ValueError("Tried to create a %dD convolution layer with "
                             "input shape %r. Expected %d input dimensions "
                             "(batch_size, channels, %d spatial dimensions)." %
                             (conv_dim, self.input_shape, conv_dim+2, conv_dim))
        self.conv_dim     = conv_dim       # convolution dimension
        self.num_filters  = num_filters
        self.filter_size  = utils.as_tuple(filter_size, conv_dim, int)
        self.flip_filters = flip_filters
        self.stride       = utils.as_tuple(stride, conv_dim, int)
        self.untie_biases = untie_biases

        if pad == 'same':
            if any(s % 2 == 0 for s in self.filter_size):
                raise NotImplementedError('`same` padding requires odd filter size.')
        if pad == 'valid':
            self.pad = utils.as_tuple(0, conv_dim)
        elif pad in ('full', 'same'):
            self.pad = pad
        else:
            self.pad = utils.as_tuple(pad, conv_dim, int)

        self.W = self.add_param(W, self.get_W_shape(), name='W')
        # self.W = self.register_parameter(W, self.get_W_shape(), name='W')
        if b is None:
            self.b = None
        else:
            if self.untie_biases:
                biases_shape = (num_filters,) + self.output_shape[2:]
            else:
                biases_shape = (num_filters,)
            self.b = self.add_param(b, biases_shape, name='b')

    def get_W_shape(self):
        """Get the shape of the weight matrix `W`.

        Returns
        -------
        tuple of int
            The shape of the weight matrix.
        """
        num_input_channels = self.input_shape[1]
        return (self.num_filters, num_input_channels) + self.filter_size

    def get_output_shape_for(self, input_shape):
        pad = self.pad if isinstance(self.pad, tuple) else (self.pad,) * self.conv_dim
        batch_size = input_shape[0]
        return ((batch_size, self.num_filters) + tuple(conv_output_length(input, filter, stride, p)
                for input, filter, stride, p in zip(input_shape[2:], self.filter_size, self.stride, pad)))

    def forward(self, input):
        conved = self._convolve(input)
        if self.b is None:
            activation = conved
        elif self.untie_biases:
            activation = conved + backend.shape_padleft(self.b, 1)
        else:
            activation = conved + backend.dimshuffle(self.b, ('x', 0) + ('x',) * self.conv_dim)

        return self.nonlinearity(activation)

    def _convolve(self, input):
        """
        Symbolically convolves `input` with ``self.W``, producing an output of
        shape ``self.output_shape``. To be implemented by subclasses.

        Parameters
        ----------
        input : Theano tensor
            The input minibatch to convolve
        **kwargs
            Any additional keyword arguments from :meth:`get_output_for`

        Returns
        -------
        Theano tensor
            `input` convolved according to the configuration of this layer,
            without any bias or nonlinearity applied.
        """
        raise NotImplementedError("BaseConvLayer does not implement the "
                                  "convolve() method. You will want to "
                                  "use a subclass such as Conv2DLayer.")


class Conv2DLayer(BaseConvLayer):
    """
    lasagne.layers.Conv2DLayer(incoming, num_filters, filter_size,
    stride=(1, 1), pad=0, untie_biases=False,
    W=lasagne.init.GlorotUniform(), b=lasagne.init.Constant(0.),
    nonlinearity=lasagne.nonlinearities.rectify, flip_filters=True,
    convolution=theano.tensor.nnet.conv2d, **kwargs)

    2D convolutional layer

    Performs a 2D convolution on its input and optionally adds a bias and
    applies an elementwise nonlinearity.

    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape. The
        output of this layer should be a 4D tensor, with shape
        ``(batch_size, num_input_channels, input_rows, input_columns)``.

    num_filters : int
        The number of learnable convolutional filters this layer has.

    filter_size : int or iterable of int
        An integer or a 2-element tuple specifying the size of the filters.

    stride : int or iterable of int
        An integer or a 2-element tuple specifying the stride of the
        convolution operation.

    pad : int, iterable of int, 'full', 'same' or 'valid' (default: 0)
        By default, the convolution is only computed where the input and the
        filter fully overlap (a valid convolution). When ``stride=1``, this
        yields an output that is smaller than the input by ``filter_size - 1``.
        The `pad` argument allows you to implicitly pad the input with zeros,
        extending the output size.

        A single integer results in symmetric zero-padding of the given size on
        all borders, a tuple of two integers allows different symmetric padding
        per dimension.

        ``'full'`` pads with one less than the filter size on both sides. This
        is equivalent to computing the convolution wherever the input and the
        filter overlap by at least one position.

        ``'same'`` pads with half the filter size (rounded down) on both sides.
        When ``stride=1`` this results in an output size equal to the input
        size. Even filter size is not supported.

        ``'valid'`` is an alias for ``0`` (no padding / a valid convolution).

        Note that ``'full'`` and ``'same'`` can be faster than equivalent
        integer values due to optimizations by Theano.

    untie_biases : bool (default: False)
        If ``False``, the layer will have a bias parameter for each channel,
        which is shared across all positions in this channel. As a result, the
        `b` attribute will be a vector (1D).

        If True, the layer will have separate bias parameters for each
        position in each channel. As a result, the `b` attribute will be a
        3D tensor.

    W : Theano shared variable, expression, numpy array or callable
        Initial value, expression or initializer for the weights.
        These should be a 4D tensor with shape
        ``(num_filters, num_input_channels, filter_rows, filter_columns)``.
        See :func:`lasagne.utils.create_param` for more information.

    b : Theano shared variable, expression, numpy array, callable or ``None``
        Initial value, expression or initializer for the biases. If set to
        ``None``, the layer will have no biases. Otherwise, biases should be
        a 1D array with shape ``(num_filters,)`` if `untied_biases` is set to
        ``False``. If it is set to ``True``, its shape should be
        ``(num_filters, output_rows, output_columns)`` instead.
        See :func:`lasagne.utils.create_param` for more information.

    nonlinearity : callable or None
        The nonlinearity that is applied to the layer activations. If None
        is provided, the layer will be linear.

    flip_filters : bool (default: True)
        Whether to flip the filters before sliding them over the input,
        performing a convolution (this is the default), or not to flip them and
        perform a correlation. Note that for some other convolutional layers in
        Lasagne, flipping incurs an overhead and is disabled by default --
        check the documentation when using learned weights from another layer.

    convolution : callable
        The convolution implementation to use. Usually it should be fine to
        leave this at the default value.

    **kwargs
        Any additional keyword arguments are passed to the `Layer` superclass.

    Attributes
    ----------
    W : Theano shared variable or expression
        Variable or expression representing the filter weights.

    b : Theano shared variable or expression
        Variable or expression representing the biases.
    """
    def __init__(self, incoming, num_filters, filter_size, stride=(1, 1),
                 pad=0, untie_biases=False,
                 W=init.Empty(), b=init.Empty(),
                 nonlinearity=nonlinearities.rectify, flip_filters=True,
                 op=None, **kwargs):
        super().__init__(incoming, num_filters, filter_size,
                         stride, pad, untie_biases, W, b,
                         nonlinearity, flip_filters, conv_dim=2, **kwargs)
        if op is None:   # default op
            if self.pad == 'same':
                border_mode = 'half'
            else:
                border_mode = self.pad
            op = backend.Conv(conv_dim=2, im_shape=self.input_shape, kernel_shape=self.get_W_shape(),
                              border_mode=border_mode, subsample=self.stride, filter_flip=self.flip_filters)
        self.op = op

    def _convolve(self, input):
        """
        Override BaseConvLayer's convolve()
        :param input:
        :return:
        """
        conved = self.op(input, self.W)
        return conved






if __name__ == '__main__':
    import os

    floatX = 'float32'
    # os.environ['THEANO_FLAGS'] = "floatX=float32, mode=FAST_RUN, warn_float64='raise'"
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

    def build_lasagne_model(feadim, Nclass, kernel_size=3, border_mode='same', input_length=None, W=None, b=None):
        input0 = lasa_layer.InputLayer(shape=(None, 1, feadim, input_length), name='input0')
        cnn0   = lasa_layer.Conv2DLayer(input0, num_filters=1, filter_size=kernel_size, pad=border_mode, nonlinearity=lasa_nonlinear.linear,  name='cnn0',
                           W=W, b=b)
        return cnn0

    #--- model_numpy ---#
    class build_model_np(base.Module):
        def __init__(self, feadim, Nclass, kernel_size=3, border_mode='same', input_length=None, W=None, b=None):
            super().__init__((None, 1, feadim, input_length))
            self.cnn0 = Conv2DLayer((None, 1, feadim, input_length), num_filters=1, filter_size=kernel_size,
                                           pad=border_mode, nonlinearity=nonlinearities.linear, name='cnn0', W=W, b=b)
        def forward(self, x):
            x = self.cnn0(x)
            return x


    batch_size, channel, feadim, input_length, Nclass = 1, 1, 500, None, 10
    kernel_size, border_mode = 3, 'same'
    filter_num = 1

    W = np.random.rand(filter_num, channel, kernel_size, kernel_size).astype(floatX)
    b = np.random.rand(filter_num).astype(floatX) * 0.0
    # W[0, 0, :, :] = np.ones((3, 3), floatX)

    model = build_lasagne_model(feadim=feadim, Nclass=Nclass, kernel_size=kernel_size, border_mode=border_mode, input_length=input_length,
                                W=W, b=b)
    predict_fn = compile_lasagne_model(model)

    model2 = build_model_np(feadim=feadim, Nclass=Nclass, kernel_size=kernel_size, border_mode=border_mode, input_length=input_length,
                                W=W, b=b)

    # W4 = W.copy()
    # W4 = W[(slice(None), slice(None)) + (slice(None, None, -1),) * 2]
    # W4 = W4.copy()

    # model_torch = build_model_torch(kernel_size=kernel_size, border_mode=border_mode,W=W4, b=b)
    # model_torch.eval()

    # model_torch.conv0.weight.data = torch.Tensor(W4)
    # model_torch.conv0.bias.data = torch.Tensor(b)


    run_time0, run_time2, run_time3, run_time4 = 0, 0, 0, 0
    loop = 10
    for i in range(loop):
        X = np.random.rand(batch_size, channel, feadim, 500).astype(floatX)
        X2 = X.copy()
        X3 = X[0,0,:,:]

        W3 = W[0, 0, :, :]

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