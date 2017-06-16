# coding:utf-8
"""
Backend Ops, implemented via numpy & scipy
# Created   :   6, 12, 2017
# Revised   :   6, 16, 2017
# All rights reserved
#------------------------------------------------------------------------------------------------
"""
__author__ = 'dawei.leng'

import numpy as np
import scipy

def dimshuffle(x, *pattern):
    """
    Reorder the dimensions of this variable, optionally inserting
    broadcasted dimensions.
    Same with dimshuffle() of Theano

    Parameters
    ----------
    pattern
        List/tuple of int mixed with 'x' for broadcastable dimensions.

    Examples
    --------
    For example, to create a 3D view of a [2D] matrix, call
    ``dimshuffle([0,'x',1])``.  This will create a 3D view such that the
    middle dimension is an implicit broadcasted dimension.  To do the same
    thing on the transpose of that matrix, call ``dimshuffle([1, 'x', 0])``.

    Notes
    -----
    This function supports the pattern passed as a tuple, or as a
    variable-length argument (e.g. ``a.dimshuffle(pattern)`` is equivalent
    to ``a.dimshuffle(*pattern)`` where ``pattern`` is a list/tuple of ints
    mixed with 'x' characters).

    """
    if (len(pattern) == 1) and (isinstance(pattern[0], (list, tuple))):
        pattern = pattern[0]
    new_order = tuple(pattern)
    shuffle = [x for x in new_order if x != 'x']
    augment = [i for i, x in enumerate(new_order) if x == 'x']
    drop = []
    shape = x.shape
    for i in range(len(shape)):
        if i not in new_order:
            drop.append(i)
    x = x.transpose(shuffle + drop)
    shape = list(x.shape[:len(shuffle)])
    for augm in augment:
        shape.insert(augm, 1)
    x = x.reshape(shape)
    return x

def shape_padleft(x, n_ones=1):
    """Reshape `x` by left-padding the shape with `n_ones` 1s.

    See Also
    --------
    shape_padaxis
    shape_padright
    Dimshuffle

    """
    pattern = ['x'] * n_ones + list(range(x.ndim))
    return dimshuffle(x, pattern)

def get_conv_shape_1axis(image_shape, kernel_shape, border_mode, subsample, dilation=1):
    """
    This function compute the output shape of convolution operation on one axis

    Parameters
    ----------
    image_shape: int or None. Corresponds to the input image shape on a
        given axis. None if undefined.
    kernel_shape: int or None. Corresponds to the kernel shape on a given
        axis. None if undefined.
    border_mode: string or int. If it is a string, it must be
        'valid', 'half' or 'full'. If it is an integer, it must correspond to
        the padding on the considered axis.
    subsample: int. It must correspond to the subsampling on the
        considered axis.
    dilation: int. It must correspond to the dilation on the
        considered axis.

    Returns
    -------
    out_shape: int corresponding to the output image shape on the
        considered axis. None if undefined.

    """
    if None in [image_shape, kernel_shape, border_mode, subsample, dilation]:
        return None

    dilated_kernel_shape = (kernel_shape - 1) * dilation + 1
    if border_mode == "half":
        pad = dilated_kernel_shape // 2
    elif border_mode == "full":
        pad = dilated_kernel_shape - 1
    elif border_mode == "valid":
        pad = 0
    else:
        pad = border_mode

    if pad == 0:
        out_shape = (image_shape - dilated_kernel_shape)
    else:
        out_shape = (image_shape + 2 * pad - dilated_kernel_shape)
    if subsample != 1:
        out_shape = out_shape // subsample
    out_shape = out_shape + 1

    return out_shape

def get_conv_output_shape(image_shape, kernel_shape, border_mode, subsample, kernel_dilation):
    """
    This function compute the output shape of convolution operation.

    Parameters
    ----------
    image_shape: tuple of int (symbolic or numeric) corresponding to the input
        image shape. Its four (or five) element must correspond respectively
        to: batch size, number of input channels, height and width (and
        possibly depth) of the image. None where undefined.
    kernel_shape: tuple of int (symbolic or numeric) corresponding to the
        kernel shape. Its four (or five) elements must correspond respectively
        to: number of output channels, number of input channels, height and
        width (and possibly depth) of the kernel. None where undefined.
    border_mode: string, int (symbolic or numeric) or tuple of int (symbolic
        or numeric). If it is a string, it must be 'valid', 'half' or 'full'.
        If it is a tuple, its two (or three) elements respectively correspond
        to the padding on height and width (and possibly depth) axis.
    subsample: tuple of int (symbolic or numeric). Its two or three elements
        espectively correspond to the subsampling on height and width (and
        possibly depth) axis.
    kernel_dilation: tuple of int (symbolic or numeric). Its two or three
        elements correspond respectively to the dilation on height and width axis.

    Returns
    -------
    output_shape: tuple of int corresponding to the output image shape. Its
        four element must correspond respectively to: batch size, number of
        output channels, height and width of the image. None where undefined.

    """
    batch_size, im_shape     = image_shape[0], image_shape[2:]
    kernel_num, kernel_shape = kernel_shape[0], kernel_shape[2:]

    if isinstance(border_mode, tuple):
        out_shape = tuple(get_conv_shape_1axis(im_shape[i], kernel_shape[i], border_mode[i], subsample[i],
                                               kernel_dilation[i]) for i in range(len(subsample)))
    elif border_mode in {'half', 'valid', 'full'}:
        out_shape = tuple(get_conv_shape_1axis(im_shape[i], kernel_shape[i], border_mode, subsample[i], 
                                               kernel_dilation[i]) for i in range(len(subsample)))
    else:
        raise ValueError('User given border_mode not supported', border_mode)
    return (batch_size, kernel_num) + out_shape

def conv(im, kernel, conv_dim=2, border_mode="valid", subsample=None, kernel_dilation=None):
    """
    2D or 3D convolution based on numpy & scipy
    :param im:     input, shape = (B, C, H, W)
    :param kernel: shape = (N_out, N_in, H_kernel, W_kernel)
    :param border_mode: 'valid'/'same'/'full' or 0/1/2
    """
    if subsample is None:
        subsample = (1,) * conv_dim
    if kernel_dilation is None:
        kernel_dilation = (1,) * conv_dim

    out_shape = get_conv_output_shape(im.shape, kernel.shape, border_mode, subsample, kernel_dilation)
    out = np.zeros(out_shape, dtype=im.dtype)
    dilated_kernel_shape = kernel.shape[:-conv_dim] + tuple((kernel.shape[-conv_dim + i] - 1) * kernel_dilation[i] + 1 for i in range(conv_dim))
    dilated_kernel = np.zeros(dilated_kernel_shape, dtype=kernel.dtype)
    dilated_kernel[(slice(None), slice(None)) + tuple(slice(None, None, kernel_dilation[i]) for i in range(conv_dim))] = kernel

    # [6-15-2017, DV] scipy.signal.convolve2d() runs faster than scipy.signal.fftconvolve() for 2D convolution, that's
    # why the if-else conditioning here.
    if conv_dim == 2:
        for b in range(im.shape[0]):
            for n in range(kernel.shape[0]):
                for im0 in range(im.shape[1]):
                    out[b, n, ...] += scipy.signal.convolve2d(im[b, im0, ...],  dilated_kernel[n, im0, ...], border_mode)
    elif conv_dim == 3:
        for b in range(im.shape[0]):
            for n in range(kernel.shape[0]):
                for im0 in range(im.shape[1]):
                    out[b, n, ...] += scipy.signal.fftconvolve(im[b, im0, ...], dilated_kernel[n, im0, ...], border_mode)
    else:
        raise NotImplementedError('Only 2D/3D convolution supported yet')
    return out

class Conv(object):
    """
    Op for 2D/3D convolution
    """
    def __init__(self, conv_dim=2,
                 im_shape=None, kernel_shape=None, border_mode="valid",
                 subsample=None, filter_flip=True, kernel_dilation=None):
        super().__init__()
        #--- check inputs ---#
        if conv_dim not in {2, 3}:
            raise NotImplementedError('convolution dimension {} is not supported', conv_dim)
        if subsample is None:
            subsample = (1,) * conv_dim
        if len(subsample) != conv_dim:
            raise ValueError("subsample must have {} elements".format(conv_dim))
        if kernel_dilation is None:
            kernel_dilation = (1,) * conv_dim
        if len(kernel_dilation) != conv_dim:
            raise ValueError("kernel_dilation must have {} elements".format(conv_dim))
        if isinstance(border_mode, int):
            border_mode = (border_mode,) * conv_dim
        elif isinstance(border_mode, tuple):
            if len(border_mode) != conv_dim:
                raise ValueError('border border_mode must have exactly {} values, but was {}'.format(conv_dim, border_mode))
            border_mode = tuple(map(int, border_mode))
        elif border_mode in {'valid', 'half', 'full'}:
            pass
        else:
            raise ValueError('User given border_mode not supported', border_mode)


        self.conv_dim        = conv_dim
        self.im_shape        = tuple(im_shape) if im_shape else (None,) * (2 + conv_dim)
        self.kernel_shape    = tuple(kernel_shape) if kernel_shape else (None,) * (2 + conv_dim)
        self.border_mode     = border_mode
        self.filter_flip     = filter_flip
        self.subsample       = tuple(subsample)
        self.kernel_dilation = tuple(kernel_dilation)
        dilated_kernel_shape = tuple((self.kernel_shape[2 + i] - 1) * self.kernel_dilation[i] + 1 for i in range(self.conv_dim))
        if isinstance(border_mode, str):
            if border_mode == "full":
                self.pad = tuple(dilated_kernel_shape[i] - 1 for i in range(self.conv_dim))
            elif border_mode == "half":
                self.pad = tuple(int(dilated_kernel_shape[i] // 2) for i in range(self.conv_dim))
            else:                     # border_mode == 'valid':
                self.pad = (0,) * self.conv_dim

    def __call__(self, im, kernel):
        new_img = np.zeros((im.shape[0], im.shape[1]) + tuple(im.shape[i + 2] + 2 * self.pad[i]
                           for i in range(self.conv_dim)), dtype=im.dtype)
        new_img[(slice(None), slice(None)) + tuple(slice(self.pad[i], im.shape[i + 2] + self.pad[i])
                 for i in range(self.conv_dim))] = im
        if not self.filter_flip:
            kernel = kernel[(slice(None), slice(None)) + (slice(None, None, -1),) * self.conv_dim]
        conv_out = conv(new_img, kernel, self.conv_dim,
                        border_mode="valid",
                        subsample=self.subsample,
                        kernel_dilation=self.kernel_dilation)
        conv_out = conv_out[(slice(None), slice(None)) +  tuple(slice(None, None, self.subsample[i])
                            for i in range(self.conv_dim))]
        return conv_out

class Pool(object):
    """
    sum or average over different patches.

    Parameters
    ----------
    ws : list or tuple of N ints
        Downsample factor over rows, columns etc.
        ws indicates the size of the pooling region.
    ignore_border : bool
        If ws doesn't divide imgshape, do we include an extra row/col/slice
        of partial downsampling (False) or ignore it (True).
    stride : list or tuple of N ints or None
        Stride size, which is the number of shifts over rows/cols/slices to get the
        next pool region. If stride is None, it is considered equal to ws
        (no overlap on pooling regions).
    pad : tuple of N ints or None
        For each downsampling dimension, this specifies the number of zeros to
        add as padding on both sides. For 2D and (pad_h, pad_w), pad_h specifies the
        size of the top and bottom margins, pad_w specifies the size of the left and
        right margins. No padding is added if pad is None.
    mode : {'max', 'sum', 'average_inc_pad', 'average_exc_pad'}
        ('average_inc_pad' excludes the padding from the count,
        'average_exc_pad' include it)
    ndim : int
        The number of pooling dimensions N.
        The default is 2.


    """
    def __init__(self, ws=None, ignore_border=False, stride=None, pad=None, mode='max', ndim=2):
        super().__init__()
        self.ws = ws
        self.ignore_border = ignore_border
        self.stride = stride
        self.pad = pad
        self.ndim = ndim
        if mode not in ['max', 'average_inc_pad', 'average_exc_pad', 'sum']:
            raise ValueError(
                "Pool mode parameter only support 'max', 'sum',"
                " 'average_inc_pad' and 'average_exc_pad'. Got %s" % mode)
        self.mode = mode

    @staticmethod
    def get_out_shape(imgshape, ws=None, ignore_border=False, stride=None, pad=None, ndim=2):
        """
        Return the shape of the output from this op, for input of given
        shape and flags.

        Parameters
        ----------
        imgshape : tuple, list, or similar of integer or scalar Theano variable
            The shape of a tensor of images. The last N elements are
            interpreted as the number of rows, and the number of cols.
        ws : list or tuple of N ints
            Downsample factor over rows and column.
            ws indicates the pool region size.
        ignore_border : bool
            If ws doesn't divide imgshape, do we include an extra row/col/slice
            of partial downsampling (False) or ignore it (True).
        stride : list or tuple of N ints or None
            Stride size, which is the number of shifts over rows/cols/slices to get the
            next pool region. If stride is None, it is considered equal to ws
            (no overlap on pooling regions).
        pad : tuple of N ints or None
            For each downsampling dimension, this specifies the number of zeros to
            add as padding on both sides. For 2D and (pad_h, pad_w), pad_h specifies the
            size of the top and bottom margins, pad_w specifies the size of the left and
            right margins. No padding is added if pad is None.
        ndim : int
            The number of pooling dimensions N.
            The default is 2.

        Returns
        -------
        list
            The shape of the output from this op, for input of given shape.
            This will have the same length as imgshape, but with last N
            elements reduced as per the downsampling & ignore_border flags.

        """
        # check for deprecated parameter names
        if ndim is None:
            ndim = 2
        if len(imgshape) < ndim:
            raise TypeError('imgshape must have at least {} dimensions'.format(ndim))

        if stride is None:
            stride = ws
        if pad is None:
            pad = (0,) * ndim
        padded_imgshape = tuple(imgshape[-ndim + i] + pad[i] * 2 for i in range(ndim)) # only the last ndim

        def compute_out(dim_padded, downsample, stride):
            if ignore_border:
                if downsample == stride:
                    return dim_padded // stride
                else:
                    out = (dim_padded - downsample) // stride + 1
                    return max(out, 0)
            else:
                if stride >= downsample:
                    return (dim_padded - 1) // stride + 1
                else:
                    return max(0, (dim_padded - 1 - downsample + stride) // stride) + 1

        out_shape = [compute_out(padded_imgshape[i], ws[i], stride[i]) for i in range(ndim)]

        return  list(imgshape[:-ndim]) + out_shape

    def __call__(self, x):
        out_shape      = self.get_out_shape(x.shape, self.ws, self.ignore_border, self.stride, self.pad, self.ndim)
        result         = np.empty(out_shape, dtype=x.dtype)
        pool_out_shape = result.shape[-self.ndim:]    # the last `ndim` dimensions
        img_shp        = tuple(x.shape[-self.ndim + i] + 2 * self.pad[i] for i in range(self.ndim))
        inc_pad        = self.mode == 'average_inc_pad'

        # pad the image
        if max(self.pad) != 0:
            y = np.zeros(x.shape[:-self.ndim] + img_shp, dtype=x.dtype)
            y[(slice(None),) * (len(x.shape) - self.ndim) +
              tuple(slice(self.pad[i], img_shp[i] - self.pad[i]) for i in range(self.ndim))] = x
        else:
            y = x
        func = np.max
        if self.mode == 'sum':
            func = np.sum
        elif self.mode != 'max':
            func = np.average

        # precompute the region boundaries for each dimension
        region_slices = [[] for i in range(self.ndim)]
        for i in range(self.ndim):
            for j in range(pool_out_shape[i]):
                start = j * self.stride[i]
                end = min(start + self.ws[i], img_shp[i])
                if not inc_pad:
                    start = max(start, self.pad[i])
                    end = min(end, img_shp[i] - self.pad[i])
                region_slices[i].append(slice(start, end))

        # iterate over non-pooling dimensions
        for k in np.ndindex(*x.shape[:-self.ndim]):
            rk = result[k]
            yk = y[k]
            # iterate over pooling regions
            for r in np.ndindex(*pool_out_shape):
                rk[r] = func(yk[[region_slices[i][r[i]] for i in range(self.ndim)]])
        return result


if __name__ == '__main__':
    import theano
    import theano.tensor as tensor

    # test: dimshuffle()
    if 0:
        x_tensor = tensor.tensor3('input', 'float64')
        pattern = (2, 1, 'x', 0, 'x')
        y_tensor = x_tensor.dimshuffle(pattern)
        f = theano.function([x_tensor], y_tensor)
        x = np.random.rand(3, 4, 5)
        x0 = x.copy()
        x2 = dimshuffle(x, pattern)
        x3 = f(x0)

        mask = x2 == x3
        if mask.all():
            print('Pass: dimshuffle()')
        else:
            print('Fail: dimshuffle()')

    # test: shape_padleft()
    if 1:
        x_tensor = tensor.tensor3('input', 'float64')
        y_tensor = tensor.shape_padleft(x_tensor, 2)
        f = theano.function([x_tensor], y_tensor)
        x1 = np.random.rand(3, 4, 5)
        x2 = x1.copy()
        y1 = shape_padleft(x1, 2)
        y2 = f(x2)

        mask = y1 == y2
        if mask.all():
            print('Pass: shape_padleft()')
        else:
            print('Fail: shape_padleft()')
