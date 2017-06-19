# coding:utf-8
# <Descriptions>
# Created   :  mm, dd, yyyy
# Revised   :  mm, dd, yyyy
# All rights reserved
#------------------------------------------------------------------------------------------------
__author__ = 'dawei.leng'

import init
import nonlinearities
from base import Module

class LocalResponseNormalization2DLayer(Module):
    """
    Cross-channel Local Response Normalization for 2D feature maps.

    Aggregation is purely across channels, not within channels,
    and performed "pixelwise".

    If the value of the :math:`i` th channel is :math:`x_i`, the output is

    .. math::
        x_i = \\frac{x_i}{ (k + ( \\alpha \\sum_j x_j^2 ))^\\beta }

    where the summation is performed over this position on :math:`n`
    neighboring channels.

    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape. Must
        follow *BC01* layout, i.e., ``(batchsize, channels, rows, columns)``.
    alpha : float scalar
        coefficient, see equation above
    k : float scalar
        offset, see equation above
    beta : float scalar
        exponent, see equation above
    n : int
        number of adjacent channels to normalize over, must be odd

    Notes
    -----
    This code is adapted from pylearn2. See the module docstring for license
    information.
    """

    def __init__(self, incoming, alpha=1e-4, k=2, beta=0.75, n=5, **kwargs):
        super(LocalResponseNormalization2DLayer, self).__init__(incoming,
                                                                **kwargs)
        self.alpha = alpha
        self.k = k
        self.beta = beta
        self.n = n
        if n % 2 == 0:
            raise NotImplementedError("Only works with odd n")

    def get_output_shape_for(self, input_shape):
        return input_shape

    def get_output_for(self, input, **kwargs):
        input_shape = self.input_shape
        if any(s is None for s in input_shape):
            input_shape = input.shape
        half_n = self.n // 2
        input_sqr = T.sqr(input)
        b, ch, r, c = input_shape
        extra_channels = T.alloc(0., b, ch + 2*half_n, r, c)
        input_sqr = T.set_subtensor(extra_channels[:, half_n:half_n+ch, :, :],
                                    input_sqr)
        scale = self.k
        for i in range(self.n):
            scale += self.alpha * input_sqr[:, i:i+ch, :, :]
        scale = scale ** self.beta
        return input / scale

#--- BUG due to Theano's problematic default_update functionality
class BatchNormLayer_Theano(Module):
    """
    This is an implementation using Theano's batch normalization op & .default_update mechanism
    till 2-23-2017 version, Theano still has bug to compile this implementation correctly.
    """
    def __init__(self, incoming, axes='auto', epsilon=1e-4, alpha=0.1,
                 beta=init.Constant(0), gamma=init.Constant(1),
                 mean=init.Constant(0), inv_std=init.Constant(1),
                 mode='high_mem',                                               # [DV] add mode support
                 **kwargs):
        self.mode = mode
        super(BatchNormLayer_Theano, self).__init__(incoming, **kwargs)

        if axes == 'auto':
            # default: normalize over all but the second axis
            axes = (0,) + tuple(range(2, len(self.input_shape)))
        elif isinstance(axes, int):
            axes = (axes,)
        self.axes = axes

        self.epsilon = epsilon
        self.alpha = alpha

        # create parameters, ignoring all dimensions in axes
        shape = [size for axis, size in enumerate(self.input_shape) if axis not in self.axes]  # remove all dimensions in axes
        if any(size is None for size in shape):
            raise ValueError("BatchNormLayer needs specified input sizes for "
                             "all axes not normalized over.")
        if beta is None:
            self.beta = None
        else:
            self.beta = self.add_param(beta, shape, 'beta',
                                       trainable=True, regularizable=False)
        if gamma is None:
            self.gamma = None
        else:
            self.gamma = self.add_param(gamma, shape, 'gamma',
                                        trainable=True, regularizable=True)
        self.mean = self.add_param(mean, shape, 'mean',
                                   trainable=False, regularizable=False)
        self.inv_std = self.add_param(inv_std, shape, 'inv_std',
                                      trainable=False, regularizable=False)

    def get_output_for(self, input, deterministic=False,
                       batch_norm_use_averages=None,
                       batch_norm_update_averages=None, **kwargs):
        # [DV] change the updating logic of mean and inv_std, replace dimshuffle with reshape, which is more memory efficient
        input_mean = input.mean(self.axes)
        input_inv_std = T.inv(T.sqrt(input.var(self.axes) + self.epsilon))

        # Decide whether to update the stored averages
        if batch_norm_update_averages is None:
            batch_norm_update_averages = not deterministic
        update_averages = batch_norm_update_averages

        if update_averages:
            # print('update_averages is on!')
            self.mean.default_update = ((1 - self.alpha) * self.mean + self.alpha * input_mean)
            self.inv_std.default_update = ((1 - self.alpha) * self.inv_std + self.alpha * input_inv_std)


        # Decide whether to use the stored averages or mini-batch statistics
        if batch_norm_use_averages is None:
            batch_norm_use_averages = deterministic
        use_averages = batch_norm_use_averages

        if use_averages:
            mean = self.mean
            inv_std = self.inv_std
        else:
            mean = input_mean
            inv_std = input_inv_std



            # # Trick: To update the stored statistics, we create memory-aliased
            # # clones of the stored statistics:
            # running_mean = theano.clone(self.mean, share_inputs=False)
            # running_inv_std = theano.clone(self.inv_std, share_inputs=False)
            # # set a default update for them:
            # running_mean.default_update = ((1 - self.alpha) * running_mean +
            #                                self.alpha * input_mean)
            # running_inv_std.default_update = ((1 - self.alpha) *
            #                                   running_inv_std +
            #                                   self.alpha * input_inv_std)
            # # and make sure they end up in the graph without participating in
            # # the computation (this way their default_update will be collected
            # # and applied, but the computation will be optimized away):
            # mean += 0 * running_mean
            # inv_std += 0 * running_inv_std

        # prepare dimshuffle pattern inserting broadcastable axes as needed
        # param_axes = iter(range(input.ndim - len(self.axes)))
        # pattern = ['x' if input_axis in self.axes
        #            else next(param_axes)
        #            for input_axis in range(input.ndim)]

        # apply dimshuffle pattern to all parameters
        # beta = 0 if self.beta is None else self.beta.dimshuffle(pattern)
        # gamma = 1 if self.gamma is None else self.gamma.dimshuffle(pattern)
        # mean = mean.dimshuffle(pattern)
        # inv_std = inv_std.dimshuffle(pattern)

        # normalize
        # normalized = (input - mean) * (gamma * inv_std) + beta

        broadcast_shape = [1] * input.ndim
        for i in range(input.ndim):
            if i not in self.axes:
                broadcast_shape[i] = self.input_shape[i]  # broadcast_shape = [1, C, 1, 1]
        mean = mean.reshape(broadcast_shape)
        inv_std = inv_std.reshape(broadcast_shape)
        beta = 0 if self.beta is None else T.reshape(self.beta, broadcast_shape)
        gamma = 1 if self.gamma is None else T.reshape(self.gamma, broadcast_shape)


        normalized = T.nnet.bn.batch_normalization(input, gamma, beta, mean, T.inv(inv_std),
                                           mode=self.mode)
        return normalized

#--- consume less memory than Lasagne's implementation
class BatchNormLayer_DV(Module):
    """
    """
    def __init__(self, incoming, axes='auto', epsilon=1e-4, alpha=0.1,
                 beta=init.Constant(0), gamma=init.Constant(1),
                 mean=init.Constant(0), inv_std=init.Constant(1),
                 mode='high_mem',  # [DV] add mode support
                 **kwargs):
        self.mode = mode
        super(BatchNormLayer_DV, self).__init__(incoming, **kwargs)

        if axes == 'auto':
            # default: normalize over all but the second axis
            axes = (0,) + tuple(range(2, len(self.input_shape)))
        elif isinstance(axes, int):
            axes = (axes,)
        self.axes = axes

        self.epsilon = epsilon
        self.alpha = alpha

        # create parameters, ignoring all dimensions in axes
        shape = [size for axis, size in enumerate(self.input_shape) if
                 axis not in self.axes]  # remove all dimensions in axes
        if any(size is None for size in shape):
            raise ValueError("BatchNormLayer needs specified input sizes for "
                             "all axes not normalized over.")
        if beta is None:
            self.beta = None
        else:
            self.beta = self.add_param(beta, shape, 'beta',
                                       trainable=True, regularizable=False)
        if gamma is None:
            self.gamma = None
        else:
            self.gamma = self.add_param(gamma, shape, 'gamma',
                                        trainable=True, regularizable=True)
        self.mean = self.add_param(mean, shape, 'mean',
                                   trainable=False, regularizable=False)
        self.inv_std = self.add_param(inv_std, shape, 'inv_std',
                                      trainable=False, regularizable=False)

    def get_output_for(self, input, deterministic=False,
                       batch_norm_use_averages=None,
                       batch_norm_update_averages=None, **kwargs):
        # [DV] change the updating logic of mean and inv_std, replace dimshuffle with reshape, which is more memory efficient
        input_mean = input.mean(self.axes)
        input_inv_std = T.inv(T.sqrt(input.var(self.axes) + self.epsilon))

        # Decide whether to update the stored averages
        if batch_norm_update_averages is None:
            batch_norm_update_averages = not deterministic
        update_averages = batch_norm_update_averages

        if update_averages:
            # print('update_averages is on!')
            self.add_update([[self.mean, (1 - self.alpha) * self.mean + self.alpha * input_mean],
                             [self.inv_std, (1 - self.alpha) * self.inv_std + self.alpha * input_inv_std]])

        # Decide whether to use the stored averages or mini-batch statistics
        if batch_norm_use_averages is None:
            batch_norm_use_averages = deterministic
        use_averages = batch_norm_use_averages

        if use_averages:
            mean = self.mean
            inv_std = self.inv_std
        else:
            mean = input_mean
            inv_std = input_inv_std

        broadcast_shape = [1] * input.ndim
        for i in range(input.ndim):
            if i not in self.axes:
                broadcast_shape[i] = self.input_shape[i]  # broadcast_shape = [1, C, 1, 1]
        mean = mean.reshape(broadcast_shape)
        inv_std = inv_std.reshape(broadcast_shape)
        beta = 0 if self.beta is None else T.reshape(self.beta, broadcast_shape)
        gamma = 1 if self.gamma is None else T.reshape(self.gamma, broadcast_shape)

        normalized = T.nnet.bn.batch_normalization(input, gamma, beta, mean, T.inv(inv_std),
                                                   mode=self.mode)
        return normalized

#--- this is the original Lasagne version ---#
class BatchNormLayer(Module):
    """
    lasagne.layers.BatchNormLayer(incoming, axes='auto', epsilon=1e-4,
    alpha=0.1, beta=lasagne.init.Constant(0), gamma=lasagne.init.Constant(1),
    mean=lasagne.init.Constant(0), inv_std=lasagne.init.Constant(1), **kwargs)

    Batch Normalization

    This layer implements batch normalization of its inputs, following [1]_:

    .. math::
        y = \\frac{x - \\mu}{\\sqrt{\\sigma^2 + \\epsilon}} \\gamma + \\beta

    That is, the input is normalized to zero mean and unit variance, and then
    linearly transformed. The crucial part is that the mean and variance are
    computed across the batch dimension, i.e., over examples, not per example.

    During training, :math:`\\mu` and :math:`\\sigma^2` are defined to be the
    mean and variance of the current input mini-batch :math:`x`, and during
    testing, they are replaced with average statistics over the training
    data. Consequently, this layer has four stored parameters: :math:`\\beta`,
    :math:`\\gamma`, and the averages :math:`\\mu` and :math:`\\sigma^2`
    (nota bene: instead of :math:`\\sigma^2`, the layer actually stores
    :math:`1 / \\sqrt{\\sigma^2 + \\epsilon}`, for compatibility to cuDNN).
    By default, this layer learns the average statistics as exponential moving
    averages computed during training, so it can be plugged into an existing
    network without any changes of the training procedure (see Notes).

    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape
    axes : 'auto', int or tuple of int
        The axis or axes to normalize over. If ``'auto'`` (the default),
        normalize over all axes except for the second: this will normalize over
        the minibatch dimension for dense layers, and additionally over all
        spatial dimensions for convolutional layers.
    epsilon : scalar
        Small constant :math:`\\epsilon` added to the variance before taking
        the square root and dividing by it, to avoid numerical problems
    alpha : scalar
        Coefficient for the exponential moving average of batch-wise means and
        standard deviations computed during training; the closer to one, the
        more it will depend on the last batches seen
    beta : Theano shared variable, expression, numpy array, callable or None
        Initial value, expression or initializer for :math:`\\beta`. Must match
        the incoming shape, skipping all axes in `axes`. Set to ``None`` to fix
        it to 0.0 instead of learning it.
        See :func:`lasagne.utils.create_param` for more information.
    gamma : Theano shared variable, expression, numpy array, callable or None
        Initial value, expression or initializer for :math:`\\gamma`. Must
        match the incoming shape, skipping all axes in `axes`. Set to ``None``
        to fix it to 1.0 instead of learning it.
        See :func:`lasagne.utils.create_param` for more information.
    mean : Theano shared variable, expression, numpy array, or callable
        Initial value, expression or initializer for :math:`\\mu`. Must match
        the incoming shape, skipping all axes in `axes`.
        See :func:`lasagne.utils.create_param` for more information.
    inv_std : Theano shared variable, expression, numpy array, or callable
        Initial value, expression or initializer for :math:`1 / \\sqrt{
        \\sigma^2 + \\epsilon}`. Must match the incoming shape, skipping all
        axes in `axes`.
        See :func:`lasagne.utils.create_param` for more information.
    **kwargs
        Any additional keyword arguments are passed to the :class:`Layer`
        superclass.

    Notes
    -----
    This layer should be inserted between a linear transformation (such as a
    :class:`DenseLayer`, or :class:`Conv2DLayer`) and its nonlinearity. The
    convenience function :func:`batch_norm` modifies an existing layer to
    insert batch normalization in front of its nonlinearity.

    The behavior can be controlled by passing keyword arguments to
    :func:`lasagne.layers.get_output()` when building the output expression
    of any network containing this layer.

    During training, [1]_ normalize each input mini-batch by its statistics
    and update an exponential moving average of the statistics to be used for
    validation. This can be achieved by passing ``deterministic=False``.
    For validation, [1]_ normalize each input mini-batch by the stored
    statistics. This can be achieved by passing ``deterministic=True``.

    For more fine-grained control, ``batch_norm_update_averages`` can be passed
    to update the exponential moving averages (``True``) or not (``False``),
    and ``batch_norm_use_averages`` can be passed to use the exponential moving
    averages for normalization (``True``) or normalize each mini-batch by its
    own statistics (``False``). These settings override ``deterministic``.

    Note that for testing a model after training, [1]_ replace the stored
    exponential moving average statistics by fixing all network weights and
    re-computing average statistics over the training data in a layerwise
    fashion. This is not part of the layer implementation.

    In case you set `axes` to not include the batch dimension (the first axis,
    usually), normalization is done per example, not across examples. This does
    not require any averages, so you can pass ``batch_norm_update_averages``
    and ``batch_norm_use_averages`` as ``False`` in this case.

    See also
    --------
    batch_norm : Convenience function to apply batch normalization to a layer

    References
    ----------
    .. [1] Ioffe, Sergey and Szegedy, Christian (2015):
           Batch Normalization: Accelerating Deep Network Training by Reducing
           Internal Covariate Shift. http://arxiv.org/abs/1502.03167.
    """
    def __init__(self, incoming, axes='auto', epsilon=1e-4, alpha=0.1,
                 beta=init.Constant(0), gamma=init.Constant(1),
                 mean=init.Constant(0), inv_std=init.Constant(1), **kwargs):
        super(BatchNormLayer, self).__init__(incoming, **kwargs)

        if axes == 'auto':
            # default: normalize over all but the second axis
            axes = (0,) + tuple(range(2, len(self.input_shape)))
        elif isinstance(axes, int):
            axes = (axes,)
        self.axes = axes

        self.epsilon = epsilon
        self.alpha = alpha

        # create parameters, ignoring all dimensions in axes
        shape = [size for axis, size in enumerate(self.input_shape)
                 if axis not in self.axes]
        if any(size is None for size in shape):
            raise ValueError("BatchNormLayer needs specified input sizes for "
                             "all axes not normalized over.")
        if beta is None:
            self.beta = None
        else:
            self.beta = self.add_param(beta, shape, 'beta',
                                       trainable=True, regularizable=False)
        if gamma is None:
            self.gamma = None
        else:
            self.gamma = self.add_param(gamma, shape, 'gamma',
                                        trainable=True, regularizable=True)
        self.mean = self.add_param(mean, shape, 'mean',
                                   trainable=False, regularizable=False)
        self.inv_std = self.add_param(inv_std, shape, 'inv_std',
                                      trainable=False, regularizable=False)

    def get_output_for(self, input, deterministic=False,
                       batch_norm_use_averages=None,
                       batch_norm_update_averages=None, **kwargs):
        input_mean = input.mean(self.axes)
        input_inv_std = T.inv(T.sqrt(input.var(self.axes) + self.epsilon))

        # Decide whether to use the stored averages or mini-batch statistics
        if batch_norm_use_averages is None:
            batch_norm_use_averages = deterministic
        use_averages = batch_norm_use_averages

        if use_averages:
            mean = self.mean
            inv_std = self.inv_std
        else:
            mean = input_mean
            inv_std = input_inv_std

        # Decide whether to update the stored averages
        if batch_norm_update_averages is None:
            batch_norm_update_averages = not deterministic
        update_averages = batch_norm_update_averages

        if update_averages:
            # Trick: To update the stored statistics, we create memory-aliased
            # clones of the stored statistics:
            running_mean = theano.clone(self.mean, share_inputs=False)
            running_inv_std = theano.clone(self.inv_std, share_inputs=False)
            # set a default update for them:
            running_mean.default_update = ((1 - self.alpha) * running_mean +
                                           self.alpha * input_mean)
            running_inv_std.default_update = ((1 - self.alpha) *
                                              running_inv_std +
                                              self.alpha * input_inv_std)
            # and make sure they end up in the graph without participating in
            # the computation (this way their default_update will be collected
            # and applied, but the computation will be optimized away):
            mean += 0 * running_mean
            inv_std += 0 * running_inv_std

        # prepare dimshuffle pattern inserting broadcastable axes as needed
        param_axes = iter(range(input.ndim - len(self.axes)))
        pattern = ['x' if input_axis in self.axes
                   else next(param_axes)
                   for input_axis in range(input.ndim)]

        # apply dimshuffle pattern to all parameters
        beta = 0 if self.beta is None else self.beta.dimshuffle(pattern)
        gamma = 1 if self.gamma is None else self.gamma.dimshuffle(pattern)
        mean = mean.dimshuffle(pattern)
        inv_std = inv_std.dimshuffle(pattern)

        # normalize
        normalized = (input - mean) * (gamma * inv_std) + beta
        return normalized

def batch_norm(layer, **kwargs):
    """
    Apply batch normalization to an existing layer. This is a convenience
    function modifying an existing layer to include batch normalization: It
    will steal the layer's nonlinearity if there is one (effectively
    introducing the normalization right before the nonlinearity), remove
    the layer's bias if there is one (because it would be redundant), and add
    a :class:`BatchNormLayer` and :class:`NonlinearityLayer` on top.

    Parameters
    ----------
    layer : A :class:`Layer` instance
        The layer to apply the normalization to; note that it will be
        irreversibly modified as specified above
    **kwargs
        Any additional keyword arguments are passed on to the
        :class:`BatchNormLayer` constructor.

    Returns
    -------
    BatchNormLayer or NonlinearityLayer instance
        A batch normalization layer stacked on the given modified `layer`, or
        a nonlinearity layer stacked on top of both if `layer` was nonlinear.

    Examples
    --------
    Just wrap any layer into a :func:`batch_norm` call on creating it:

    >> from lasagne.layers import InputLayer, DenseLayer, batch_norm
    >> from lasagne.nonlinearities import tanh
    >> l1 = InputLayer((64, 768))
    >> l2 = batch_norm(DenseLayer(l1, num_units=500, nonlinearity=tanh))

    This introduces batch normalization right before its nonlinearity:

    >> from lasagne.layers import get_all_layers
    >> [l.__class__.__name__ for l in get_all_layers(l2)]
    ['InputLayer', 'DenseLayer', 'BatchNormLayer', 'NonlinearityLayer']
    """
    nonlinearity = getattr(layer, 'nonlinearity', None)
    if nonlinearity is not None:
        layer.nonlinearity = nonlinearities.identity
    if hasattr(layer, 'b') and layer.b is not None:
        del layer.params['b']
        layer.b = None
    bn_name = (kwargs.pop('name', None) or (getattr(layer, 'name', None) and layer.name + '_bn'))
    #--- added by [DV]
    ver = kwargs.pop('ver', None)
    if ver == 'DV':
        layer = BatchNormLayer_DV(layer, name=bn_name, **kwargs)
    elif ver == 'Theano':
        layer = BatchNormLayer_Theano(layer, name=bn_name, **kwargs)
    else:
        layer = BatchNormLayer(layer, name=bn_name, **kwargs)

    if nonlinearity is not None:
        from .special import NonlinearityLayer
        nonlin_name = bn_name and bn_name + '_nonlin'
        layer = NonlinearityLayer(layer, nonlinearity, name=nonlin_name)
    return layer



if __name__ == '__main__':
    INFO = ['This is Python script template.\n',
            'Author: David Leon\n',
            'All rights reserved\n']
    print(*INFO)