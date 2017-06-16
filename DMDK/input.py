# coding:utf-8
# <Descriptions>
# Created   :   6, 14, 2017
# Revised   :   6, 14, 2017
# All rights reserved
#------------------------------------------------------------------------------------------------
__author__ = 'dawei.leng'

from collections import OrderedDict
from base import Layer
import numpy as np

__all__ = [
    "InputLayer",
]


class InputLayer(Layer):
    """
    This layer holds a symbolic variable that represents a network input. A
    variable can be specified when the layer is instantiated, else it is
    created.

    Parameters
    ----------
    shape : tuple of `int` or `None` elements
        The shape of the input. Any element can be `None` to indicate that the
        size of that dimension is not fixed at compile time.

    input_var : Theano symbolic variable or `None` (default: `None`)
        A variable representing a network input. If it is not provided, a
        variable will be created.

    Raises
    ------
    ValueError
        If the dimension of `input_var` is not equal to `len(shape)`

    Notes
    -----
    The first dimension usually indicates the batch size. If you specify it,
    Theano may apply more optimizations while compiling the training or
    prediction function, but the compiled function will not accept data of a
    different batch size at runtime. To compile for a variable batch size, set
    the first shape element to `None` instead.

    Examples
    --------
    >>> from lasagne.layers import InputLayer
    >>> l_in = InputLayer((100, 20))
    """
    def __init__(self, shape, input_var=None, name=None, **kwargs):
        self.shape = tuple(shape)
        if any(d is not None and d <= 0 for d in self.shape):
            raise ValueError((
                "Cannot create InputLayer with a non-positive shape "
                "dimension. shape=%r, self.name=%r") % ( self.shape, name))

        ndim = len(self.shape)
        if input_var is None:
            input_var = np.empty(shape)
        else:
            # ensure the given variable has the correct dimensionality
            if input_var.ndim != ndim:
                raise ValueError("shape has %d dimensions, but variable has "
                                 "%d" % (ndim, input_var.ndim))
        self.input_var = input_var
        self.name = name
        self.params = OrderedDict()

    @Layer.output_shape.getter
    def output_shape(self):
        return self.shape




if __name__ == '__main__':
    INFO = ['This is Python script template.\n',
            'Author: David Leon\n',
            'All rights reserved\n']
    print(*INFO)