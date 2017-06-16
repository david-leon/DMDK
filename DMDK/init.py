# coding:utf-8
# Functions to create initializers for parameter variables.
# Created   :   6, 13, 2017
# Revised   :   6, 13, 2017
# All rights reserved
#------------------------------------------------------------------------------------------------
__author__ = 'dawei.leng'

import numpy as np
import os
if 'MODEL_PRECISION' in os.environ:
    floatX = os.environ['MODEL_PRECISION']
else:
    floatX = 'float32'  #todo: make this package global

class Initializer(object):
    """Base class for parameter tensor initializers.

    The :class:`Initializer` class represents a weight initializer used
    to initialize weight parameters in a neural network layer. It should be
    subclassed when implementing new types of weight initializers.

    """
    def __call__(self, shape):
        """
        Makes :class:`Initializer` instances callable like a function, invoking
        their :meth:`sample()` method.
        """
        return self.sample(shape)

    def sample(self, shape):
        """
        Sample should return a theano.tensor of size shape and data type
        theano.config.floatX.

        Parameters
        -----------
        shape : tuple or int
            Integer or tuple specifying the size of the returned
            matrix.
        returns : theano.tensor
            Matrix of size shape and dtype theano.config.floatX.
        """
        raise NotImplementedError()

class Constant(Initializer):
    """Initialize weights with constant value.

    Parameters
    ----------
     val : float
        Constant value for weights.
    """
    def __init__(self, val=0.0):
        self.val = val

    def sample(self, shape):
        return np.ones(shape, floatX) * self.val
class Empty(Initializer):
    """
    Initialize weights with empty np.ndarray
    """
    def sample(self, shape):
        return np.empty(shape, floatX)

if __name__ == '__main__':
    INFO = ['This is Python script template.\n',
            'Author: David Leon\n',
            'All rights reserved\n']
    print(*INFO)