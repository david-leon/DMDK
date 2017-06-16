# coding:utf-8
# <Descriptions>
# Created   :  mm, dd, yyyy
# Revised   :  mm, dd, yyyy
# All rights reserved
#------------------------------------------------------------------------------------------------
__author__ = 'dawei.leng'

import numpy as np
import numbers

def as_tuple(x, N, t=None):
    """
    Coerce a value to a tuple of given length (and possibly given type).

    Parameters
    ----------
    x : value or iterable
    N : integer
        length of the desired tuple
    t : type, optional
        required type for all elements

    Returns
    -------
    tuple
        ``tuple(x)`` if `x` is iterable, ``(x,) * N`` otherwise.

    Raises
    ------
    TypeError
        if `type` is given and `x` or any of its elements do not match it
    ValueError
        if `x` is iterable, but does not have exactly `N` elements
    """
    try:
        X = tuple(x)
    except TypeError:
        X = (x,) * N

    if (t is not None) and not all(isinstance(v, t) for v in X):
        raise TypeError("expected a single value or an iterable "
                        "of {0}, got {1} instead".format(t.__name__, x))

    if len(X) != N:
        raise ValueError("expected a single value or an iterable "
                         "with length {0}, got {1} instead".format(N, x))

    return X

def create_param(spec, shape):
    """
    Equivalent to Lasagne's create_param()

    Parameters
    ----------
    spec : scalar number, numpy array, Theano expression, or callable
        Either of the following:

        * a scalar or a numpy array with the initial parameter values
        * a Theano expression or shared variable representing the parameters
        * a function or callable that takes the desired shape of
          the parameter array as its single argument and returns
          a numpy array, a Theano expression, or a shared variable
          representing the parameters.

    shape : iterable of int
        a tuple or other iterable of integers representing the desired
        shape of the parameter array.

    Returns np.ndarray
    -------
    Notes
    -----
    This function is called by :meth:`Layer.add_param()` in the constructor
    of most :class:`Layer` subclasses. This enables those layers to
    support initialization with scalars, numpy arrays, existing Theano shared
    variables or expressions, and callables for generating initial parameter
    values, Theano expressions, or shared variables.
    """
    shape = tuple(shape)  # convert to tuple if needed

    if callable(spec):
        spec = spec(shape)

    if isinstance(spec, numbers.Number) or isinstance(spec, np.generic) and spec.dtype.kind in 'biufc':
        spec = np.asarray(spec)

    return spec



if __name__ == '__main__':
    INFO = ['This is Python script template.\n',
            'Author: David Leon\n',
            'All rights reserved\n']
    print(*INFO)