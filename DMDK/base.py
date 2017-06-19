# coding:utf-8
# base class definitions
# Created   :   6, 12, 2017
# Revised   :   6, 12, 2017
# All rights reserved
#------------------------------------------------------------------------------------------------
__author__ = 'dawei.leng'

from collections import OrderedDict
import utils
import numpy as np

class Layer(object):
    """
    The :class:`Layer` class represents a single layer of a neural network. It
    should be subclassed when implementing new types of layers.

    Because each layer can keep track of the layer(s) feeding into it, a
    network's output :class:`Layer` instance can double as a handle to the full
    network.

    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape.
    name : a string or None
        An optional name to attach to this layer.
    """
    def __init__(self, incoming, name=None):
        if isinstance(incoming, tuple):
            self.input_shape = incoming
            self.input_layer = None
        else:
            self.input_shape = incoming.output_shape
            self.input_layer = incoming

        self.name = name
        self.params = OrderedDict()

    @property
    def output_shape(self):
        shape = self.get_output_shape_for(self.input_shape)
        return shape

    def get_params(self, unwrap_shared=True, **tags):
        """
        Returns a list of Theano shared variables or expressions that
        parameterize the layer.

        By default, all shared variables that participate in the forward pass
        will be returned (in the order they were registered in the Layer's
        constructor via :meth:`add_param()`). The list can optionally be
        filtered by specifying tags as keyword arguments. For example,
        ``trainable=True`` will only return trainable parameters, and
        ``regularizable=True`` will only return parameters that can be
        regularized (e.g., by L2 decay).

        If any of the layer's parameters was set to a Theano expression instead
        of a shared variable, `unwrap_shared` controls whether to return the
        shared variables involved in that expression (``unwrap_shared=True``,
        the default), or the expression itself (``unwrap_shared=False``). In
        either case, tag filtering applies to the expressions, considering all
        variables within an expression to be tagged the same.

        Parameters
        ----------
        unwrap_shared : bool (default: True)
            Affects only parameters that were set to a Theano expression. If
            ``True`` the function returns the shared variables contained in
            the expression, otherwise the Theano expression itself.

        **tags (optional)
            tags can be specified to filter the list. Specifying ``tag1=True``
            will limit the list to parameters that are tagged with ``tag1``.
            Specifying ``tag1=False`` will limit the list to parameters that
            are not tagged with ``tag1``. Commonly used tags are
            ``regularizable`` and ``trainable``.

        Returns
        -------
        list of Theano shared variables or expressions
            A list of variables that parameterize the layer

        Notes
        -----
        For layers without any parameters, this will return an empty list.
        """
        result = list(self.params.keys())

        only = set(tag for tag, value in tags.items() if value)
        if only:
            # retain all parameters that have all of the tags in `only`
            result = [param for param in result
                      if not (only - self.params[param])]

        exclude = set(tag for tag, value in tags.items() if not value)
        if exclude:
            # retain all parameters that have none of the tags in `exclude`
            result = [param for param in result
                      if not (self.params[param] & exclude)]

        if unwrap_shared:
            return utils.collect_shared_vars(result)
        else:
            return result

    def get_output_shape_for(self, input_shape):
        """
        Computes the output shape of this layer, given an input shape.

        Parameters
        ----------
        input_shape : tuple
            A tuple representing the shape of the input. The tuple should have
            as many elements as there are input dimensions, and the elements
            should be integers or `None`.

        Returns
        -------
        tuple
            A tuple representing the shape of the output of this layer. The
            tuple has as many elements as there are output dimensions, and the
            elements are all either integers or `None`.

        Notes
        -----
        This method will typically be overridden when implementing a new
        :class:`Layer` class. By default it simply returns the input
        shape. This means that a layer that does not modify the shape
        (e.g. because it applies an elementwise operation) does not need
        to override this method.
        """
        return input_shape

    def get_output_for(self, input, **kwargs):
        """
        Propagates the given input through this layer (and only this layer).

        Parameters
        ----------
        input : Theano expression
            The expression to propagate through this layer.

        Returns
        -------
        output : Theano expression
            The output of this layer given the input to this layer.


        Notes
        -----
        This is called by the base :meth:`lasagne.layers.get_output()`
        to propagate data through a network.

        This method should be overridden when implementing a new
        :class:`Layer` class. By default it raises `NotImplementedError`.
        """
        raise NotImplementedError

    def add_param(self, spec, shape, **tags):
        """
        register tunable model parameters
        :param spec: np.ndarray or expression generating np.ndarray
        :param shape:
        :param tags:
        :return:
        """
        param = utils.create_param(spec, shape)  # np.ndarray
        # parameters should be trainable and regularizable by default
        tags['trainable'] = tags.get('trainable', True)     # todo: check these 2 lines whether should be kept finally?
        tags['regularizable'] = tags.get('regularizable', True)
        self.params[param] = set(tag for tag, value in tags.items() if value)

        return param


def _addindent(s_, numSpaces):
    s = s_.split('\n')
    # dont do anything for single-line stuff
    if len(s) == 1:
        return s_
    first = s.pop(0)
    s = [(numSpaces * ' ') + line for line in s]
    s = '\n'.join(s)
    s = first + '\n' + s
    return s

class Module(object):
    """Base class for all neural network modules.

    Your models should also subclass this class.

    Modules can also contain other Modules, allowing to nest them in
    a tree structure. You can assign the submodules as regular attributes::

        import torch.nn as nn
        import torch.nn.functional as F

        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.conv1 = nn.Conv2d(1, 20, 5)
                self.conv2 = nn.Conv2d(20, 20, 5)

            def forward(self, x):
               x = F.relu(self.conv1(x))
               return F.relu(self.conv2(x))

    Submodules assigned in this way will be registered, and will have their
    parameters converted too when you call .cuda(), etc.
    """

    dump_patches = False

    def __init__(self, incoming, name=None):

        if isinstance(incoming, tuple):  # incoming = input shape
            self.input_shape = incoming
            self.input_layer = None
        else:                            # incoming = instance of previous layer
            self.input_shape = incoming.output_shape
            self.input_layer = incoming

        self.name = name
        self.parameters    = OrderedDict()
        self.buffers       = OrderedDict()
        self.modules       = OrderedDict()

    def forward(self, *input):
        """Defines the computation performed at every call.

        Should be overriden by all subclasses.
        """
        raise NotImplementedError

    def register_buffer(self, name, tensor):
        """Adds a persistent buffer to the module.

        This is typically used to register a buffer that should not to be
        considered a model parameter. For example, BatchNorm's ``running_mean``
        is not a parameter, but is part of the persistent state.

        Buffers can be accessed as attributes using given names.

        Example:
            >> self.register_buffer('running_mean', torch.zeros(num_features))
        """
        self.buffers[name] = tensor

    def register_parameter(self, name, param):
        """Adds a parameter to the module.

        The parameter can be accessed as an attribute using given name.
        """
        if 'parameters' not in self.__dict__:
            raise AttributeError(
                "cannot assign parameter before Module.__init__() call")
        self.parameters[name] = param

    def add_param(self, spec, shape, name):
        """
        register tunable model parameters
        :param spec: np.ndarray or expression generating np.ndarray
        :param shape:
        :param tags:
        :return:
        """
        param = utils.create_param(spec, shape)  # np.ndarray
        # parameters should be trainable and regularizable by default
        # tags['trainable'] = tags.get('trainable', True)     # todo: check these 2 lines whether should be kept finally?
        # tags['regularizable'] = tags.get('regularizable', True)
        # self.params[param] = set(tag for tag, value in tags.items() if value)
        self.parameters[name] = param
        return param

    def add_module(self, name, module):
        """Adds a child module to the current module.

        The module can be accessed as an attribute using the given name.
        """
        if hasattr(self, name):
            raise KeyError("attribute already exists '{}'".format(name))
        if not isinstance(module, Module) and module is not None:
            raise TypeError("{} is not a Module subclass".format(type(module)))
        self.modules[name] = module

    def _apply(self, fn):
        for module in self.children():
            module._apply(fn)

        for param in self.parameters.values():
            if param is not None:
                # Variables stored in modules are graph leaves, and we don't
                # want to create copy nodes, so we have to unpack the data.
                param = fn(param)

        for key, buf in self.buffers.items():
            if buf is not None:
                self.buffers[key] = fn(buf)
        return self

    def apply(self, fn):
        for module in self.children():
            module.apply(fn)
        fn(self)
        return self

    def __call__(self, *input, **kwargs):
        """
        Equivalent to pytorch's Module.__call__(), except that all forword_hooks are disabled for now.s
        :param input:
        :param kwargs:
        :return:
        """

        result = self.forward(*input, **kwargs)
        return result

    def __getattr__(self, name):
        if 'parameters' in self.__dict__:
            parameters = self.__dict__['parameters']
            if name in parameters:
                return parameters[name]
        if 'buffers' in self.__dict__:
            buffers = self.__dict__['buffers']
            if name in buffers:
                return buffers[name]
        if 'modules' in self.__dict__:
            modules = self.__dict__['modules']
            if name in modules:
                return modules[name]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, name))

    def __setattr__(self, name, value):
        def remove_from(*dicts):
            for d in dicts:
                if name in d:
                    del d[name]

        if isinstance(value, np.ndarray):
            self.register_parameter(name, value)
        else:
            modules = self.__dict__.get('modules')
            if isinstance(value, Module):
                if modules is None:
                    raise AttributeError(
                        "cannot assign module before Module.__init__() call")
                remove_from(self.__dict__, self.parameters, self.buffers)
                modules[name] = value
            elif modules is not None and name in modules:
                if value is not None:
                    raise TypeError("cannot assign '{}' as child module '{}' "
                                    "(torch.nn.Module or None expected)"
                                    .format(type(value), name))
                modules[name] = value
            else:
                buffers = self.__dict__.get('buffers')
                if buffers is not None and name in buffers:
                    buffers[name] = value
                else:
                    object.__setattr__(self, name, value)

    def __delattr__(self, name):
        if name in self.parameters:
            del self.parameters[name]
        elif name in self.buffers:
            del self.buffers[name]
        elif name in self.modules:
            del self.modules[name]
        else:
            object.__delattr__(self, name)

    def state_dict(self, destination=None, prefix=''):
        """Returns a dictionary containing a whole state of the module.

        Both parameters and persistent buffers (e.g. running averages) are
        included. Keys are corresponding parameter and buffer names.

        Example:
            >> module.state_dict().keys()
            ['bias', 'weight']
        """
        if destination is None:
            destination = OrderedDict()
        for name, param in self.parameters.items():
            if param is not None:
                destination[prefix + name] = param
        for name, buf in self.buffers.items():
            if buf is not None:
                destination[prefix + name] = buf
        for name, module in self.modules.items():
            if module is not None:
                module.state_dict(destination, prefix + name + '.')
        return destination

    def load_state_dict(self, state_dict):
        """Copies parameters and buffers from :attr:`state_dict` into
        this module and its descendants. The keys of :attr:`state_dict` must
        exactly match the keys returned by this module's :func:`state_dict()`
        function.

        Arguments:
            state_dict (dict): A dict containing parameters and
                persistent buffers.
        """
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                raise KeyError('unexpected key "{}" in state_dict'
                               .format(name))
            own_state[name].copy_(param)

        missing = set(own_state.keys()) - set(state_dict.keys())
        if len(missing) > 0:
            raise KeyError('missing keys in state_dict: "{}"'.format(missing))

    def parameters(self):
        """Returns an iterator over module parameters.

        This is typically passed to an optimizer.

        Example:
            >> for param in model.parameters():
            >>     print(type(param), param.size())
            <class 'torch.FloatTensor'> (20L,)
            <class 'torch.FloatTensor'> (20L, 1L, 5L, 5L)
        """
        for name, param in self.named_parameters():
            yield param

    def named_parameters(self, memo=None, prefix=''):
        """Returns an iterator over module parameters, yielding both the
        name of the parameter as well as the parameter itself

        Example:
            >> for name, param in self.named_parameters():
            >>    if name in ['bias']:
            >>        print(param.size())
        """
        if memo is None:
            memo = set()
        for name, p in self.parameters.items():
            if p is not None and p not in memo:
                memo.add(p)
                yield prefix + ('.' if prefix else '') + name, p
        for mname, module in self.named_children():
            submodule_prefix = prefix + ('.' if prefix else '') + mname
            for name, p in module.named_parameters(memo, submodule_prefix):
                yield name, p

    def children(self):
        """Returns an iterator over immediate children modules."""
        for name, module in self.named_children():
            yield module

    def named_children(self):
        """Returns an iterator over immediate children modules, yielding both
        the name of the module as well as the module itself.

        Example:
            >> for name, module in model.named_children():
            >>     if name in ['conv4', 'conv5']:
            >>         print(module)
        """
        memo = set()
        for name, module in self.modules.items():
            if module is not None and module not in memo:
                memo.add(module)
                yield name, module

    def modules(self):
        """Returns an iterator over all modules in the network.

        Note:
            Duplicate modules are returned only once. In the following
            example, ``l`` will be returned only once.

            >> l = nn.Linear(2, 2)
            >> net = nn.Sequential(l, l)
            >> for idx, m in enumerate(net.modules()):
            >>     print(idx, '->', m)
            0 -> Sequential (
              (0): Linear (2 -> 2)
              (1): Linear (2 -> 2)
            )
            1 -> Linear (2 -> 2)
        """
        for name, module in self.named_modules():
            yield module

    def named_modules(self, memo=None, prefix=''):
        """Returns an iterator over all modules in the network, yielding
        both the name of the module as well as the module itself.

        Note:
            Duplicate modules are returned only once. In the following
            example, ``l`` will be returned only once.

            >> l = nn.Linear(2, 2)
            >> net = nn.Sequential(l, l)
            >> for idx, m in enumerate(net.named_modules()):
            >>     print(idx, '->', m)
            0 -> ('', Sequential (
              (0): Linear (2 -> 2)
              (1): Linear (2 -> 2)
            ))
            1 -> ('0', Linear (2 -> 2))
        """

        if memo is None:
            memo = set()
        if self not in memo:
            memo.add(self)
            yield prefix, self
            for name, module in self.modules.items():
                submodule_prefix = prefix + ('.' if prefix else '') + name
                for m in module.named_modules(memo, submodule_prefix):
                    yield m

    def share_memory(self):
        return self._apply(lambda t: t.share_memory_())

    def __repr__(self):
        tmpstr = self.__class__.__name__ + ' (\n'
        for key, module in self.modules.items():
            modstr = module.__repr__()
            modstr = _addindent(modstr, 2)
            tmpstr = tmpstr + '  (' + key + '): ' + modstr + '\n'
        tmpstr = tmpstr + ')'
        return tmpstr

    def __dir__(self):
        module_attrs = dir(self.__class__)
        attrs = list(self.__dict__.keys())
        parameters = list(self.parameters.keys())
        modules = list(self.modules.keys())
        buffers = list(self.buffers.keys())
        keys = module_attrs + attrs + parameters + modules + buffers
        return sorted(keys)
