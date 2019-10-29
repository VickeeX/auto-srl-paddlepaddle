# -*- coding: utf-8 -*-

"""
    @File name    :    flatten_reshape_wrapper.py
    @Date         :    2019-10-15 16:05
    @Description  :    {TODO}
    @Author       :    VickeeX
"""


class Flatten(object):
    def __init__(self, axis=1, name=None):
        self.axis = axis
        self.name = str('flatten' + name) if name else 'flatten'


class Reshape(object):
    def __init__(self, shape, actual_shape=None, act=None, inplace=False, name=None):
        self.shape = shape
        self.actual_shape = actual_shape
        self.act = act
        self.inplace = inplace
        self.name = str('reshape' + name) if name else 'reshape'
