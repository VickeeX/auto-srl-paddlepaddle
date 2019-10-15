# -*- coding: utf-8 -*-

"""
    @File name    :    layers_recorder_helper.py
    @Date         :    2019-10-15 16:10
    @Description  :    {TODO}
    @Author       :    VickeeX
"""

from parl.core.fluid import layers
from flatten_reshape_wrapper import Flatten, Reshape
import json


def layer_recorder(args):
    with open('encoder_args_record', 'a+') as f:
        f.write(json.dumps(args) + '\n')


def conv2d_helper(num_filters, filter_size, stride=1, padding=0, dilation=1, groups=None,
                  param_attr=None, bias_attr=None, use_cudnn=True, act=None, name=None):
    conv2d = layers.conv2d(num_filters=num_filters,
                           filter_size=filter_size,
                           stride=stride,
                           padding=padding,
                           dilation=dilation,
                           groups=groups,
                           param_attr=param_attr,
                           bias_attr=bias_attr,
                           use_cudnn=use_cudnn,
                           act=act,
                           name=name)
    args = {"type": "conv2d",
            "num_filters": num_filters,
            "filter_size": filter_size,
            "stride": stride,
            "padding": padding,
            "dilation": dilation,
            "groups": groups,
            "param_attr": param_attr,
            "bias_attr": bias_attr,
            "use_cudnn": use_cudnn,
            "act": act,
            "name": name}

    layer_recorder(args)
    return conv2d


def fc_helepr(size, num_flatten_dims=1, param_attr=None, bias_attr=None, act=None, name=None):
    fc = layers.fc(size=size,
                   num_flatten_dims=num_flatten_dims,
                   param_attr=param_attr,
                   bias_attr=bias_attr,
                   act=act,
                   name=name)
    args = {"type": "fc",
            "size": size,
            "num_flatten_dims": num_flatten_dims,
            "param_attr": param_attr,
            "bias_attr": bias_attr,
            "act": act,
            "name": name}
    layer_recorder(args)
    return fc


def flatten_helper(axis=1, name=None):
    ft = Flatten(axis, name)
    args = {"type": "flatten",
            "axis": axis,
            "name": name}
    layer_recorder(args)
    return ft


def reshape_helper(actual_shape=None, act=None, inplace=False, name=None):
    rs = Reshape(actual_shape, act, inplace, name)
    args = {"type": "reshape",
            "actual_shape": actual_shape,
            "act": act,
            "inplace": inplace,
            "name": name}
    layer_recorder(args)
    return rs
