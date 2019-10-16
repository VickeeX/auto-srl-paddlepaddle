# -*- coding: utf-8 -*-

"""
    @File name    :    layers_decoder_generators.py
    @Date         :    2019-10-15 19:38
    @Description  :    {TODO}
    @Author       :    VickeeX
"""
from parl.core.fluid import layers
from flatten_reshape_wrapper import Flatten, Reshape
from functools import reduce


def conv2d_decoder(data, shape):
    num_filters = data['num_filters']
    filter_size = data['filter_size']
    stride = data['stride']
    padding = data['padding']
    dilation = data['dilation']
    param_attr = data['param_attr']
    bias_attr = data['bias_attr']
    use_cudnn = data['use_cudnn']
    act = data['act']
    name = data['name']

    def compute_conv2d_HW(HW):
        # TODO: while padding is an list: padding[0]!=paddings[1]
        # TODO: while groups!=1
        HW = (HW + 2 * padding - (dilation * (filter_size - 1) + 1)) // stride + 1
        return HW

    H_, W_ = compute_conv2d_HW(shape[2]), compute_conv2d_HW(shape[3])
    output_shape = [shape[0], num_filters, H_, W_]
    de_padding = ((H_ - 1) * stride + dilation * (filter_size - 1) + 1 - shape[2]) // 2

    layer = layers.conv2d_transpose(num_filters=shape[1],
                                    name=str('de' + name) if name else None,
                                    padding=de_padding,
                                    filter_size=filter_size,
                                    stride=stride,
                                    dilation=dilation,
                                    param_attr=param_attr,
                                    bias_attr=bias_attr,
                                    use_cudnn=use_cudnn,
                                    act=act)

    return layer, output_shape


def fc_decoder(data, shape):
    size = data['size']
    num_flatten_dims = data['num_flatten_dims']
    param_attr = data['param_attr']
    bias_attr = data['bias_attr']
    act = data['act']
    name = data['name']

    output_shape = shape[:num_flatten_dims] + [size]
    # TODO: we usually flatten before fc, so reduce is not essential.
    desize = reduce(lambda x, y: x * y, shape[num_flatten_dims:])

    layer = layers.fc(size=desize,
                      name=str('de' + name) if name else None,
                      num_flatten_dims=num_flatten_dims,
                      param_attr=param_attr,
                      bias_attr=bias_attr,
                      act=act)
    return layer, output_shape


def flatten_decoder(data, shape):
    name = data['name']
    output_shape = shape[:data['axis']] + [reduce(lambda x, y: x * y, shape[data['axis']:])]
    return Reshape(shape=shape, name=str('de' + name) if name else None), output_shape
