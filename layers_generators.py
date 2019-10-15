# -*- coding: utf-8 -*-

"""
    @File name    :    layers_generators.py
    @Date         :    2019-10-15 14:45
    @Description  :    generate layer and its decode layer
    @Author       :    VickeeX
"""
from functools import reduce
from parl.core.fluid import layers


def fc_generator(shape, size, num_flatten_dims=1, param_attr=None, bias_attr=None, act=None, name=None):
    fc = layers.fc(size=size,
                   num_flatten_dims=num_flatten_dims,
                   param_attr=param_attr,
                   bias_attr=bias_attr,
                   act=act,
                   name=name)

    output_shape = shape[:num_flatten_dims] + [size]
    # TODO: we usually flatten before fc, so reduce is not essential.
    desize = reduce(lambda x, y: x * y, shape[num_flatten_dims:])

    defc = layers.fc(size=desize,
                     name=str('de' + name) if name else None,
                     num_flatten_dims=num_flatten_dims,
                     param_attr=param_attr,
                     bias_attr=bias_attr,
                     act=act)
    print(size, desize, output_shape)
    return fc, defc, output_shape


def conv2d_generator(shape, num_filters, filter_size, stride=1, padding=0, dilation=1, groups=None,
                     param_attr=None, bias_attr=None, use_cudnn=True, act=None, name=None):
    enc = layers.conv2d(num_filters=num_filters,
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

    def compute_conv2d_HW(HW):
        # TODO: while padding is an list: padding[0]!=paddings[1]
        # TODO: while groups!=1
        HW = (HW + 2 * padding - (dilation * (filter_size - 1) + 1)) // stride + 1
        return HW

    H_, W_ = compute_conv2d_HW(shape[2]), compute_conv2d_HW(shape[3])
    output_shape = [shape[0], num_filters, H_, W_]
    de_padding = ((H_ - 1) * stride + dilation * (filter_size - 1) + 1 - shape[2]) // 2

    dec = layers.conv2d_transpose(num_filters=shape[1],
                                  name=str('de' + name) if name else None,
                                  padding=de_padding,
                                  filter_size=filter_size,
                                  stride=stride,
                                  dilation=dilation,
                                  param_attr=param_attr,
                                  bias_attr=bias_attr,
                                  use_cudnn=use_cudnn,
                                  act=act)
    return enc, dec, output_shape
