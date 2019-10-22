# -*- coding: utf-8 -*-

"""
    @File name    :    layers_decoder_model_template.py.py
    @Date         :    2019-10-21 21:16
    @Description  :    {TODO}
    @Author       :    VickeeX
"""

#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import parl
import json
from parl import layers
from functools import reduce


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


class AtariModel(parl.Model):
    def __init__(self, act_dim):
        self.conv1 = self.conv2d_helper(
            num_filters=32, filter_size=8, stride=4, padding=1, act='relu')
        self.conv2 = self.conv2d_helper(
            num_filters=64, filter_size=4, stride=2, padding=2, act='relu')
        self.conv3 = self.conv2d_helper(
            num_filters=64, filter_size=3, stride=1, padding=0, act='relu')
        self.flat = self.flatten_helper(axis=1)
        self.fc = self.fc_helepr(size=512, act='relu')

        self.policy_fc = layers.fc(size=act_dim)
        self.value_fc = layers.fc(size=1)

        self.encoder = [self.conv1, self.conv2, self.conv3, self.flat, self.fc]
        self.decoder = self.decoder_generator(shape=[-1, 4, 84, 84])

    def decoder_generator(self, shape):
        # switch = {
        #     'conv2d': lambda x, s: self.conv2d_decoder(x, s),
        #     'fc': lambda x, s: self.fc_decoder(x, s),
        #     'flatten': lambda x, s: self.flatten_decoder(x, s)
        # }
        # layer, shape = switch[data['type']](data, shape)

        decoder, datas = [], []
        with open('encoder_args_record', 'r') as f:
            for line in f:
                datas.append(json.loads(line.strip('\n')))

        # for i in range(len(datas)):
        #     datas[i]['last_type'] = 0 if i == 0 else datas[i - 1]['type']

        de_conv_sp, de_fc_sp, de_flatten_sp = shape, [], []
        # shapes = {0: de_conv_sp, 1: de_fc_sp, 2: de_flatten_sp}

        for data in datas:
            # x, y = data['type'], data['last_type']
            # if x == 0:
            #     deconv, shapes[x] = self.conv2d_decoder(data, shapes[y])
            #     decoder.append(deconv)
            # elif x == 1:
            #     defc, shapes[x] = self.fc_decoder(data, shapes[y])
            # else:
            #     shapes[x] = self.flatten_decoder(data, shapes[y])

            if data['type'] == 0:
                deconv, de_conv_sp = self.conv2d_decoder(data, de_conv_sp)
                decoder.append(deconv)
            elif data['type'] == 1:
                defc, de_fc_sp = self.fc_decoder(data, de_flatten_sp)
            elif data['type'] == 2:
                de_flatten_sp = self.flatten_decoder(data, de_conv_sp)
        # decoder = decoder[::-1]
        # decoder.append(defc)
        return decoder[::-1] + [defc]

        # def obs_encode_decode(self, code_tag, obs):

    #     coder = self.encoder if code_tag else self.decoder
    #     out = obs / 255.0
    #     for layer in coder:
    #         try:
    #             nm = layer.name
    #             if nm.startswith("flatten"):
    #                 out = layers.flatten(out, axis=layer.axis, name=layer.name)
    #             #  omit ```nm.startswith("reshape")``` , only "flatten" and "reshape"
    #             else:
    #                 out = layers.reshape(out, shape=layer.shape, name=layer.name)
    #         except AttributeError:
    #             out = layer(out)
    #     return out

    def obs_ae(self, obs):
        obs = obs / 255.0
        conv1 = self.conv1(obs)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        shape_conv3 = conv3.shape
        flatten = layers.flatten(conv3, axis=1)
        fc = self.fc(flatten)

        defc = self.decoder[-1](fc)
        x = layers.reshape(defc, shape_conv3)
        for layer in self.decoder[:-1]:
            x = layer(x)
        return x

    def policy(self, obs):
        """
        Args:
            obs: A float32 tensor of shape [B, C, H, W]

        Returns:
            policy_logits: B * ACT_DIM
        """
        obs = obs / 255.0
        conv1 = self.conv1(obs)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)

        flatten = layers.flatten(conv3, axis=1)
        fc_output = self.fc(flatten)

        policy_logits = self.policy_fc(fc_output)
        return policy_logits

    def value(self, obs):
        """
        Args:
            obs: A float32 tensor of shape [B, C, H, W]

        Returns:
            values: B
        """
        obs = obs / 255.0
        conv1 = self.conv1(obs)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)

        flatten = layers.flatten(conv3, axis=1)
        fc_output = self.fc(flatten)

        values = self.value_fc(fc_output)
        values = layers.squeeze(values, axes=[1])
        return values

    def policy_and_value(self, obs):
        """
        Args:
            obs: A float32 tensor of shape [B, C, H, W]

        Returns:
            policy_logits: B * ACT_DIM
            values: B
        """
        obs = obs / 255.0
        conv1 = self.conv1(obs)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)

        flatten = layers.flatten(conv3, axis=1)
        fc_output = self.fc(flatten)

        policy_logits = self.policy_fc(fc_output)

        values = self.value_fc(fc_output)
        values = layers.squeeze(values, axes=[1])

        return policy_logits, values

    def layer_recorder(self, args):
        with open('encoder_args_record', 'a+') as f:
            f.write(json.dumps(args) + '\n')

    def conv2d_helper(self, num_filters, filter_size, stride=1, padding=0, dilation=1, groups=None,
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
        args = {"type": 0,
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

        self.layer_recorder(args)
        return conv2d

    def fc_helepr(self, size, num_flatten_dims=1, param_attr=None, bias_attr=None, act=None, name=None):
        fc = layers.fc(size=size,
                       num_flatten_dims=num_flatten_dims,
                       param_attr=param_attr,
                       bias_attr=bias_attr,
                       act=act,
                       name=name)
        args = {"type": 1,
                "size": size,
                "num_flatten_dims": num_flatten_dims,
                "param_attr": param_attr,
                "bias_attr": bias_attr,
                "act": act,
                "name": name}
        self.layer_recorder(args)
        return fc

    def flatten_helper(self, axis=1, name=None):
        ft = Flatten(axis, name)
        args = {"type": 2,
                "axis": axis,
                "name": name}
        self.layer_recorder(args)
        return ft

    def conv2d_decoder(self, data, out_shape):
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

        h, w = out_shape[2], out_shape[3]
        hh = compute_conv2d_HW(h)
        ww = compute_conv2d_HW(w)
        de_padding = ((hh - 1) * stride + dilation * (filter_size - 1) + 1 - h) // 2

        layer = layers.conv2d_transpose(num_filters=out_shape[1],
                                        name=str('de' + name) if name else None,
                                        padding=de_padding,
                                        filter_size=filter_size,
                                        stride=stride,
                                        dilation=dilation,
                                        param_attr=param_attr,
                                        bias_attr=bias_attr,
                                        use_cudnn=use_cudnn,
                                        act=act)

        return layer, [out_shape[0], num_filters, hh, ww]

    def fc_decoder(self, data, shape):
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

    def flatten_decoder(self, data, shape):
        name = data['name']
        output_shape = shape[:data['axis']] + [reduce(lambda x, y: x * y, shape[data['axis']:])]
        # return Reshape(shape=shape), output_shape
        # print(shape, output_shape)
        return output_shape
        # return Reshape(shape=shape, name=str('reshape' + name) if name else name), output_shape
