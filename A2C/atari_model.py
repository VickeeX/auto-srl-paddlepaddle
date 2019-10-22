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

import multiprocessing


# from flatten_reshape_wrapper import Flatten, Reshape


class Flatten(object):
    def __init__(self, axis=1, name=None):
        self.axis = axis
        self.name = name


class Reshape(object):
    def __init__(self, shape, actual_shape=None, act=None, inplace=False, name=None):
        self.shape = shape
        self.actual_shape = actual_shape
        self.act = act
        self.inplace = inplace
        self.name = name


mgr = multiprocessing.Manager()
SHAPE = mgr.list([-1, 4, 84, 84])


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
        self.decoder = self.decoder_generator()

    def decoder_generator(self):
        switch = {
            0: lambda x: self.conv2d_decoder(x),
            'conv2d': lambda x: self.conv2d_decoder(x),
            1: lambda x: self.fc_decoder(x),
            'fc': lambda x: self.fc_decoder(x),
            2: lambda x: self.flatten_decoder(x),
            'flatten': lambda x: self.flatten_decoder(x)
        }

        decoder, datas = [], []
        with open('encoder_args_record', 'r') as f:
            for line in f:
                datas.append(json.loads(line.strip('\n')))

        for data in datas[:3] + datas[-1:]:
            # for data in datas:
            layer = switch[data['type']](data)
            decoder.append(layer)
            print(data)
            print(layer)

        return decoder[::-1]

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

        defc = self.decoder[0](fc)
        x = layers.reshape(defc, shape_conv3)
        for layer in self.decoder[1:]:
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
        # build conv2d, compute conv2d_transpose args and record
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

        def compute_conv2d_HW(HW):
            # TODO: while padding is an list: padding[0]!=paddings[1]
            # TODO: while groups!=1
            HW = (HW + 2 * padding - (dilation * (filter_size - 1) + 1)) // stride + 1
            return HW

        new_num_filters, h, w = SHAPE[1], SHAPE[2], SHAPE[3]
        hh = compute_conv2d_HW(h)
        ww = compute_conv2d_HW(w)
        SHAPE[1] = num_filters
        SHAPE[2] = hh
        SHAPE[3] = ww
        de_padding = ((hh - 1) * stride + dilation * (filter_size - 1) + 1 - h) // 2

        args = {"type": 0,
                "num_filters": new_num_filters,
                "name": str('de' + name) if name else None,
                "padding": de_padding,
                "filter_size": filter_size,
                "stride": stride,
                "dilation": dilation,
                "param_attr": param_attr,
                "bias_attr": bias_attr,
                "use_cudnn": use_cudnn,
                "act": act
                }

        self.layer_recorder(args)
        return conv2d

    def fc_helepr(self, size, num_flatten_dims=1, param_attr=None, bias_attr=None, act=None, name=None):
        fc = layers.fc(size=size,
                       num_flatten_dims=num_flatten_dims,
                       param_attr=param_attr,
                       bias_attr=bias_attr,
                       act=act,
                       name=str('de' + name) if name else None)

        # TODO: we usually flatten before fc, so reduce is not essential.
        desize = reduce(lambda x, y: x * y, SHAPE[num_flatten_dims:num_flatten_dims + 1])
        SHAPE[num_flatten_dims] = size

        args = {"type": 1,
                "size": desize,
                "num_flatten_dims": num_flatten_dims,
                "param_attr": param_attr,
                "bias_attr": bias_attr,
                "act": act,
                "name": str('de' + name) if name else None}
        self.layer_recorder(args)
        return fc

    def flatten_helper(self, axis=1, name=None):
        ft = Flatten(axis, name)
        SHAPE[axis] = reduce(lambda x, y: x * y, SHAPE[axis:])
        args = {"type": 2,
                "name": str('de' + name) if name else None,
                "shape": SHAPE[:axis + 1]}
        self.layer_recorder(args)
        return ft

    def conv2d_decoder(self, data):
        # TODO: arg: output_size, not involved now
        layer = layers.conv2d_transpose(num_filters=data['num_filters'],
                                        name=data['name'],
                                        padding=data['padding'],
                                        filter_size=data['filter_size'],
                                        stride=data['stride'],
                                        dilation=data['dilation'],
                                        param_attr=data['param_attr'],
                                        bias_attr=data['bias_attr'],
                                        use_cudnn=data['use_cudnn'],
                                        act=data['act'])
        return layer

    def fc_decoder(self, data):
        layer = layers.fc(size=data['size'],
                          name=data['name'],
                          num_flatten_dims=data['num_flatten_dims'],
                          param_attr=data['param_attr'],
                          bias_attr=data['bias_attr'],
                          act=data['act'])
        return layer

    def flatten_decoder(self, data):
        return Reshape(shape=data['name'], name=data['shape'])
