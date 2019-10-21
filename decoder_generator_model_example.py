# -*- coding: utf-8 -*-

"""
    @File name    :    decoder_generator_model_example.py
    @Date         :    2019-10-14 21:20
    @Description  :    {TODO}
    @Author       :    VickeeX
"""
import json
from parl import Model
from parl.core.fluid import layers
from layers_recorder_helper import *
from layers_decoder_generators import *


class DecoderGeneratorModelExample(Model):
    def __init__(self, act_dim, obs_shape):
        self.conv1 = conv2d_helper(
            num_filters=32, filter_size=8, stride=4, padding=1, act='relu')
        self.conv2 = conv2d_helper(
            num_filters=64, filter_size=4, stride=2, padding=2, act='relu')
        self.conv3 = conv2d_helper(
            num_filters=64, filter_size=3, stride=1, padding=0, act='relu')
        self.flatten = flatten_helper(axis=1)
        self.fc = fc_helepr(size=512, act='relu')

        self.policy_fc = layers.fc(size=act_dim)
        self.value_fc = layers.fc(size=1)

        self.encoder = [self.conv1, self.conv2, self.conv3, self.flatten, self.fc]
        # to generate decoder according to layers_helper records
        self.decoder = self.decoder_generator(obs_shape)

    def decoder_generator(self, shape):
        # TODO: modify generator and obs_encode_decode
        decoder = []
        switch = {
            'conv2d': lambda x, s: conv2d_decoder(x, s),
            'fc': lambda x, s: fc_decoder(x, s),
            'flatten': lambda x, s: flatten_decoder(x, s)
        }

        with open('encoder_args_record', 'r') as f:
            for line in f:
                data = json.loads(line.strip('\n'))
                layer, shape = switch[data['type']](data, shape)
                decoder.append(layer)
        return decoder[::-1]

    def obs_encode_decode(self, code_tag, obs):
        coder = self.encoder if code_tag else self.decoder
        out = obs / 255.0
        for layer in coder:
            try:
                nm = layer.name
                if nm.startswith("flatten"):
                    out = layers.flatten(out, axis=layer.axis, name=layer.name)
                #  omit ```nm.startswith("reshape")``` , only "flatten" and "reshape"
                else:
                    out = layers.reshape(out, shape=layer.shape, name=layer.name)
            except AttributeError:
                out = layer(out)
        return out


if __name__ == '__main__':
    model = DecoderGeneratorModelExample([], [-1, 4, 84, 84])
