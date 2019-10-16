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
        super(DecoderGeneratorModelExample, self).__init__()
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
        self.decoder = self.decoder_generator(obs_shape)

        # to generate decoder according to encoder

    def decoder_generator(self, shape):
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
                print(shape)
                decoder.append(layer)
        return decoder

    def obs_encode(self, obs):
        for layer in self.encoder:
            out = layer(out)


if __name__ == '__main__':
    model = DecoderGeneratorModelExample([], [-1, 4, 84, 84])
