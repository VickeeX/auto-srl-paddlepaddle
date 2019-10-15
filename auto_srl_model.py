# -*- coding: utf-8 -*-

"""
    @File name    :    auto_srl_model.py
    @Date         :    2019-10-15 12:28
    @Description  :    {TODO}
    @Author       :    VickeeX
"""

from parl import Model
from parl.core.fluid import layers
from layers_generators import conv2d_generator, fc_generator


class AutoSRLModel(Model):

    def __init__(self, act_dim, obs_shape):
        self.conv1, self.deconv1, shape = conv2d_generator(
            obs_shape, num_filters=32, filter_size=8, stride=4, padding=1, act='relu')
        self.conv2, self.deconv2, shape = conv2d_generator(
            shape, num_filters=64, filter_size=4, stride=2, padding=2, act='relu')
        self.conv3, self.deconv3, _ = conv2d_generator(
            shape, num_filters=64, filter_size=3, stride=1, padding=0, act='relu')

        self.fc, self.defc, shape = fc_generator(shape=[-1, 5184], size=512, act='relu')
        self.policy_fc = layers.fc(size=act_dim)
        self.value_fc = layers.fc(size=1)


if __name__ == '__main__':
    model = AutoSRLModel([], [-1, 4, 84, 84])
