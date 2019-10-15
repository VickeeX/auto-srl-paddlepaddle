# -*- coding: utf-8 -*-

"""
    @File name    :    auto_generator_model_example.py
    @Date         :    2019-10-15 12:28
    @Description  :    an example to use layers_generators
    @Author       :    VickeeX
"""

from parl import Model
from parl.core.fluid import layers
from layers_generators import conv2d_generator, fc_generator

""" example to reconstruct obs in algorithm:
    ''' state, shape = self.model.obs_encode(obs)
        obs_ = self.model.obs_decode(state, shape)
    '''
"""


class AutoGeneratorModelExample(Model):

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

    def obs_encode(self, obs):
        obs = obs / 255.0
        conv1 = self.conv1(obs)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        shape_conv = conv3.shape
        flatten = layers.flatten(conv3, axis=1)
        state = self.fc(flatten)
        return state, shape_conv

    def obs_decode(self, state, shape_conv):
        defc = self.defc(state)
        deflatten = layers.reshape(defc, shape_conv)
        deconv3 = self.deconv3(deflatten)
        deconv2 = self.deconv2(deconv3)
        deconv1 = self.deconv1(deconv2)
        return deconv1 * 255.0

# if __name__ == '__main__':
#     model = AutoGeneratorModelExample([], [-1, 4, 84, 84])
