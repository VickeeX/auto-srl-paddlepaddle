# -*- coding: utf-8 -*-

"""
    @File name    :    srl_model.py
    @Date         :    2019-10-09 21:23
    @Description  :    {TODO}
    @Author       :    VickeeX
"""

from models.srl_base_model import SRLBaseModel
from parl import layers


class AEModel(SRLBaseModel):
    def __init__(self, act_dim):
        super(AEModel, self).__init__(act_dim)

        # decode layers according to layers in SRLBaseModel.__init__(*)
        self.defc = layers.fc(size=5184, act='relu')
        self.deconv3 = layers.conv2d_transpose(
            num_filters=64, filter_size=3, stride=1, padding=0, act='relu')
        self.deconv2 = layers.conv2d_transpose(
            num_filters=32, filter_size=4, stride=2, padding=2, act='relu')
        self.deconv1 = layers.conv2d_transpose(
            num_filters=4, filter_size=8, stride=4, padding=0, act='relu')

    def obs_reconstrut(self, obs):
        obs = obs / 255.0
        conv1 = self.conv1(obs)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        shape_conv3 = conv3.shape

        flatten = layers.flatten(conv3, axis=1)
        fc = self.fc(flatten)

        defc = self.defc(fc)
        deflatten = layers.reshape(defc, shape_conv3)
        deconv3 = self.deconv3(deflatten)
        deconv2 = self.deconv2(deconv3)
        deconv1 = self.deconv1(deconv2)

        return deconv1 * 255.0
