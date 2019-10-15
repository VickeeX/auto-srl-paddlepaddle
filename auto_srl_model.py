# -*- coding: utf-8 -*-

"""
    @File name    :    auto_srl_model.py
    @Date         :    2019-10-15 12:28
    @Description  :    {TODO}
    @Author       :    VickeeX
"""

from parl import Model
from parl.core.fluid import layers


class AutoSRLBaseModel(Model):

    def __init__(self, act_dim, obs_shape):
        self.conv1, self.deconv1, shape = self.conv2d_generator(
            obs_shape, num_filters=32, filter_size=8, stride=4, padding=1, act='relu')
        self.conv2, self.deconv2, shape = self.conv2d_generator(
            shape, num_filters=64, filter_size=4, stride=2, padding=2, act='relu')
        self.conv3, self.deconv3, _ = self.conv2d_generator(
            shape, num_filters=64, filter_size=3, stride=1, padding=0, act='relu')

        self.fc = layers.fc(size=512, act='relu')
        self.policy_fc = layers.fc(size=act_dim)
        self.value_fc = layers.fc(size=1)

    def conv2d_generator(self, shape, num_filters, filter_size, stride=1, padding=0, dilation=1, groups=None,
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

        # print(shape, output_shape)
        # print([num_filters, filter_size, stride, padding], [shape[1], filter_size, stride, de_padding])
        return enc, dec, output_shape


if __name__ == '__main__':
    model = AutoSRLBaseModel([], [-1, 4, 84, 84])
