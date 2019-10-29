# -*- coding: utf-8 -*-

"""
    @File name    :    decoder_generator_model_example.py
    @Date         :    2019-10-14 21:20
    @Description  :    {TODO}
    @Author       :    VickeeX
"""
from parl import layers
from decoder_generator.decoder_generator_model_template import DecoderGeneratorModel


class DecoderGeneratorModelExample(DecoderGeneratorModel):
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

        # self.encoder = [self.conv1, self.conv2, self.conv3, self.flat, self.fc]
        self.decoder = self.decoder_generator()

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


if __name__ == '__main__':
    model = DecoderGeneratorModelExample([], [-1, 4, 84, 84])
