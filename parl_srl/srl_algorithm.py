# -*- coding: utf-8 -*-

"""
    @File name    :    srl_algorithm.py
    @Date         :    2019-10-10 14:16
    @Description  :    {TODO}
    @Author       :    VickeeX
"""

from parl.core.fluid.algorithm import Algorithm
from parl.core.fluid import layers


class SRLAlgorithm(Algorithm):
    def __init__(self, model, hyperparas=None, vf_loss_coeff=None):
        self.model = model

    def learn(self, obs, actions, advantages, target_values):
        """
        note: if you use SRL methods, you need to add loss to total_loss.
        for example:
                ```ae_loss = self.ae_loss(obs)
                   total_loss = pi_loss + ... + self.ae_loss(obs)
                   optimizer = fluid.optimizer.Adam(learning_rate)
                   optimizer.minimize(cost)
                ```
        """
        return NotImplementedError

    def ae_loss(self, obs):
        """ auto encoder loss: obs_reconstruct and obs
        """
        obs_rec = self.model.obs_reconstruct(obs)
        ae_loss = layers.reduce_sum(layers.square_error_cost(obs_rec, obs))
        return ae_loss

    def sample(self, obs):
        return NotImplementedError

    def predict(self, obs):
        return NotImplementedError

    def value(self, obs):
        return NotImplementedError
