# -*- coding: utf-8 -*-

"""
    @File name    :    srl_algorithm.py
    @Date         :    2019-10-10 14:16
    @Description  :    {TODO}
    @Author       :    VickeeX
"""

""" We want to customize loss in the process of learning, which means the loss
    should be dynamic. So we compute some vars(eg, reconstruction of observation)
    needed for loss in the defined **srl loss helper**(eg, ae_loss) instead
    of *learn()*. Other common vars(eg, state from obs) are computed in **learn()**
"""

from parl.core.fluid.algorithm import Algorithm
from parl.core.fluid import layers


class SRLAlgorithm(Algorithm):
    def __init__(self, model, hyperparas=None, vf_loss_coeff=None):
        self.model = model

    def learn(self, obs, actions, advantages, target_values):
        """ Parameters should be customized according to SRL methods
        note: if you use SRL methods, you need to add loss to total_loss.
        for example:
                ```ae_loss = self.ae_loss(obs)
                   total_loss = pi_loss + ... + self.ae_loss(obs)
                   optimizer = fluid.optimizer.Adam(learning_rate)
                   optimizer.minimize(cost)
                ```
        """
        state, shape_conv = self.model.obs_encode(obs)
        return NotImplementedError

    def square_error_helper(self, input, target):
        return layers.reduce_sum(layers.square_error_cost(input, target))

    def ae_loss(self, obs, state, shape_conv):
        """ Autoencoder loss: reconstruct obs and return ae_loss
        """
        obs_rec = self.model.obs_decode(state, shape_conv)
        ae_loss = self.square_error_helper(obs, obs_rec)
        return ae_loss

    def dae_loss(self, obs):
        """ Denoising autoencoder loss: add noise to obs and reconstruct
            the noised_obs, return the reconstruction of noised_obs.
        """
        obs_noise = self.model.obs_noise(obs)
        state_noise, shape = self.model.obs_encode(obs_noise)
        obs_noise_rec = self.model.obs_decode(state_noise, shape)
        dae_loss = self.square_error_helper(obs, obs_noise_rec)
        return dae_loss

    def forward_loss(self, state, act, state_next):
        """ Forward dynamics model: predict next state, compute the loss
            between real state_next and preiction.
        """
        state_next_pred = self.model.state_predict(state, act)
        forward_loss = self.square_error_helper(state_next, state_next_pred)
        return forward_loss

    def inverse_loss(self, state, state_next, act):
        """ Inverse dynamics model: predict action with obs and obs_next
            or state and state_next, compute the loss between action and
            action prediction.
        """
        act_pred = self.model.act_predict(state, state_next)
        inverse_loss = self.square_error_helper(act, act_pred)
        return inverse_loss

    def reward_prediction_loss(self, state, state_next, reward):
        reward_pred = self.model.reward_predict(state, state_next)
        reward_loss = self.square_error_helper(reward, reward_pred)
        return reward_loss

    def sample(self, obs):
        return NotImplementedError

    def predict(self, obs):
        return NotImplementedError

    def value(self, obs):
        return NotImplementedError
