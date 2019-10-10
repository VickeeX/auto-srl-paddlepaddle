## SRL models with PARL

***note: reference the realization of PARL/examples***

#### Available methods::
* Autoencoder (reconstruction loss)  [ok]
* Denoising Autoencoder (DAE)
* Forward Dynamics model
* Inverse Dynamics model
* Reward prediction loss
* Variational Autoencoder (VAE) and beta-VAE


#### Realization
* SRLBaseModel:
    the original model as PARL/examples/A2C/atari_model
* Models inheriting from SRLBaseModel:
    realize the decoder layers, reconstruct obs/states/.etc
* SRLAlgorithm:
    template of algorithm, define losses of different SRL mehods
