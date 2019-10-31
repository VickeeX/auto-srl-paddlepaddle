## SRL models with PARL

***note: reference the realization of PARL/examples***

#### Available methods::
* Autoencoder (reconstruction loss)  [ok]
* Denoising Autoencoder (DAE)  [ok]
* Forward Dynamics model  [ok]
* Inverse Dynamics model  [ok]
* Reward prediction loss  [ok]
* Variational Autoencoder (VAE) and beta-VAE


#### Realization
* SRLModel:
    the original model as PARL/examples/A2C/atari_model.
    realize the decoder layers, define different computation(eg, encode/decode).
* SRLAlgorithm:
    template of algorithm, define losses of different SRL mehods
