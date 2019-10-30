## An auto SRL module with paddlepaddle

***Contents***

    —— `decoder_generator`
      |—— `not_use_decoder_helper` package as utils with very slow speed, not suggested
      |—— `decoder_generator_model_template.py` 
      |—— `decoder_generator_model_example.py`
    —— `layers_generator`
      |—— `layers_generators.py` layers generator utils
      |—— `layers_generators_model_example.py`
    —— `parl_srl`
      |—— `srl_algorithm.py` define different loos in algorithm
      |—— `srl_model.py` AEModel example
    —— `README.md`
    
    
***Different SRL methods***
    You can realize different SRL models as choosing different SRL methods to use
    losses are defined, u just need to choose when using.

***Two ways to use SRL***

* way 1: layers_generators 
    Use layers_generator when defining network.
    Different has corresponding generator, which returns normal layer and its de-layer.
    eg:
        ```
        self.conv1, self.deconv1, shape = conv2d_generator(
            obs_shape, num_filters=32, filter_size=8, stride=4, padding=1, act='relu')
        self.conv2, self.deconv2, shape = conv2d_generator(
            shape, num_filters=64, filter_size=4, stride=2, padding=2, act='relu')
        ```
    Initial shape is needed, then pass *shape* in each call of generator.
    
 * way 2: decoder_generator
    Some users may want seperate the definition of encoder and decoder. So we use extra
    packaged fluid.layers to define encoder, and record the config into files(structs).
    Then use decoder_generator when you want to produce it. We first record initial
    layers of encoder which is clear, but layers' info for decoders after computation 
    from encoder is recorded in current version.
    Realization methods are integrated in `decoder_generator/DecoderGeneratorModel`, you
    can find example from `decoder_generator/DecoderGeneratorModelExample` .
