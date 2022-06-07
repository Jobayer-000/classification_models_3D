from clf_models_3D import efficientnetv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras import utils as keras_utils


def Conv3dBn(
        filters,
        kernel_size,
        strides=(1, 1, 1),
        padding='valid',
        data_format=None,
        dilation_rate=(1, 1, 1),
        activation=None,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        use_batchnorm=False,
        **kwargs
):
    """Extension of Conv3D layer with batchnorm"""

    conv_name, act_name, bn_name = None, None, None
    block_name = kwargs.pop('name', None)

    if block_name is not None:
        conv_name = block_name + '_conv'

    if block_name is not None and activation is not None:
        act_str = activation.__name__ if callable(activation) else str(activation)
        act_name = block_name + '_' + act_str

    if block_name is not None and use_batchnorm:
        bn_name = block_name + '_bn'

    bn_axis = 4

    def wrapper(input_tensor):

        x = layers.Conv3D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=None,
            use_bias=not (use_batchnorm),
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            name=conv_name,
        )(input_tensor)

        if use_batchnorm:
            x = layers.BatchNormalization(axis=bn_axis, name=bn_name)(x)

        if activation:
            x = layers.Activation(activation, name=act_name)(x)

        return x

    return wrapper
  
  
  
  
  

  
  
  
def Conv3x3BnReLU(filters, use_batchnorm, name=None):
    def wrapper(input_tensor):
        return Conv3dBn(
            filters,
            kernel_size=3,
            activation='relu',
            kernel_initializer='he_uniform',
            padding='same',
            use_batchnorm=use_batchnorm,
            name=name
        )(input_tensor)

    return wrapper


def DecoderUpsamplingX2Block(filters, stage, use_batchnorm=False):
    up_name = 'decoder_stage{}_upsampling'.format(stage)
    conv1_name = 'decoder_stage{}a'.format(stage)
    conv2_name = 'decoder_stage{}b'.format(stage)
    concat_name = 'decoder_stage{}_concat'.format(stage)

    concat_axis = 4

    def wrapper(input_tensor, skip=None):
        x = layers.UpSampling3D(size=2, name=up_name)(input_tensor)

        if skip is not None:
            x = layers.Concatenate(axis=concat_axis, name=concat_name)([x, skip])

        x = Conv3x3BnReLU(filters, use_batchnorm, name=conv1_name)(x)
        x = Conv3x3BnReLU(filters, use_batchnorm, name=conv2_name)(x)

        return x

    return wrapper


def DecoderTransposeX2Block(filters, stage, use_batchnorm=False):
    transp_name = 'decoder_stage{}a_transpose'.format(stage)
    bn_name = 'decoder_stage{}a_bn'.format(stage)
    relu_name = 'decoder_stage{}a_relu'.format(stage)
    conv_block_name = 'decoder_stage{}b'.format(stage)
    concat_name = 'decoder_stage{}_concat'.format(stage)

    concat_axis = bn_axis = 4 if backend.image_data_format() == 'channels_last' else 1

    def layer(input_tensor, skip=None):

        x = layers.Conv3DTranspose(
            filters,
            kernel_size=(4, 4, 4),
            strides=(2, 2, 2),
            padding='same',
            name=transp_name,
            use_bias=not use_batchnorm,
        )(input_tensor)

        if use_batchnorm:
            x = layers.BatchNormalization(axis=bn_axis, name=bn_name)(x)

        x = layers.Activation('relu', name=relu_name)(x)

        if skip is not None:
            x = layers.Concatenate(axis=concat_axis, name=concat_name)([x, skip])

        x = Conv3x3BnReLU(filters, use_batchnorm, name=conv_block_name)(x)

        return x

    return layer


# ---------------------------------------------------------------------
#  Unet Decoder
# ---------------------------------------------------------------------

def build_unet(
        backbone,
        decoder_block,
        skip_connection_layers,
        decoder_filters=(256, 128, 64, 32, 16),
        n_upsample_blocks=5,
        classes=1,
        activation='sigmoid',
        use_batchnorm=True,
        dropout=None,
):
 
    backbone_1 = keras.Model([backbone[0].input], [backbone[0].output, backbone[0].get_layer(name=skip_connection_layers[0])])
    backbone_2 = keras.Model([backbone[1].input], [backbone[1].output, backbone[1].get_layer(name=skip_connection_layers[1])])
    backbone_3 = backbone[2]
    backbone_4 = keras.Model([backbone[3].input], [backbone[3].output, backbone[3].get_layer(name=skip_connection_layers[2])])
    backbone_5 = backbone[4]
    backbone_6 = keras.Model([backbone[5].input], [backbone[5].output, backbone[5].get_layer(name=skip_connection_layers[3])])
    
    skips  =[bacebone_1.output[1], bacebone_2.output[1], bacebone_4.output[1], bacebone_6.output[1], bacebone_6.output[0]]
    x = skips[-1]
    # add center block if previous operation was maxpooling (for vgg models)
    if isinstance(backbone.layers[-1], layers.MaxPooling3D):
        x = Conv3x3BnReLU(512, use_batchnorm, name='center_block1')(x)
        x = Conv3x3BnReLU(512, use_batchnorm, name='center_block2')(x)

    # building decoder blocks
    for i in range(n_upsample_blocks):

        if i < len(skips[:-1]):
            skip = skips[i]
        else:
            skip = None

        x = decoder_block(decoder_filters[i], stage=i, use_batchnorm=use_batchnorm)(x, skip)

    if dropout:
        x = layers.SpatialDropout3D(dropout, name='pyramid_dropout')(x)

    # model head (define number of output classes)
    x = layers.Conv3D(
        filters=classes,
        kernel_size=(3, 3, 3),
        padding='same',
        use_bias=True,
        kernel_initializer='glorot_uniform',
        name='final_conv',
    )(x)
    x = layers.Activation(activation, name=activation)(x)

    # create keras model instance
    model = models.Model(inputs=skips, outputs=x)

    return backbone_1,backbone_2,backbone_3,backbone_4,backbone_5,backbone_6,model


# ---------------------------------------------------------------------
#  Unet Model
# ---------------------------------------------------------------------
get_features_name = {
        'efficientnetb0' : ('block6a_expand_activation', 'block4a_expand_activation',
                           'block3a_expand_activation', 'block2a_expand_activation'),
        'efficientnetb1' : ('block6a_expand_activation', 'block4a_expand_activation',
                           'block3a_expand_activation', 'block2a_expand_activation'),
        'efficientnetb2' : ('block6a_expand_activation', 'block4a_expand_activation',
                           'block3a_expand_activation', 'block2a_expand_activation'),
        'efficientnetb3' : ('block6a_expand_activation', 'block4a_expand_activation',
                           'block3a_expand_activation', 'block2a_expand_activation'),
        'efficientnetv2-b0': ('block6a_expand_activation', 'block4a_expand_activation',
                             'block2b_add', 'block1a_project_activation'),
        'efficientnetv2-b1': ('block6a_expand_activation', 'block4a_expand_activation',
                             'block2b_add', 'block1a_project_activation'),
        'efficientnetv2-b2': ('block6a_expand_activation', 'block4a_expand_activation',
                             'block2b_add', 'block1a_project_activation'),
        'efficientnetv2-b3': ('block6a_expand_activation', 'block4a_expand_activation',
                             'block2b_add', 'block1a_project_activation'),
        'efficientnetv2-s': ('block6a_expand_activation', 'block4a_expand_activation',
                            'block2b_add', 'block1a_project_activation'),
        'efficientnetv2-m': ('block6a_expand_activation', 'block4a_expand_activation',
                            'block2b_add', 'block1a_project_activation'),
        'efficientnetv2-l': ('block6a_expand_activation', 'block4a_expand_activation',
                            'block2b_add', 'block1a_project_activation')
}
def Unet(
        backbone_name='efficientnetv2-s',
        input_shape=(None, None, None, 3),
        classes=1,
        activation='sigmoid',
        weights=None,
        encoder_weights=None,
        encoder_freeze=False,
        encoder_features='default',
        decoder_block_type='upsampling',
        decoder_filters=(256, 128, 64, 32, 16),
        decoder_use_batchnorm=True,
        dropout=None,
        **kwargs
):
    """ Unet_ is a fully convolution neural network for image semantic segmentation
    Args:
        backbone_name: name of classification model (without last dense layers) used as feature
            extractor to build segmentation model.
        input_shape: shape of input data/image ``(H, W, C)``, in general
            case you do not need to set ``H`` and ``W`` shapes, just pass ``(None, None, C)`` to make your model be
            able to process images af any size, but ``H`` and ``W`` of input images should be divisible by factor ``32``.
        classes: a number of classes for output (output shape - ``(h, w, classes)``).
        activation: name of one of ``keras.activations`` for last model layer
            (e.g. ``sigmoid``, ``softmax``, ``linear``).
        weights: optional, path to model weights.
        encoder_weights: one of ``None`` (random initialization), ``imagenet`` (pre-training on ImageNet).
        encoder_freeze: if ``True`` set all layers of encoder (backbone model) as non-trainable.
        encoder_features: a list of layer numbers or names starting from top of the model.
            Each of these layers will be concatenated with corresponding decoder block. If ``default`` is used
            layer names are taken from ``DEFAULT_SKIP_CONNECTIONS``.
        decoder_block_type: one of blocks with following layers structure:
            - `upsampling`:  ``UpSampling2D`` -> ``Conv2D`` -> ``Conv2D``
            - `transpose`:   ``Transpose2D`` -> ``Conv2D``
        decoder_filters: list of numbers of ``Conv2D`` layer filters in decoder blocks
        decoder_use_batchnorm: if ``True``, ``BatchNormalisation`` layer between ``Conv2D`` and ``Activation`` layers
            is used.
    Returns:
        ``keras.models.Model``: **Unet**
    .. _Unet:
        https://arxiv.org/pdf/1505.04597
    """

    if decoder_block_type == 'upsampling':
        decoder_block = DecoderUpsamplingX2Block
    elif decoder_block_type == 'transpose':
        decoder_block = DecoderTransposeX2Block
    else:
        raise ValueError('Decoder block type should be in ("upsampling", "transpose"). '
                         'Got: {}'.format(decoder_block_type))
    if backbone_name=='efficientnetb0':
        backbone = efficientnet.EfficientNetB0(input_shape=input_shape, weight=None)
    if backbone_name=='efficientnetb1':
        backbone = efficientnet.EfficientNetB1(input_shape=input_shape)
    if backbone_name=='efficientnetb2':
        backbone = efficientnet.EfficientNetB2(input_shape=input_shape)
    if backbone_name=='efficientnetb3':
        backbone = efficientnet.EfficientNetB3(input_shape=input_shape)
    if backbone_name=='efficientnetv2-b0':
        backbone = efficientnetv2.EfficientNetV2B0(input_shape=input_shape)
    if backbone_name=='efficientnetv2-b1':
        backbone = efficientnetv2.EfficientNetV2B1(input_shape=input_shape)
    if backbone_name=='efficientnetv2-b2':
        backbone = efficientnetv2.EfficientNetV2B2(input_shape=input_shape)
    if backbone_name=='efficientnetv2-b3':
        backbone = efficientnetv2.EfficientNetV2B3(input_shape=input_shape)
    if backbone_name=='efficientnetv2-s':
        backbone = efficientnetv2.EfficientNetV2S(input_shape=input_shape, weight=None)
    if backbone_name=='efficientnetv2-m':
        backbone = efficientnetv2.EfficientNetV2M(input_shape=input_shape)
    if backbone_name=='efficientnetv2-l':
        backbone = efficientnetv2.EfficientNetV2L(input_shape=input_shape)
   
    if encoder_features == 'default':
        encoder_features = get_features_name[backbone_name]

    model = build_unet(
        backbone=backbone,
        decoder_block=decoder_block,
        skip_connection_layers=encoder_features,
        decoder_filters=decoder_filters,
        classes=classes,
        activation=activation,
        n_upsample_blocks=len(decoder_filters),
        use_batchnorm=decoder_use_batchnorm,
        dropout=dropout,
    )

    # lock encoder weights for fine-tuning
    if encoder_freeze:
        freeze_model(backbone, **kwargs)

    # loading model weights
    if weights is not None:
        model.load_weights(weights)

    return model
