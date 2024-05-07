import numpy as np
import os, sys
import nibabel as nib

from .process import norm_image
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import concatenate, Conv2D, Conv3D, UpSampling3D, Activation, BatchNormalization
from tensorflow.keras.layers import SpatialDropout3D, MaxPooling3D, Conv3DTranspose, Dropout, LeakyReLU #old: DECONV3D
from tensorflow.keras.optimizers import Adam
from tensorflow_addons.layers import InstanceNormalization
import tensorflow as tf
import segmentation_models_3D as sm
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K

def crop_image_brats(img, OUT_SHAPE=(192, 224, 160)):
    # manual cropping
    input_shape = np.array(img.shape)
    # center the cropped image
    offset = np.array((input_shape - OUT_SHAPE) / 2).astype(int)
    offset[offset < 0] = 0
    x, y, z = offset
    crop_img = img[x:x + OUT_SHAPE[0], y:y + OUT_SHAPE[1], z:z + OUT_SHAPE[2]]
    # pad the preprocessed image
    padded_img = np.zeros(OUT_SHAPE)
    x, y, z = np.array((OUT_SHAPE - np.array(crop_img.shape)) / 2).astype(int)
    padded_img[x:x + crop_img.shape[0], y:y + crop_img.shape[1], z:z + crop_img.shape[2]] = crop_img
    return padded_img

def postprocess_tumor(seg_data, OUT_SHAPE = (240, 240, 155), POST_ENHANC=False, THRES=200):
    # post-process the enhancing tumor region
    if POST_ENHANC:
        seg_enhancing = (seg_data == 4)
        if np.sum(seg_enhancing) < THRES:
            if np.sum(seg_enhancing) > 0:
                seg_data[seg_enhancing] = 1
                print("\tConverted {} voxels from label 4 to label 1!".format(np.sum(seg_enhancing)))

    input_shape = np.array(seg_data.shape)
    OUT_SHAPE = np.array(OUT_SHAPE)
    offset = np.array((OUT_SHAPE - input_shape)/2).astype(int)
    offset[offset<0] = 0
    x, y, z = offset

    # pad the preprocessed image
    padded_seg = np.zeros(OUT_SHAPE).astype(np.uint8)
    padded_seg[x:x+seg_data.shape[0],y:y+seg_data.shape[1],z:z+seg_data.shape[2]] = seg_data[:,:,2:padded_seg.shape[2]+2]

    return padded_seg #.astype(np.uint8)

def load_images(model, ID, PATH_DATA='./', DIM=(192, 224, 160), VALID_SET=True, POST_ENHANC=False):
    img1 = os.path.join(PATH_DATA, ID, ID+'_flair.nii')
    img2 = os.path.join(PATH_DATA, ID, ID+'_t1.nii')
    img3 = os.path.join(PATH_DATA, ID, ID+'_t1ce.nii')
    img4 = os.path.join(PATH_DATA, ID, ID+'_t2.nii')

    # combine the four imaging modalities (flair, t1, t1ce, t2)
    imgs_input = nib.concat_images([img1, img2, img3, img4]).get_fdata()

    imgs_preprocess = np.zeros((DIM[0],DIM[1],DIM[2],4)) # (5, 192, 224, 160)
    if VALID_SET:
        for i in range(imgs_preprocess.shape[-1]):
            imgs_preprocess[:, :, :, i] = crop_image_brats(imgs_input[:, :, :, i])
            imgs_preprocess[:, :, :, i] = norm_image(imgs_preprocess[:, :, :, i])

    return imgs_preprocess[np.newaxis, ...]

def create_convolution_block(input_layer, n_filters, BN=True, KERNEL=(3, 3, 3), ACTIV=None,
                             PAD='same', STR=(1, 1, 1), IN=False):
    layer = Conv3D(n_filters, KERNEL, padding=PAD, strides=STR)(input_layer)
    if BN:
        layer = BatchNormalization(axis=-1)(layer)
    elif IN:
        layer = InstanceNormalization(axis=-1)(layer)
    if ACTIV is None:
        return Activation('relu')(layer)
    else:
        return ACTIV()(layer)

def get_up_convolution(n_filters, pool_size, KERNEL=(3, 3, 3), STR=(2, 2, 2),
                       DECONV=False):
    if DECONV:
        return Conv3DTranspose(filters=n_filters, kernal=KERNEL,
                               strides=STR)
    else:
        return UpSampling3D(size=pool_size)

def create_localization_module(input_layer, n_filters):
    convolution1 = create_convolution_block(input_layer, n_filters)
    convolution2 = create_convolution_block(convolution1, n_filters)
    return convolution2

def create_context_module(input_layer, n_level_filters, DROPOUT=0.3):
    convolution1 = create_convolution_block(input_layer=input_layer, n_filters=n_level_filters)
    dropout = SpatialDropout3D(rate=DROPOUT)(convolution1)
    convolution2 = create_convolution_block(input_layer=dropout, n_filters=n_level_filters)
    return convolution2

def create_up_sampling_module(input_layer, n_filters, SIZE=(2, 2, 2)):
    convolution1 = create_convolution_block(input_layer, n_filters, KERNEL=(2, 2, 2))
    up_sample = UpSampling3D(size=SIZE)(convolution1)
    return up_sample


def get_deepseg(INP_SHAPE=(192, 224, 160, 4), N_FILTERS=8, DEPTH=5, DROPOUT=0.5,
                      N_SEG=3, N_LABELS=4, OPT=Adam, INIT_LR=1e-4,
                      LOSS="mse", ACTIV="softmax", WEIGHTS=None):
    inputs = Input(INP_SHAPE)

    current_layer = inputs
    level_output_layers = list()
    level_filters = list()
    for level_number in range(DEPTH):
        n_level_filters = (2**level_number) * N_FILTERS
        level_filters.append(n_level_filters)

        in_conv = create_convolution_block(current_layer, n_level_filters)
        context_output_layer = create_convolution_block(in_conv, n_level_filters)

        level_output_layers.append(current_layer)
        current_layer = MaxPooling3D(pool_size=(2, 2, 2))(context_output_layer)

    current_layer = SpatialDropout3D(rate=DROPOUT)(current_layer)

    for level_number in reversed(range(DEPTH)):
        up_sampling = create_up_sampling_module(current_layer, level_filters[level_number])
        concatenation_layer = concatenate([level_output_layers[level_number], up_sampling], axis=-1)
        
        localization_output = create_localization_module(concatenation_layer, level_filters[level_number])
        current_layer = localization_output

    output_layer = Conv3D(N_LABELS, (1, 1, 1), name="output_layer")(current_layer)
    activ_block = Activation(ACTIV, name="output_layer_soft")(output_layer)
    model = Model(inputs=inputs, outputs=activ_block)
    
    model.compile(optimizer=Adam(lr=INIT_LR), loss=LOSS, metrics=["accuracy"])

    if WEIGHTS:
        model.load_weights(WEIGHTS)
    return model

def get_last_layer(model):
    last_layer = model.layers[-1]
    return last_layer

def get_last_conv_layer(model, DIM="3d"):
    if DIM == "2d":
        final_conv = list(filter(lambda x: isinstance(x, Conv2D), 
                                   model.layers))[-1]
    elif DIM == "3d":
        final_conv = list(filter(lambda x: isinstance(x, Conv3D), 
                                   model.layers))[-1]
    return final_conv

def get_xai_segmentation_model(s_model, layer_n = None): 
    if layer_n == None:
        xai_layer = get_last_conv_layer(s_model)
    else:
        xai_layer = s_model.get_layer(layer_n)
   
    model = tf.keras.models.Model([s_model.inputs], [xai_layer.output, s_model.output])

    return model

# def dice_coef(y_true, y_pred):
#     y_true = tf.keras.layers.Flatten()(y_true)
#     y_pred = tf.keras.layers.Flatten()(y_pred)
#     intersection = tf.reduce_sum(y_true * y_pred)
#     return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)  


# wt0 = 0.0
# wt1 = 0.7236091298145506
# wt2 = 0.31134094151212555
# wt3 = 1.0
# kernel_initializer =  'he_uniform'
# metrics = ['accuracy', sm.metrics.IOUScore(threshold=0.5),dice_coef]
# smooth = 1e-15
# dice_loss = sm.losses.DiceLoss(class_weights=np.array([wt0, wt1, wt2, wt3])) 
# focal_loss = sm.losses.CategoricalFocalLoss()
# total_loss = dice_loss + (1 * focal_loss)

# optim = Adam(0.0001)

 

# def simple_unet_model(IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH, IMG_CHANNELS, num_classes, WEIGHTS=None):

#     inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH, IMG_CHANNELS))

#     s = inputs
#     #Contraction path
#     c1 = Conv3D(16, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same', name='1')(s)
#     c1 = Dropout(0.1)(c1)
#     c1 = Conv3D(16, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same', name='2')(c1)
#     p1 = MaxPooling3D((2, 2, 2))(c1)
    
#     c2 = Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same', name='3')(p1)
#     c2 = Dropout(0.1)(c2)
#     c2 = Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same', name='4')(c2)
#     p2 = MaxPooling3D((2, 2, 2))(c2)
     
#     c3 = Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same', name='5')(p2)
#     c3 = Dropout(0.2)(c3)
#     c3 = Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same', name='6')(c3)
#     p3 = MaxPooling3D((2, 2, 2))(c3)

#     c4 = Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same', name='7')(p3)
#     c4 = Dropout(0.2)(c4)
#     c4 = Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same', name='8')(c4)
#     p4 = MaxPooling3D(pool_size=(2, 2, 2))(c4)

#     c5 = Conv3D(256, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same', name='9')(p4)
#     c5 = Dropout(0.3)(c5)
#     c5 = Conv3D(256, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same', name='10')(c5)

#     #Expansive path 
#     u6 = Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), padding='same')(c5)
#     u6 = concatenate([u6, c4])
#     c6 = Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same', name='a')(u6)
#     c6 = Dropout(0.2)(c6)
#     c6 = Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same', name='b')(c6)
     
#     u7 = Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), padding='same')(c6)
#     u7 = concatenate([u7, c3])
#     c7 = Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same', name='c')(u7)
#     c7 = Dropout(0.2)(c7)
#     c7 = Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same', name='d')(c7)

#     u8 = Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), padding='same')(c7)
#     u8 = concatenate([u8, c2])
#     c8 = Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same', name='e')(u8)
#     c8 = Dropout(0.1)(c8)
#     c8 = Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same', name='f')(c8)

#     u9 = Conv3DTranspose(16, (2, 2, 2), strides=(2, 2, 2), padding='same')(c8)
#     u9 = concatenate([u9, c1])
#     c9 = Conv3D(16, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same', name='g')(u9)
#     c9 = Dropout(0.1)(c9)
#     c9 = Conv3D(16, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same', name='h')(c9)

#     outputs = Conv3D(num_classes, (1, 1, 1), activation='softmax')(c9)

#     model = Model(inputs=[inputs], outputs=[outputs])
#     #compile model outside of this function to make it flexible. 
#     model.compile(optimizer = optim, loss=total_loss, metrics=metrics)

#     if WEIGHTS:
#         model.load_weights(WEIGHTS)
    
#     return model


def total_loss(y_true, y_pred):
    # Calculate Dice Loss
    dice_loss = sm.losses.DiceLoss(class_weights=np.array([wt0, wt1, wt2, wt3]))(y_true, y_pred)
    # Calculate Focal Loss
    focal_loss = sm.losses.CategoricalFocalLoss()(y_true, y_pred)
    # Combine the losses with weights alpha and beta
    alpha = 0.5
    beta = 0.5
    total_loss = alpha * dice_loss + beta * focal_loss
    return total_loss

def weighted_categorical_crossentropy(weights):
  """
  This function defines the weighted categorical cross-entropy loss.

  Args:
      weights: A list or tensor containing the weights for each class.

  Returns:
      A custom loss function object.
  """

  def loss(y_true, y_pred):
    """
    This function calculates the weighted categorical cross-entropy loss.

    Args:
        y_true: One-hot encoded ground truth labels with shape (batch_size, 128, 128, 128, 4).
        y_pred: Model predictions with shape (batch_size, 128, 128, 128, 4).

    Returns:
        A tensor representing the weighted categorical cross-entropy loss.
    """

    # Clip predictions to avoid overflow in log function
    epsilon = K.epsilon()  # Small value to avoid division by zero
    y_pred = K.clip(y_pred, epsilon, 1 - epsilon)

    # Calculate categorical cross-entropy loss
    categorical_crossentropy = -K.mean(K.sum(y_true * K.log(y_pred), axis=-1))

    # Apply weights and return the loss
    return K.mean(categorical_crossentropy * K.cast(weights, K.floatx()))

  return loss


def dice_coef(y_true, y_pred):
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)   

smooth = 1e-15
LR = 0.0001

wt0 = 0.0
wt1 = 0.8024553571428571
wt2 = 0.32328869047619047
wt3 = 1.0

wt00 = 0.0
wt11 = 1.0
wt22 = 0.3814821798981708
wt33 = 0.8331133320761833

kernel_initializer =  'he_uniform'
seg_metrics = ['accuracy', sm.metrics.IOUScore(threshold=0.5), dice_coef]
edge_metrics = ['accuracy']


optim = Adam(LR)

losses = {
    "Seg_Output": total_loss,
    # "Edge_Output": weighted_binary_crossentropy
    "Edge_Output": weighted_categorical_crossentropy([wt00, wt11, wt22, wt33])
}

metrics = {
    "Seg_Output": seg_metrics,
    "Edge_Output": edge_metrics
}

def two_decoder_unet_model(IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH, IMG_CHANNELS, num_classes, WEIGHTS=None):

    input_shape = (IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH, IMG_CHANNELS)

    inputs = Input(input_shape, name='shared_encoder_input')
    # s = Lambda(lambda x: x / 255)(inputs)   #No need for this if we normalize our inputs beforehand
    s = inputs
    
    #Encoder path
    c1 = Conv3D(16, (3, 3, 3), activation=LeakyReLU(alpha=0.1), kernel_initializer=kernel_initializer, padding='same', name='1')(s)
    c1 = Dropout(0.1)(c1)
    c1 = Conv3D(16, (3, 3, 3), activation=LeakyReLU(alpha=0.1), kernel_initializer=kernel_initializer, padding='same', name='2')(c1)
    p1 = MaxPooling3D((2, 2, 2))(c1)
    
    c2 = Conv3D(32, (3, 3, 3), activation=LeakyReLU(alpha=0.1), kernel_initializer=kernel_initializer, padding='same', name='3')(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv3D(32, (3, 3, 3), activation=LeakyReLU(alpha=0.1), kernel_initializer=kernel_initializer, padding='same', name='4')(c2)
    p2 = MaxPooling3D((2, 2, 2))(c2)
     
    c3 = Conv3D(64, (3, 3, 3), activation=LeakyReLU(alpha=0.1), kernel_initializer=kernel_initializer, padding='same', name='5')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv3D(64, (3, 3, 3), activation=LeakyReLU(alpha=0.1), kernel_initializer=kernel_initializer, padding='same', name='6')(c3)
    p3 = MaxPooling3D((2, 2, 2))(c3)

    c4 = Conv3D(128, (3, 3, 3), activation=LeakyReLU(alpha=0.1), kernel_initializer=kernel_initializer, padding='same', name='7')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv3D(128, (3, 3, 3), activation=LeakyReLU(alpha=0.1), kernel_initializer=kernel_initializer, padding='same', name='8')(c4)
    p4 = MaxPooling3D(pool_size=(2, 2, 2))(c4)

    c5 = Conv3D(256, (3, 3, 3), activation=LeakyReLU(alpha=0.1), kernel_initializer=kernel_initializer, padding='same', name='9')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv3D(256, (3, 3, 3), activation=LeakyReLU(alpha=0.1), kernel_initializer=kernel_initializer, padding='same', name='10')(c5)


    #S Decoder
    u6_s = Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), padding='same')(c5)
    u6_s = concatenate([u6_s, c4])
    c6_s = Conv3D(128, (3, 3, 3), activation=LeakyReLU(alpha=0.1), kernel_initializer=kernel_initializer, padding='same', name='1d')(u6_s)
    c6_s = Dropout(0.2)(c6_s)
    c6_s = Conv3D(128, (3, 3, 3), activation=LeakyReLU(alpha=0.1), kernel_initializer=kernel_initializer, padding='same', name='2d')(c6_s)
     
    u7_s = Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), padding='same')(c6_s)
    u7_s = concatenate([u7_s, c3])
    c7_s = Conv3D(64, (3, 3, 3), activation=LeakyReLU(alpha=0.1), kernel_initializer=kernel_initializer, padding='same', name='3d')(u7_s)
    c7_s = Dropout(0.2)(c7_s)
    c7_s = Conv3D(64, (3, 3, 3), activation=LeakyReLU(alpha=0.1), kernel_initializer=kernel_initializer, padding='same', name='4d')(c7_s)

    u8_s = Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), padding='same')(c7_s)
    u8_s = concatenate([u8_s, c2])
    c8_s = Conv3D(32, (3, 3, 3), activation=LeakyReLU(alpha=0.1), kernel_initializer=kernel_initializer, padding='same', name='5d')(u8_s)
    c8_s = Dropout(0.1)(c8_s)
    c8_s = Conv3D(32, (3, 3, 3), activation=LeakyReLU(alpha=0.1), kernel_initializer=kernel_initializer, padding='same', name='6d')(c8_s)

    u9_s = Conv3DTranspose(16, (2, 2, 2), strides=(2, 2, 2), padding='same')(c8_s)
    u9_s = concatenate([u9_s, c1])
    c9_s = Conv3D(16, (3, 3, 3), activation=LeakyReLU(alpha=0.1), kernel_initializer=kernel_initializer, padding='same', name='7d')(u9_s)
    c9_s = Dropout(0.1)(c9_s)
    c9_s = Conv3D(16, (3, 3, 3), activation=LeakyReLU(alpha=0.1), kernel_initializer=kernel_initializer, padding='same', name='8d')(c9_s)

    output_S = Conv3D(num_classes, (1, 1, 1), activation='softmax', name="Seg_Output")(c9_s)

    #P Decoder
    u6_p = Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), padding='same')(c5)
    u6_p = concatenate([u6_p, c4])
    c6_p = Conv3D(128, (3, 3, 3), activation=LeakyReLU(alpha=0.1), kernel_initializer=kernel_initializer, padding='same')(u6_p)
    c6_p = Dropout(0.2)(c6_p)
    c6_p = Conv3D(128, (3, 3, 3), activation=LeakyReLU(alpha=0.1), kernel_initializer=kernel_initializer, padding='same')(c6_p)

    u7_p = Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), padding='same')(c6_p)
    u7_p = concatenate([u7_p, c3])
    c7_p = Conv3D(64, (3, 3, 3), activation=LeakyReLU(alpha=0.1), kernel_initializer=kernel_initializer, padding='same')(u7_p)
    c7_p = Dropout(0.2)(c7_p)
    c7_p = Conv3D(64, (3, 3, 3), activation=LeakyReLU(alpha=0.1), kernel_initializer=kernel_initializer, padding='same')(c7_p)

    u8_p = Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), padding='same')(c7_p)
    u8_p = concatenate([u8_p, c2])
    c8_p = Conv3D(32, (3, 3, 3), activation=LeakyReLU(alpha=0.1), kernel_initializer=kernel_initializer, padding='same')(u8_p)
    c8_p = Dropout(0.1)(c8_p)
    c8_p = Conv3D(32, (3, 3, 3), activation=LeakyReLU(alpha=0.1), kernel_initializer=kernel_initializer, padding='same')(c8_p)

    u9_p = Conv3DTranspose(16, (2, 2, 2), strides=(2, 2, 2), padding='same')(c8_p)
    u9_p = concatenate([u9_p, c1])
    c9_p = Conv3D(16, (3, 3, 3), activation=LeakyReLU(alpha=0.1), kernel_initializer=kernel_initializer, padding='same')(u9_p)
    c9_p = Dropout(0.1)(c9_p)
    c9_p = Conv3D(16, (3, 3, 3), activation=LeakyReLU(alpha=0.1), kernel_initializer=kernel_initializer, padding='same')(c9_p)

    output_P = Conv3D(4, (1, 1, 1), activation='softmax', name="Edge_Output")(c9_p)

    outputs = [output_S, output_P]

    model = Model(inputs=[inputs], outputs=outputs)

    model.compile(optimizer=optim, loss=losses, metrics=metrics)

    if WEIGHTS:
        model.load_weights(WEIGHTS)

    return model