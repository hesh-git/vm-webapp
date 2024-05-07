import numpy as np
from keras.models import load_model

import segmentation_models_3D as sm
from tensorflow import keras
import numpy as np
import tensorflow.keras.backend as K

import tensorflow as tf
import os

wt0 = 0.0
wt1 = 0.7236091298145506
wt2 = 0.31134094151212555
wt3 = 1.0

wt00 = 0.0
wt11 = 1.0
wt22 = 0.3947627485725536
wt33 = 0.9121874384721403

# Loss Metrics for segmentation

# dice_loss = sm.losses.DiceLoss(class_weights=np.array([wt0, wt1, wt2, wt3])) 
# focal_loss = sm.losses.CategoricalFocalLoss()
# alpha = 0.5
# beta = 0.5
# total_loss = alpha * dice_loss + beta * focal_loss

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

# Loss Metrics for edge detection

def weighted_binary_crossentropy(y_true, y_pred):
    # Calculate class weights to handle class imbalance
    class_weights = K.sum(K.cast(K.equal(y_true, 0), 'float32')) / K.sum(K.cast(K.equal(y_true, 1), 'float32'))
    
    # Calculate weighted binary cross-entropy
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    loss = -(class_weights * y_true * K.log(y_pred) + (1 - y_true) * K.log(1 - y_pred))
    
    return K.mean(loss)

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

losses = {
    "Seg_Output": total_loss,
    # "Edge_Output": weighted_binary_crossentropy
    "Edge_Output": weighted_categorical_crossentropy([wt00, wt11, wt22, wt33])
}

seg_metrics = ['accuracy', sm.metrics.IOUScore(threshold=0.5), dice_coef]
edge_metrics = ['accuracy']

metrics = {
    "Seg_Output": seg_metrics,
    "Edge_Output": edge_metrics
}

smooth = 1e-15

model = load_model('./dual_decoder_simsiam_3d_unet.hdf5',
                   custom_objects={
                       'total_loss': total_loss, 
                       'weighted_categorical_crossentropy': weighted_categorical_crossentropy([wt00, wt11, wt22, wt33]), 
                       'loss': losses,
                       'iou_score':sm.metrics.IOUScore(threshold=0.5),
                       'dice_coef':dice_coef
                    }
        )

def predict(test_img_input):
    # print("Predict called")
    test_prediction_seg, test_prediction_edg = model.predict(test_img_input)
    test_prediction_argmax=np.argmax(test_prediction_seg, axis=4)[0,:,:,:]
    test_prediction_edge_argmax=np.argmax(test_prediction_edg, axis=4)[0,:,:,:]
    # print("Predict done")
    # print(test_prediction_argmax.shape)
    # print(test_prediction_edge_argmax.shape)

    return test_prediction_argmax, test_prediction_edge_argmax, test_prediction_seg, model