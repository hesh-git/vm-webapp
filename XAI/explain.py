from .config import DIMENSIONS, XAI_MODES, MODALITIES, XAIs
from .process import get_last_layer, get_last_conv_layer
from .visualize import show_image, show_gray_image, show_heatmap, visualize_tensor
from .visualize import overlay_gradcam, overlay_grad, overlay_pred
from .grad_cam import get_grad_cam

import matplotlib.pyplot as plt
import numpy as np
import os

# Visualization and save of all NeuroXAI methods
def get_neuroxai(ID, model, io_imgs, CLASS_ID=0, SLICE_ID=77, LAYER_NAME=None, 
                 MODALITY="FLAIR", XAI_MODE="classification", 
                 DIMENSION="2d", CLASS_IDs=[0,1], TUMOR_LABEL="all", SAVE_RESULTS=False, SAVE_PATH=None):

    # sanity checks
    assert DIMENSION in DIMENSIONS, "Input dimension must be in {}".format([d for d in DIMENSIONS])
    assert XAI_MODE in XAI_MODES, "XAI mode must be in {}".format([m for m in XAI_MODES])
    assert MODALITY in MODALITIES, "MRI modality must be in {}".format([m for m in MODALITIES])

    modality_dict = {"FLAIR": 0, "T1": 1, "T1CE": 2, "T2": 3}

    if LAYER_NAME==None:
        LAYER_NAME = get_last_layer(model).name
        CONV_LAYER_NAME = get_last_conv_layer(model, DIMENSION).name

    else:
        # A solution for classification visualization of all XAI
        layer = model.get_layer(LAYER_NAME)
        last_layer = get_last_layer(model)
        if layer.name == last_layer.name:
            if (not isinstance(layer, Conv2D)) or (not isinstance(layer, Conv3D)):
                CONV_LAYER_NAME = get_last_conv_layer(model, DIMENSION).name
            else:
                CONV_LAYER_NAME = LAYER_NAME
        else:
            CONV_LAYER_NAME = LAYER_NAME
    
    print("Visual exaplanations for...\n\t ID: {}, layer: {}".format(ID, LAYER_NAME))
    # Build the model
    if XAI_MODE=="classification": # 2d and 3d
        im_orig = io_imgs[0]
        SmoothXAI="IG"
    elif XAI_MODE=="segmentation": # 3d only
        im_orig = io_imgs[0,:,:,SLICE_ID,modality_dict[MODALITY]] # (1, 192, 224, 160, 4)
        SmoothXAI="GIG"

    # Get XAI heat maps (1 samples for SmoothGrad)

    
    if XAI_MODE=="segmentation" and TUMOR_LABEL=="all":
        gcam_grads = get_grad_cam(model, io_imgs, CLASS_IDs[0], CONV_LAYER_NAME, MODALITY, XAI_MODE, DIMENSION)
        for c_id in CLASS_IDs[1:]:
            gcam_grads += get_grad_cam(model, io_imgs, c_id, CONV_LAYER_NAME, MODALITY, XAI_MODE, DIMENSION)
    else:
        gcam_grads = get_grad_cam(model, io_imgs, CLASS_ID, CONV_LAYER_NAME, MODALITY, XAI_MODE, DIMENSION)
    
    
    # Get 2D images for visualizations
    # Call the visualization methods to convert the 3D tensors to 2D grayscale.
    gradcam = visualize_tensor(gcam_grads)
    
    if DIMENSION=="3d":
        gradcam = gradcam[:,:,SLICE_ID]

    # Get overlay images (Over FLAIR MRI)
    gradcam_overlay = overlay_gradcam(im_orig, gradcam, DIMENSION)
 

    # Set up matplot lib figures.
    ROWS = 1
    COLS = 11 if XAI_MODE=="segmentation" else 9
    UPSCALE_FACTOR = 25
    plt.figure(figsize=(ROWS * UPSCALE_FACTOR, COLS * UPSCALE_FACTOR))

    # Render the saliency masks.
    show_heatmap(gradcam, TITLE='Grad-CAM', AX=plt.subplot(ROWS, COLS, 7), CMAP="jet")

    
    if XAI_MODE=="segmentation":
        # Predict the tumor segmentation
        _, prediction_3ch = np.squeeze(model(io_imgs)) # model prediction 
        prediction = np.argmax(prediction_3ch, axis=-1)
        prediction = (prediction[:,:,SLICE_ID] == CLASS_ID).astype(np.uint8)
        if CLASS_ID!= 0:
            prediction[prediction>0] = CLASS_ID

        pred_overlay = overlay_pred(io_imgs[0,:,:,SLICE_ID,modality_dict[MODALITY]], prediction)
        show_image(prediction, TITLE='Prediction', AX=plt.subplot(ROWS, COLS, 10))
        show_image(pred_overlay, TITLE='Prediction Overlay', AX=plt.subplot(ROWS, COLS, 11))
        
    # Save the results
    if SAVE_RESULTS:
        if not os.path.exists(os.path.join(SAVE_PATH, ID)):
            os.makedirs(os.path.join(SAVE_PATH, ID))
        # Save 2D heatmaps
        if XAI_MODE=="classification":
            LAYER_NAME=""
        if TUMOR_LABEL=="all":
            CLASS_ID="all"

        plt.imsave("{}_{}_{}_gcam.png".format(SAVE_PATH, LAYER_NAME, CLASS_ID), gradcam, CMAP="jet")


# Visualization and save of a single NeuroXAI method
def visualize_neuroxai_cnn(ID, model, io_imgs, grads, CLASS_ID=0, SLICE_ID=77,
                           LAYER_NAME=None, MODALITY="FLAIR", 
                           XAI_MODE="classification", XAI="GCAM",
                           DIMENSION="2d", TUMOR_LABEL="all",
                           SAVE_RESULTS=False, SAVE_PATH=None):

    modality_dict = {"FLAIR": 0, "T1": 1, "T1CE": 2, "T2": 3}
        
    if XAI_MODE=="classification": # 2d and 3d
        im_orig = io_imgs[0]
        SmoothXAI="IG"
        #smooth_grads = get_smoothgrad(model, io_imgs, CLASS_ID, XAI_MODE=XAI_MODE, XAI="IG")
    elif XAI_MODE=="segmentation": # 3d only
        im_orig = io_imgs[0,:,:,SLICE_ID,modality_dict[MODALITY]] # (1, 192, 224, 160, 4)
        SmoothXAI="GIG"
        
    # Get 2D images for visualizations
    heatmap = visualize_tensor(grads)    
    if DIMENSION=="3d":
        heatmap = heatmap[:,:,SLICE_ID]

    # Set up matplot lib figures.
    ROWS = 1
    COLS = 5 if XAI_MODE=="segmentation" else 3
    UPSCALE_FACTOR = 25 if XAI_MODE=="segmentation" else 10
    plt.figure(figsize=(ROWS * UPSCALE_FACTOR, COLS * UPSCALE_FACTOR))
    
    # Render the saliency masks.
    show_gray_image(im_orig, TITLE=MODALITY, AX=plt.subplot(ROWS, COLS, 1))
    # Get overlay images (Over MRI)
    if XAI=="GCAM" or XAI=="GGCAM":
        overlay = overlay_gradcam(im_orig, heatmap, DIMENSION)
        if LAYER_NAME==None:
            LAYER_NAME = get_last_conv_layer(model, DIMENSION).name
        show_heatmap(heatmap, TITLE=XAI+'-'+LAYER_NAME+'-'+str(CLASS_ID), AX=plt.subplot(ROWS, COLS, 2), CMAP="jet")
    else:
        overlay = overlay_grad(im_orig, heatmap, DIMENSION)
        if LAYER_NAME==None:
            LAYER_NAME = get_last_layer(model).name
        show_gray_image(heatmap, TITLE=XAI+'-'+LAYER_NAME+'-'+str(CLASS_ID), AX=plt.subplot(ROWS, COLS, 2)) # CMAP=plt.cm.gray
    show_image(overlay, TITLE=XAI+' Overlay', AX=plt.subplot(ROWS, COLS, 3)) 
    
    
    # Save the results
    if SAVE_RESULTS:
        if not os.path.exists(os.path.join(SAVE_PATH, ID)):
            os.makedirs(os.path.join(SAVE_PATH, ID))
    
        if XAI_MODE=="classification":
            LAYER_NAME=""
        if TUMOR_LABEL=="all":
            CLASS_ID="all"
        
        # Save 2D heatmaps
        if XAI=="GCAM":
            plt.imsave("{}/{}/{}_{}_{}_{}.png".format(SAVE_PATH, ID,LAYER_NAME, CLASS_ID, MODALITY, XAI), heatmap, cmap="jet")
        else:
            plt.imsave("{}/{}/{}_{}_{}_{}.png".format(SAVE_PATH, ID,LAYER_NAME, CLASS_ID, MODALITY, XAI), heatmap, cmap=plt.cm.gray)
            
    else:
        return heatmap
            
        
def get_neuroxai_cnn(ID, model, io_imgs, CLASS_ID=0, SLICE_ID=77, LAYER_NAME=None,
                     MODALITY="FLAIR", XAI_MODE="classification", XAI="GCAM", 
                     DIMENSION="2d", CLASS_IDs=[0,1], TUMOR_LABEL="all", 
                     SAVE_RESULTS=False, SAVE_PATH=None):
    # sanity checks
    assert DIMENSION in DIMENSIONS, "Input dimension must be in {}".format([d for d in DIMENSIONS])
    assert XAI_MODE in XAI_MODES, "XAI mode must be in {}".format([m for m in XAI_MODES])
    assert MODALITY in MODALITIES, "MRI modality must be in {}".format([m for m in MODALITIES])
    assert XAI in XAIs, "XAI method must be in {}".format([m for m in XAIs])

    if LAYER_NAME==None:
        LAYER_NAME = get_last_layer(model).name
        CONV_LAYER_NAME = get_last_conv_layer(model, DIMENSION).name

    else:
        layer = model.get_layer(LAYER_NAME)
        last_layer = get_last_layer(model)
        if layer.name == last_layer.name:
            if (not isinstance(layer, Conv2D)) or (not isinstance(layer, Conv3D)):

                CONV_LAYER_NAME = get_last_conv_layer(model, DIMENSION).name
            else:
                CONV_LAYER_NAME = LAYER_NAME
        else:
            CONV_LAYER_NAME = LAYER_NAME


    print("Visual exaplanations for...\n\t ID: {}, layer: {}".format(ID, LAYER_NAME))
    # Get the gradients    
    LAYER_NAME = CONV_LAYER_NAME
    if XAI_MODE=="segmentation" and TUMOR_LABEL=="all":
        grads = get_grad_cam(model, io_imgs, CLASS_IDs[0], LAYER_NAME, MODALITY, XAI_MODE, DIMENSION)
        for c_id in CLASS_IDs[1:]:
            grads += get_grad_cam(model, io_imgs, c_id, LAYER_NAME, MODALITY, XAI_MODE, DIMENSION)
    else:
        grads = get_grad_cam(model, io_imgs, CLASS_ID, LAYER_NAME, MODALITY, XAI_MODE, DIMENSION)

    # Visualize the saliency map
    return visualize_neuroxai_cnn(ID, model, io_imgs, grads, CLASS_ID, SLICE_ID, LAYER_NAME, MODALITY, 
                           XAI_MODE, XAI, DIMENSION, TUMOR_LABEL, SAVE_RESULTS, SAVE_PATH)

