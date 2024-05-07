import numpy as np
import nibabel as nib
from sklearn.preprocessing import MinMaxScaler
# from keras.utils import to_categorical
from glob import glob
import os

def preprocess(temp_data_directory):

    scaler = MinMaxScaler()

    flair_path = glob(os.path.normpath('data/*/*_flair.nii'))[0]
    t1_path = glob(os.path.normpath('data/*/*_t1.nii'))[0]
    t2_path = glob(os.path.normpath('data/*/*_t2.nii'))[0]
    t1ce_path = glob(os.path.normpath('data/*/*_t1ce.nii'))[0]  
    
    temp_image_t2=nib.load(t2_path).get_fdata()
    temp_image_t2=scaler.fit_transform(temp_image_t2.reshape(-1, temp_image_t2.shape[-1])).reshape(temp_image_t2.shape)
    
    temp_image_t1ce=nib.load(t1ce_path).get_fdata()
    temp_image_t1ce=scaler.fit_transform(temp_image_t1ce.reshape(-1, temp_image_t1ce.shape[-1])).reshape(temp_image_t1ce.shape)
    
    temp_image_flair=nib.load(flair_path).get_fdata()
    temp_image_flair=scaler.fit_transform(temp_image_flair.reshape(-1, temp_image_flair.shape[-1])).reshape(temp_image_flair.shape)
    temp_image_t1 = nib.load(t1_path).get_fdata()
    temp_image_t1=scaler.fit_transform(temp_image_t1.reshape(-1, temp_image_t1.shape[-1])).reshape(temp_image_t1.shape)

    temp_combined_images = np.stack([temp_image_flair, temp_image_t1ce, temp_image_t2,temp_image_t1], axis=3)
    #Crop to a size to be divisible by 64 so we can later extract 64x64x64 patches. 
    #cropping x, y, and z
    test_img=temp_combined_images[56:184, 56:184, 13:141]
    
    test_img_input = np.expand_dims(test_img, axis=0)

    return test_img_input
