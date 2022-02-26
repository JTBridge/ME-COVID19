'''
Comparison models, reimplemented based on the papers cited.

Please cite the relevant papers.

joshua.bridge@liverpool.ac.uk
github.com/JTBridge/ME-COVID19
Apache License 2.0
'''

from tensorflow.keras import models, layers, applications
import tensorflow as tf
from tensorflow.python.keras.applications import efficientnet


tf.keras.mixed_precision.set_global_policy('mixed_float16')

def bai():
	'''
	Bai et al. Artificial Intelligence Augmentation of 
	Radiologist Performance in Distinguishing COVID-19 
	from Pneumonia of Other Origin at Chest CT. 
	Radiology, 2020. https://doi.org/10.1148/radiol.2020201491
	'''

	mod = efficientnet.EfficientNetB4(weights='imagenet', include_top=False, input_shape=(256,256,3))
	input1 = layers.Input((20,256,256,3))
	x = layers.TimeDistributed(mod)(input1)
	x = layers.TimeDistributed(layers.Flatten())(x)
	x = layers.TimeDistributed(layers.Dense(256, activation='relu'))(x)
	x = layers.TimeDistributed(layers.BatchNormalization())(x)
	x = layers.TimeDistributed(layers.Dropout(0.5))(x)
	x = layers.TimeDistributed(layers.Dense(128, activation='relu'))(x)
	x = layers.TimeDistributed(layers.BatchNormalization())(x)
	x = layers.TimeDistributed(layers.Dropout(0.5))(x)
	x = layers.TimeDistributed(layers.Dense(64, activation='relu'))(x)
	x = layers.TimeDistributed(layers.BatchNormalization())(x)
	x = layers.TimeDistributed(layers.Dropout(0.5))(x)
	x = layers.TimeDistributed(layers.Dense(32, activation='relu'))(x)
	x = layers.TimeDistributed(layers.BatchNormalization())(x)
	x = layers.TimeDistributed(layers.Dropout(0.5))(x)
	x= layers.Activation('relu')(x)
	x = layers.BatchNormalization()(x)
	x = layers.TimeDistributed(layers.Dense(16, activation='relu'))(x)
	x = layers.TimeDistributed(layers.Dense(1, activation='sigmoid'))(x)
	x = layers.GlobalAveragePooling1D()(x)                             
	out = layers.Dense(1, activation='sigmoid', dtype='float32', name='out')(x)   
	return models.Model([input1], [out])

def covinet():
	'''
	Mittal et al. CoviNet: Covid-19 diagnosis using machine 
	learning analyses for computerized tomography images. 2021.
	ICDIP 2021. https://doi.org/10.1117/12.2601065
	'''
	inputs = layers.Input((20,256,256,3))
	x = layers.Conv3D(64, (3, 3, 3), padding='same', activation='relu')(inputs)
	x = layers.MaxPooling3D(2,2)(x)
	x = layers.BatchNormalization()(x)
	x = layers.Conv3D(128, (3, 3, 3), padding='same', activation='relu')(x)
	x = layers.MaxPooling3D(2,2)(x)
	x = layers.BatchNormalization()(x)
	x = layers.Conv3D(256, (3, 3, 3), padding='same', activation='relu')(x)
	x = layers.MaxPooling3D(2,2)(x)
	x = layers.BatchNormalization()(x)       
	x = layers.Conv3D(256, (3, 3, 3), padding='same', activation='relu')(x)
	x = layers.MaxPooling3D(2,2)(x)
	x = layers.BatchNormalization()(x)            
	x = layers.GlobalAveragePooling3D()(x)
	x = layers.Dense(512)(x)
	x = layers.Dropout(0.3)(x)
	x = layers.Dense(1, activation='sigmoid', dtype='float32')(x)       
	return models.Model(inputs, outputs)

def covnet(px=256, slices=20):
	'''
	Li et al. Using Artificial Intelligence to Detect COVID-19 
	and Community-acquired Pneumonia Based on Pulmonary CT: 
	Evaluation of the Diagnostic Accuracy. Radiology, 2020.
	https://doi.org/10.1148/radiol.2020200905
	'''
	mod = applications.ResNet50(weights='imagenet', include_top=False, input_shape=(px,px,3))
	input1 = layers.Input((slices,px,px,3))
	x = layers.TimeDistributed(mod)(input1)
	x = layers.TimeDistributed(layers.GlobalAveragePooling2D())(x)
	x = layers.GlobalMaxPooling1D()(x)
	out = layers.Dense(1, activation='sigmoid', dtype='float32')(x)               
	return models.Model(input1, out)
