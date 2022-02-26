'''
Code for a mixed-effects deep learning layer.

Please cite:
Development and External Validation of a Mixed-Effects Deep Learning Model to Diagnose COVID-19 from CT Imaging
Joshua Bridge, Yanda Meng, Wenyue Zhu, Thomas Fitzmaurice, Caroline McCann, Cliff Addison, Manhui Wang, 
Cristin Merritt, Stu Franks, Maria Mackey, Steve Messenger, Renrong Sun, Yitian Zhao, Yalin Zheng
medRxiv 2022.01.28.22270005; doi: https://doi.org/10.1101/2022.01.28.22270005

joshua.bridge@liverpool.ac.uk
github.com/JTBridge/ME-COVID19
Apache License 2.0
'''

import tensorflow as tf
from tensorflow.keras import models, layers, applications, regularizers
from spatial import ME
from tensorflow.keras.mixed_precision import experimental as mixed_precision

tf.keras.mixed_precision.set_global_policy('mixed_float16')


slices=20
px = 256


def ME_model():
	mod = applications.InceptionV3(weights='imagenet', include_top=False, input_shape=(px,px,3))
	input1 = layers.Input((slices,px,px,3))
	input2 = layers.Input((slices,slices))
	x = layers.TimeDistributed(mod)(input1)
	x = layers.TimeDistributed(layers.GlobalAveragePooling2D())(x)
	x = layers.Dropout(0.6)(x)
	x, beta = ME(name='ME',
		fixed_intercept= True,
		random_intercept= True
		)([x, input2])
	beta = layers.Activation('linear', name='beta')(beta)
	x = layers.Dense(1, activation=None, dtype='float32', 
		kernel_regularizer = regularizers.l1_l2(l1=1.0, l2=0.1)
		)(x)
	out = layers.Activation('sigmoid', name='prob')(x)
	return models.Model([input1, input2], [beta, out])
