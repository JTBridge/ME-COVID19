'''
Code for output results for analysis.

Please cite:
Development and External Validation of a Mixed-Effects Deep Learning Model to Diagnose COVID-19 from CT Imaging
Joshua Bridge, Yanda Meng, Wenyue Zhu, Thomas Fitzmaurice, Caroline McCann, Cliff Addison, Manhui Wang, 
Cristin Merritt, Stu Franks, Maria Mackey, Steve Messenger, Renrong Sun, Yitian Zhao, Yalin Zheng
medRxiv 2022.01.28.22270005; doi: https://doi.org/10.1101/2022.01.28.22270005

joshua.bridge@liverpool.ac.uk
github.com/JTBridge/ME-COVID19
Apache License 2.0
'''

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from data_loader import data_gen
from tensorflow.keras import losses, optimizers, models, metrics, layers, applications, regularizers, backend
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import os, sklearn
import numpy as np
import matplotlib.pyplot as plt
from spatial import ME, normal, mse_wgt
from ME import ME_model
from comparison_models import bai, covinet, covnet
from sklearn.metrics import roc_curve, auc
import tensorflow as tf

 
tf.keras.mixed_precision.set_global_policy('mixed_float16')

mtype = 'ME'
px = 256
slices = 20

wgt0 = (1/169) * (1025/2.0)
wgt1 = (1/856) * (1025/2.0)
t0 = 1/(169+2)
t1 = (856+1)/(856+2)

if mtype == 'ME':
    model=ME_model()
    model.compile( 
        loss={'beta':normal, 'prob':mse_wgt(class_weights=[wgt0, wgt1], targets=[t0, t1])},
        optimizer=optimizers.Adam(1e-4),
        metrics={'prob':[metrics.AUC(name='AUC')]}
        )
if mtype == 'Bai':
    model=bai()
    model.compile( 
        loss=mse_wgt(class_weights=[wgt0, wgt1], targets=[t0, t1]),
        optimizer=optimizers.Adam(1e-4),
        metrics=metrics.AUC(name='AUC'))
if mtype == 'covinet':
    model=covinet()
    model.compile( 
        loss=mse_wgt(class_weights=[wgt0, wgt1], targets=[t0, t1]),
        optimizer=optimizers.Adam(1e-4),
        metrics=metrics.AUC(name='AUC'))
if mtype == 'covnet':
    model=covnet()
    model.compile( 
        loss=mse_wgt(class_weights=[wgt0, wgt1], targets=[t0, t1]),
        optimizer=optimizers.Adam(1e-4),
        metrics=metrics.AUC(name='AUC'))


print(model.summary())
if mtype == "ME":
    model.load_weights('../models/ME.h5')
elif mtype == 'Bai':
    model.load_weights('../models/Bai.h5')
elif mtype == 'covinet':
    model.load_weights('../models/covinet.h5')
elif mtype == 'covnet':
    model.load_weights('../models/covnet.h5')

#############################################################################
print('Internal Validation: Mosmed')
img_dir = '../data/mosmed/val/'
test_generator = data_gen(
    img_dir, slices, 1, mtype=mtype, px=px, augment=False, shuffle=False)
test_steps = len(os.listdir(img_dir))
if mtype == 'ME':
    predict = model.predict(test_generator, steps=test_steps, verbose=1)[1]
else:
    predict = model.predict(test_generator, steps=test_steps, verbose=1)
class_nor = np.squeeze(np.repeat(0, 85))
class_c19 = np.squeeze(np.repeat(1, 285))
true_class = np.concatenate((class_nor, class_c19), axis=0)
fpr, tpr, thresholds = roc_curve(true_class, predict, pos_label=1)
print("AUC:",auc(fpr, tpr))

np.savetxt('../results/internal/true.csv', true_class)   
if mtype == "ME":
    np.savetxt('../results/internal/ME.csv', predict) 
elif mtype == 'Bai':
    np.savetxt('../results/internal/Bai.csv', predict)
elif mtype == 'covinet':
    np.savetxt('../results/internal/covinet.csv', predict)    
elif mtype == 'covnet':
    np.savetxt('../results/internal/covnet.csv', predict)    

#############################################################################
print('External Validation: Zhang et al.')
img_dir = '../data/zhang/'
test_generator = data_gen(
    img_dir, slices, 1, mtype=mtype, px=px, augment=False, shuffle=False)

test_steps = len(os.listdir(img_dir))
if mtype == 'ME':
    predict = model.predict(test_generator, steps=test_steps, verbose=1)[1]
else:
    predict = model.predict(test_generator, steps=test_steps, verbose=1)
class_nor = np.squeeze(np.repeat(0, 243))
class_c19 = np.squeeze(np.repeat(1, 553))
true_class = np.concatenate((class_nor, class_c19), axis=0)

fpr, tpr, thresholds = roc_curve(true_class, predict, pos_label=1)
print(auc(fpr, tpr))
    
np.savetxt('../results/external/true.csv', true_class)   
if mtype == "ME":
    np.savetxt('../results/external/ME.csv', predict) 
elif mtype == 'Bai':
    np.savetxt('../results/external/Bai.csv', predict)
elif mtype == 'covinet':
    np.savetxt('../results/external/covinet.csv', predict) 
elif mtype == 'covnet':
    np.savetxt('../results/external/covnet.csv', predict) 

#############################################################################
if mtype=='ME':
    print('Sensitvity')
    print("Missing data")
    img_dir = '../data/missing/'
    for i in range(10,20):
        test_generator = data_gen(
            img_dir+str(i)+'/', slices, 1, mtype=mtype, px=px, augment=False, shuffle=False)
        test_steps = len(os.listdir(img_dir+str(i)+'/'))
        if mtype == 'ME':
            predict = model.predict(test_generator, steps=test_steps, verbose=1)[1]
        else:
            predict = model.predict(test_generator, steps=test_steps, verbose=1)    
        class_nor = np.squeeze(np.repeat(0, 243))
        class_c19 = np.squeeze(np.repeat(1, 553))
        true_class = np.concatenate((class_nor, class_c19), axis=0)
        fpr, tpr, thresholds = roc_curve(true_class, predict, pos_label=1)
        print("AUC for", i, ":",auc(fpr, tpr))
        np.savetxt('../results/sensitivity/missing/'+mtype+'/slices'+str(i)+'.csv', predict)

    print("Noise")
    img_dir = '../data/zhang/'
    for i in range(1,11):
        test_generator = data_gen(
            img_dir, slices, 1, mtype=mtype, px=px, augment=False, shuffle=False, sensitivity=i*0.001)
        test_steps = len(os.listdir(img_dir))
        if mtype == 'ME':
            predict = model.predict(test_generator, steps=test_steps, verbose=1)[1]
        else:
            predict = model.predict(test_generator, steps=test_steps, verbose=1)
        class_nor = np.squeeze(np.repeat(0, 243))
        class_c19 = np.squeeze(np.repeat(1, 553))
        true_class = np.concatenate((class_nor, class_c19), axis=0)
        fpr, tpr, thresholds = roc_curve(true_class, predict, pos_label=1)
        print("AUC for", i, ":",auc(fpr, tpr))
        np.savetxt('../results/sensitivity/noise/'+mtype+'/noise'+str(i)+'.csv', predict)
