'''
Code for training proposed and comparison models.

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
import tensorflow as tf
from tensorflow.keras import optimizers, metrics, losses
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from data_loader import data_gen
from spatial import ME, normal, bce_wgt, mse_wgt
from ME import ME_model
from comparison_models import bai, covinet, covnet

tf.keras.mixed_precision.set_global_policy('mixed_float16')

train_batch = 8
val_batch = 10
mtype = 'ME'
slices = 20
px = 256

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

train_dir = "../data/mosmed/train/"
val_dir = "../data/mosmed/val/"

train_generator = data_gen(
    train_dir, slices, train_batch, mtype=mtype, px=px, shuffle=True)
val_generator = data_gen(
    val_dir, slices, val_batch, mtype=mtype, px=px, augment=False, shuffle=True)

train_steps = len(os.listdir(train_dir))// train_batch
val_steps = len(os.listdir(val_dir))// val_batch

if mtype == "ME":
    save_to = '../models/ME.h5'
elif mtype == "Bai":
    save_to = '../models/Bai.h5'
elif mtype == "covinet":
    save_to = '../models/covinet.h5'
elif mtype == "covnet":
    save_to = '../models/covnet.h5'


mcp_save = ModelCheckpoint(
    save_to, save_best_only=True, monitor='val_loss', mode='min', verbose=1)
earlyStopping = EarlyStopping(
    monitor='val_loss', patience=10, verbose=0, mode='min')
reduceLROnPlat = ReduceLROnPlateau(
    monitor="val_loss", factor=0.2, patience=3, verbose=1, mode="min", epsilon=0.0001)



history = model.fit(
    train_generator,
    steps_per_epoch=train_steps,
    epochs=100,
    verbose=1,
    validation_data=val_generator,
    validation_steps=val_steps,
    callbacks=[
               reduceLROnPlat,
               earlyStopping,
               mcp_save]
)
