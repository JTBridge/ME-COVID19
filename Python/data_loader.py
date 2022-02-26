'''
Data loader. 

Please cite:
Development and External Validation of a Mixed-Effects Deep Learning Model to Diagnose COVID-19 from CT Imaging
Joshua Bridge, Yanda Meng, Wenyue Zhu, Thomas Fitzmaurice, Caroline McCann, Cliff Addison, Manhui Wang, 
Cristin Merritt, Stu Franks, Maria Mackey, Steve Messenger, Renrong Sun, Yitian Zhao, Yalin Zheng
medRxiv 2022.01.28.22270005; doi: https://doi.org/10.1101/2022.01.28.22270005

joshua.bridge@liverpool.ac.uk
github.com/JTBridge/ME-COVID19
Apache License 2.0
'''

import os, random
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image, ImageEnhance
import threading
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from PIL import ImageOps 
import math as maths
 
px = 256

tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Make it threadsafe
class threadsafe_iter:
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return self.it.__next__()
def threadsafe_generator(f):
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g

# Function to add Gaussian noise
def gauss_noise(img, std):
    noise = np.random.normal(size=(px, px), loc=0.0, scale=std) 
    noise = np.repeat(noise[:, :, np.newaxis], 3, axis=2)
    return img + noise  

# Function to augment the data
def augmentation_fn(img, rot, br, cr, flip, mirror, cont):   
    img = img.rotate(rot, resample=Image.BILINEAR)
    ht, wd = img.size
    img = img.crop((cr*wd, cr*ht, (1-cr)*wd, (1-cr)*ht))    
    bright = ImageEnhance.Brightness(img) 
    img = bright.enhance(br)
    contrast = ImageEnhance.Contrast(img) 
    img = contrast.enhance(cont)
    if flip:
        img = ImageOps.flip(img)
    if mirror:
        img = ImageOps.mirror(img)                
    return img

# Load a blank slice
def load_blank(px=px): 
    return np.zeros((px,px,3))


# Image loader
def load_image(path, rot, br,  cr, flip, mirror, cont, 
    augmentation=None, px=px):

    img = image.load_img(path)
    if augmentation:
        img = augmentation(img, rot, br, cr, flip, mirror, cont)      
    img = img.resize((px, px), resample=Image.BILINEAR) 
    img = np.asarray(img) 
    img = img/255.
    return img

# The main data generator
@threadsafe_generator
def data_gen(img_folder, scan_size, batch_size, mtype, px=px, 
                augment=augmentation_fn,
                shuffle=False, sensitivity=False):
    c = 0
    if shuffle:
        n = os.listdir(img_folder)  
        random.shuffle(n)
    else:
        n = sorted(os.listdir(img_folder))

    while True:
        img_out = np.zeros((batch_size, scan_size, px, px, 3))
        label = np.zeros((batch_size, 1)).astype("int")
        Z = np.zeros((batch_size, scan_size, scan_size))
        zero = np.zeros((batch_size, scan_size,))


        for i in range(c, c + batch_size):
            folder = n[i]
            folder_split = folder.split("_")
            if '00NOR' in folder:
                label[i-c, 0] = 0
            elif '02C19' in folder:
                label[i-c, 0] = 1

            # Randomly generate augmentation variables
            # They are selected here so they are constant within scans
            br = random.uniform(0.8, 1.2) 
            cont = random.uniform(0.8, 1.2) 
            rot = random.uniform(-5, 5) 
            cr = random.uniform(-0.2, 0.2) 
            flip = random.getrandbits(1) 
            mirror = random.getrandbits(1) 

            # Add or remove slices to reach required number
            img_names = os.listdir(img_folder + "/" + n[i]+'/')
            if scan_size<=len(img_names):
                choose = np.linspace(0, len(img_names)-1, scan_size, dtype='int')
                img_names = [sorted(img_names)[i] for i in choose] 
                paths = sorted(img_names)                          
            if len(img_names)<scan_size:
                paths = sorted(img_names, key = lambda name: str(name.split('.')[0]))
                img_blank = 'blank_slice'
                step = (scan_size)/(scan_size-len(img_names))
                insert = np.arange(0, scan_size, step, dtype='int') + int(step)//2
                for k in range(len(insert)):
                    paths.insert(insert[k]+1, img_blank)
            
            # Load the slices and place into an array
            if paths[0] == 'blank_slice':
                img_tmp = load_blank()   
            else:
                img_tmp = load_image(img_folder + "/" + n[i]+'/'+paths[0], 
                    rot, br, cr, flip, mirror, cont, 
                    augment)                      
            img_tmp = np.expand_dims(img_tmp, axis=0)
            if scan_size>1:
                for j in range(1, scan_size):
                    if paths[j] == 'blank_slice':
                        img_new = load_blank()   
                    else:
                        img_new = load_image(img_folder + "/" + n[i]+'/'+paths[j], 
                            rot, br, cr, flip, mirror, cont,
                            augment)   
                    img_new = np.expand_dims(img_new, axis=0)                
                    img_tmp = np.append(img_tmp, img_new, axis=0)

            # Adding random noise for the sensitivity analysis
            if sensitivity:
                img_tmp = gauss_noise(img_tmp, maths.sqrt(sensitivity))
            img_out[i-c] = img_tmp 
            Z[i-c] = np.eye(20) 

        c += batch_size
        if c + batch_size >= len(os.listdir(img_folder)):
            c = 0
            if shuffle:
                random.shuffle(n)
        if mtype == 'ME':
            yield [img_out, Z], [zero, label]
        else:
            yield img_out, label
