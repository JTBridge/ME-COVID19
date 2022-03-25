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


from tensorflow.keras.mixed_precision import experimental as mixed_precision
from tensorflow.keras import regularizers, initializers, losses
from tensorflow.keras.layers import Layer   
import tensorflow as tf

tf.keras.mixed_precision.set_global_policy('mixed_float16')


class ME(Layer):
    '''
    Mixed effects layer.
    ''' 

    def __init__(self, 
        fixed_intercept=False, random_intercept=False,
        fixed_regularizer = None, random_regularizer = None,
        **kwargs):
        self.fixed_intercept = fixed_intercept
        self.random_intercept = random_intercept
        self.fixed_regularizer = fixed_regularizer
        self.random_regularizer = random_regularizer
        super(ME, self).__init__(**kwargs)

    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'fixed_intercept': self.fixed_intercept,
            'random_intercept': self.random_intercept,
            'fixed_regularizer' : self.fixed_regularizer,
            'random_regularizer' : self.random_regularizer,
        })
        return config        


    def build(self, input_shape):
        if self.fixed_intercept:
            self.alpha = self.add_weight(name='alpha', 
                                          shape=(input_shape[0][2]+1,),
                                          initializer=initializers.HeNormal(),
                                          regularizer = self.fixed_regularizer,
                                          trainable=True,
                                          dtype="float32")   

        else:
            self.alpha = self.add_weight(name='alpha', 
                                          shape=(input_shape[0][2],),
                                          initializer=initializers.HeNormal(),
                                          regularizer = self.fixed_regularizer,
                                          trainable=True,
                                          dtype="float32")   
        if self.random_intercept:          
            self.beta = self.add_weight(name='beta', 
                                          shape=(input_shape[1][2]+1,),
                                          initializer=initializers.HeNormal(),
                                              regularizer = self.random_regularizer,
                                          trainable=True,
                                          dtype="float32")     
        else:
            self.beta = self.add_weight(name='beta', 
                                          shape=(input_shape[1][2],),
                                          initializer=initializers.HeNormal(),
                                              regularizer = self.random_regularizer,
                                          trainable=True,
                                          dtype="float32")                                                                                                                                                                                                            
        super(ME, self).build(input_shape) 

    def call(self, x):
        # Make sure everything is in Float32
        X = tf.cast(x[0], dtype = tf.float32)
        Z = tf.cast(x[1], dtype = tf.float32)
        alpha = tf.cast(self.alpha, dtype = tf.float32)
        beta = tf.cast(self.beta, dtype = tf.float32)

        # Add an intercept to the fixed effects design matrix
        if self.fixed_intercept:
            intercept = tf.ones((tf.shape(X)[0], tf.shape(X)[1], 1))
            X = tf.concat([intercept, X], 2)
        if self.random_intercept:
            intercept = tf.ones((tf.shape(Z)[0], tf.shape(Z)[1], 1))
            Z = tf.concat([intercept, Z], 2)

        # This is the mixed effects model Y = X\alpha + Z\beta
        ME = tf.linalg.matvec(X, alpha)  + tf.linalg.matvec(Z, beta)

        # Output the mixed effects model and the random effects parameters,
        # we can then enforce the assumptions of mean 0 and normality on the random effects.
        return ME, beta 
    def compute_output_shape(self, input_shape):
        return input_shape[0][0], (input_shape[1][-1])   


def normal(y_true, y_pred):
    '''
    Loss function for the random effects parameters.
    Aim to make the parameters normally distributed with mean 0 by enforcing:
    - Mean = 0
    - Skewness = 0
    - Excess kurtosis = 0 (or kurtosis = 3 for the normal distribution)
    '''
    y_true = tf.cast(y_true, dtype = tf.float32)
    y_pred = tf.cast(y_pred, dtype = tf.float32) 

    n = tf.cast(tf.shape(y_pred)[-1], tf.float32)
    mean = tf.abs(tf.math.reduce_mean(y_pred))   
    skew_num = tf.math.reduce_mean((y_pred-mean)**3)
    skew_den = tf.math.reduce_mean((y_pred-mean)**2)**1.5 
    skew = tf.abs((tf.math.pow(n*(n-1),0.5)/(n-2))*(skew_num/skew_den))
    kurt_num = tf.math.reduce_mean((y_pred-mean)**4)
    kurt_den = tf.math.reduce_mean((y_pred-mean)**2)**2
    kurt = tf.abs((tf.math.reduce_mean((kurt_num/kurt_den))/n**2)-3.)
    return mean+skew+kurt

def mse_wgt(class_weights=[1,1], targets=[0,1]):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, dtype = tf.int32)
        y_pred = tf.cast(y_pred, dtype = tf.float32) 
        weights = tf.reduce_sum(tf.one_hot(y_true, depth=2)*class_weights, axis=-1)
        y_true = tf.reduce_sum(tf.one_hot(y_true, depth=2)*targets, axis=-1)
        mse = losses.MeanSquaredError()
        return mse(y_true, y_pred, weights)
    return loss
