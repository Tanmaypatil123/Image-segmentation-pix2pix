import tensorflow as tf
from tensorflow import keras


class CNNBlock(keras.Model):
    def __init__(self,filters,size,apply_batchnorm = True,stride = 2,**kwargs):
        super().__init__(**kwargs)
        self.apply_batchnorm = apply_batchnorm
        self.initializer =  tf.random_normal_initializer(0., 0.02)
        self.conv = keras.layers.Conv2D(filters=filters,kernel_size=size,strides= stride,padding="same",use_bias=False,kernel_initializer=self.initializer)
        self.batch_norm = keras.layers.BatchNormalization()
        self.leaky_relu = keras.layers.LeakyReLU(0.2)
    def call(self, inputs):
        x = self.conv(inputs)
        if self.apply_batchnorm == True :
            x = self.batch_norm(x)
        x = self.leaky_relu(x)
        return x


def Discriminator():
    initializer = tf.random_normal_initializer(0.,0.02)
    input_ = tf.keras.layers.Input(shape=[256,256,3],name = 'input_image')
    target_ = tf.keras.layers.Input(shape = [256,256,3],name = 'target_image')

    x = tf.keras.layers.concatenate([input_,target_])

    x = CNNBlock(filters=64,size=4,apply_batchnorm=False)(x)
    x = CNNBlock(128,size=4,stride=2)(x)
    x = CNNBlock(256,size=4,stride=2)(x)
    
    x = keras.layers.ZeroPadding2D()(x)
    x = keras.layers.Conv2D(512, 4, strides=1,
                                kernel_initializer=initializer,
                                use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.ZeroPadding2D()(x)
    x = keras.layers.Conv2D(1, 4, strides=1,
                                kernel_initializer=initializer)(x)
    return keras.Model(inputs=[input_,target_],outputs = x)
