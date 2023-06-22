import tensorflow as tf
from tensorflow import keras


class UpsampleBlock(keras.Model):
    def __init__(self,filters ,down = True,use_dropout = False,act = "relu", trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self.conv = keras.Sequential()

        if down :
            self.conv.add(keras.layers.Conv2D(filters,4,2,"same",use_bias=False))
        else :
            self.conv.add(keras.layers.Conv2DTranspose(filters,4,2,"same",use_bias=False))
        
        self.conv.add(keras.layers.BatchNormalization())
        if act == "relu":
            self.conv.add(keras.layers.ReLU())
        else :
            self.conv.add(keras.layers.LeakyReLU(0.2))
        self.dropout = keras.layers.Dropout(0.5)
        self.use_dropout = use_dropout
        self.down = down
    def call(self, inputs, *args, **kwargs):
        x = self.conv(inputs)
        return self.dropout(x) if self.use_dropout else x


def Generator():
    input_ = keras.layers.Input(shape = [256,256,3])
    down =[ keras.Sequential([
            keras.layers.Conv2D(64,4,2,"same"),
            keras.layers.LeakyReLU(0.2)
        ]) ,
        UpsampleBlock(128,down=True,act="leaky",use_dropout=False),
        UpsampleBlock(256,down=True,act="leaky",use_dropout=False),
        UpsampleBlock(512,down=True,act="leaky",use_dropout=False),
        UpsampleBlock(512,down=True,act="leaky",use_dropout=False),
        UpsampleBlock(512,down=True,act="leaky",use_dropout=False),
        UpsampleBlock(512,down=True,act="leaky",use_dropout=False),
        UpsampleBlock(512,down=True,act="leaky",use_dropout=False)
    ] 
    up = [
            UpsampleBlock(512,down=False,act="relu",use_dropout=True),
            UpsampleBlock(512,down=False,act="relu",use_dropout=True),
            UpsampleBlock(512,down=False,act="relu",use_dropout=True),
            UpsampleBlock(512,down=False,act="relu",use_dropout=False),
            UpsampleBlock(256,down=False,act="relu",use_dropout=False),
            UpsampleBlock(128,down=False,act="relu",use_dropout=False),
            UpsampleBlock(64,down=False,act="relu",use_dropout=False)
        ]
    finelup = keras.layers.Conv2DTranspose(3,4,strides=2,padding="same",kernel_initializer=tf.random_normal_initializer(0.,0.02),activation='tanh')
    x = input_
    skips = []
    for i in down :
        x = i(x)
        skips.append(x)
    skips = reversed(skips[:-1])
    for i , skip in zip(up,skips):
        x = i(x)
        x = keras.layers.Concatenate()([x,skip])
    x = finelup(x)
    return keras.Model(inputs = input_,outputs = x)