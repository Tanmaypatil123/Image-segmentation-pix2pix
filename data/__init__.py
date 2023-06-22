from tensorflow import keras
import numpy as np
from PIL import Image
import os
### Custome data generator
class Datagenerator(keras.utils.Sequence):
    def __init__(self, root_dir,batch_size = 32,augment = True,shuffle = False):
        self.root_dir = root_dir
        self.list_files = os.listdir(self.root_dir)
        self.batch_size = batch_size
        self.augment_model = keras.Sequential([
            keras.layers.Rescaling(1./255)
        ])
        self.augment = augment
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        ## Will return how many batches present in dataset
        return len(self.list_files) // self.batch_size
    
    def __getitem__(self, index):
        ## Selecting data instances from directory for particular batch
        batch_indexes = self.list_files[index*self.batch_size:(index+1)*self.batch_size]
        target_ = []
        input_ = []
        for i in batch_indexes :
            img_path = os.path.join(self.root_dir, i)
            img = np.array(Image.open(img_path))
            # left side of image is input
            input_image = img[:, :256, :]
            # Right side of image will be output
            target_image = img[:,256:,:]
            if self.augment == True :
                target_image = self.augment_model(target_image)
                input_image = self.augment_model(input_image)
            target_.append(target_image)
            input_.append(input_image)
        
        return np.array(input_) , np.array(target_)
    
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_files))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)