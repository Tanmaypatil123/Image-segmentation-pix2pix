import tensorflow as tf
from model import Generator , Discriminator
from data import Datagenerator
import os

data_directory = "./dataset/cityscapes_data/"

train_data = Datagenerator(os.path.join(data_directory,'train'))
test_data = Datagenerator(os.path.join(data_directory,'test'))

generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

generator = Generator()
discriminator = Discriminator()

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)