from variables import *
from utils.loss import *
import tensorflow as tf
import time


## Custom training loop 
@tf.function
def train_step(input_image, target, step):
    
    
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image , training = True)
        disc_real_output = discriminator([input_image , target],training = True)
        disc_gen_output = discriminator([input_image , gen_output],training = True)
        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_gen_output, gen_output, target)
        disc_loss = discriminator_loss(disc_real_output, disc_gen_output)
    
        
    generator_gradients = gen_tape.gradient(gen_total_loss,
                                      generator.trainable_weights)
    discriminator_gradients = disc_tape.gradient(disc_loss,
                                           discriminator.trainable_weights)
    generator_optimizer.apply_gradients(zip(generator_gradients,
                                      generator.trainable_weights))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                          discriminator.trainable_weights))
    

def fit(train_ds , test_ds , steps):
    example_input , example_target = test_ds[1]
    start = time.time()
    for step , (input_image , target) in enumerate(train_ds):
        if(step) % 1000 == 0 :
            if step != 0 :
                print(f"Time taken for 100 steps : {time.time() - start:.2f} sec\n")
            start = time.time()
            print(f"Step : {step // 1000}k")
        train_step(input_image , target , step)
        if (step + 1) % 10 == 0 :
            print(".",end = '',flush = True)
        if (step + 1) % 5000 == 0 :
            checkpoint.save(file_prefix= checkpoint_prefix)

if __name__ == "__main__":
    fit(train_data,test_data,40000)