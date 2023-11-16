import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from keras.models import Model
from sklearn.metrics import mean_squared_error as mse
from progressbar import ProgressBar
from image_process import ImageProcess
import const, cv2
from tensorflow.image import psnr, ssim
from drive_data import DriveData
from config import Config

import matplotlib.pyplot as plt
import sys
config = Config.neural_net
sys.setrecursionlimit(10**7)
tf.compat.v1.experimental.output_all_intermediates(True)
tf.compat.v1.disable_eager_execution()

# tf.random.set_seed(3)
# -----

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
gpu_config = tf.compat.v1.ConfigProto()
gpu_config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=gpu_config)
tf.compat.v1.keras.backend.set_session(sess)

# -----

batch_size = 8
"""
## Encoder
"""

"""
## Create a sampling layer
"""
class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

encoder_inputs_img = tf.keras.Input(batch_shape=(batch_size*2, 1, 160, 160, 3))
encoder_inputs_str = tf.keras.Input(batch_shape=(batch_size*2, 1, 1))
encoder_inputs_vel = tf.keras.Input(batch_shape=(batch_size*2, 1, 1))
encoder_inputs_time = tf.keras.Input(batch_shape=(batch_size*2, 1, 1))
x = layers.TimeDistributed(layers.Conv2D(24, (5, 5), padding='same', name='conv2d_1'))(encoder_inputs_img)
x = layers.TimeDistributed(layers.BatchNormalization())(x)
x = layers.TimeDistributed(layers.Activation('elu'))(x)
x = layers.TimeDistributed(layers.MaxPooling2D(pool_size=(2, 2), name='pool2d_1'))(x)
x = layers.TimeDistributed(layers.Conv2D(36, (5, 5), padding='same', name='conv2d_2'))(x)
x = layers.TimeDistributed(layers.BatchNormalization())(x)
x = layers.TimeDistributed(layers.Activation('elu'))(x)
x = layers.TimeDistributed(layers.MaxPooling2D(pool_size=(2, 2), name='pool2d_2'))(x)
x = layers.TimeDistributed(layers.Conv2D(48, (5, 5), padding='same', name='conv2d_3'))(x)
x = layers.TimeDistributed(layers.BatchNormalization())(x)
x = layers.TimeDistributed(layers.Activation('elu'))(x)
x = layers.TimeDistributed(layers.Conv2D(64, (3, 3), padding='same', name='conv2d_4'))(x)
x = layers.TimeDistributed(layers.BatchNormalization())(x)
x = layers.TimeDistributed(layers.Activation('elu'))(x)
x = layers.TimeDistributed(layers.Conv2D(64, (3, 3), padding='same', name='conv2d_5'))(x)
x = layers.TimeDistributed(layers.BatchNormalization())(x)
x = layers.TimeDistributed(layers.Activation('elu'))(x)
latent = layers.TimeDistributed(layers.Flatten())(x)
latent = layers.TimeDistributed(layers.Dense(500))(latent)
latent = layers.TimeDistributed(layers.LayerNormalization())(latent)
latent = layers.TimeDistributed(layers.Activation('tanh'))(latent)
fc_s1  = layers.TimeDistributed(layers.Dense(100))(encoder_inputs_str)
fc_s1  = layers.TimeDistributed(layers.BatchNormalization())(fc_s1)
fc_s1  = layers.TimeDistributed(layers.Activation('elu'))(fc_s1)
fc_s2  = layers.TimeDistributed(layers.Dense(50))(fc_s1)
fc_s2  = layers.TimeDistributed(layers.BatchNormalization())(fc_s2)
fc_s2  = layers.TimeDistributed(layers.Activation('elu'))(fc_s2)
fc_v1  = layers.TimeDistributed(layers.Dense(100))(encoder_inputs_vel)
fc_v1  = layers.TimeDistributed(layers.BatchNormalization())(fc_v1)
fc_v1  = layers.TimeDistributed(layers.Activation('elu'))(fc_v1)
fc_v2  = layers.TimeDistributed(layers.Dense(50))(fc_v1)
fc_v2  = layers.TimeDistributed(layers.BatchNormalization())(fc_v2)
fc_v2  = layers.TimeDistributed(layers.Activation('elu'))(fc_v2)
fc_t1  = layers.TimeDistributed(layers.Dense(100))(encoder_inputs_time)
fc_t1  = layers.TimeDistributed(layers.BatchNormalization())(fc_t1)
fc_t1  = layers.TimeDistributed(layers.Activation('elu'))(fc_t1)
fc_t2  = layers.TimeDistributed(layers.Dense(50))(fc_t1)
fc_t2  = layers.TimeDistributed(layers.BatchNormalization())(fc_t2)
fc_t2  = layers.TimeDistributed(layers.Activation('elu'))(fc_t2)
conc_1 = layers.concatenate([latent, fc_s2, fc_v2, fc_t2])
bilstm = layers.Bidirectional(layers.LSTM(500, batch_size=batch_size*2, stateful=True))(conc_1)
# bilstm = layers.Bidirectional(layers.LSTM(500, input_shape=(1000, 1), batch_size=10, stateful=True))(conc_1)
fc_1   = layers.Dense(500)(bilstm)
fc_1   = layers.Activation('elu')(fc_1)
fc_2   = layers.Dense(100)(fc_1)
x   = layers.Activation('elu')(fc_2)
z_mean = layers.Dense(50, name="z_mean")(x)
z_log_var = layers.Dense(50, name="z_log_var")(x)
encoder_output = Sampling()([z_mean, z_log_var])
encoder = tf.keras.Model([ encoder_inputs_img, encoder_inputs_str,
                        encoder_inputs_vel, encoder_inputs_time], 
                        [z_mean, z_log_var, encoder_output], 
                        name="encoder")

"""
## Decoder
"""
latent_inputs = tf.keras.Input(shape=(50,))
x = layers.Dense(40 * 40 * 64, activation="elu")(latent_inputs)
x = layers.Reshape((40, 40, 64))(x)
x = layers.Conv2DTranspose(64, 3, activation="elu",  padding="same")(x)
x = layers.Conv2DTranspose(48, 3, activation="elu", strides=2, padding="same")(x)
x = layers.Conv2DTranspose(36, 3, activation="elu", strides=2, padding="same")(x)
decoder_outputs = layers.Conv2DTranspose(3, 3, activation="sigmoid", padding="same")(x)
decoder = tf.keras.Model(latent_inputs, decoder_outputs, name="decoder")

"""
## VAE
"""
model_input = [ encoder_inputs_img, encoder_inputs_str,
                        encoder_inputs_vel, encoder_inputs_time]
model_output = decoder(encoder_output)

Fwd=Model(model_input, model_output)
Fwd.summary()


# -----

# load weight
pretrained_path = '/home2/kdh/vae/new_dataset/vqvae2/vae_b64_mse_ckpt.1268-1218.8559.h5'
Fwd.load_weights(pretrained_path)
# -----

"""
## Dataset
"""
image_process = ImageProcess()
config = Config.neural_net
from multiprocessing import Pool, cpu_count
def process_row(row):
    image_name = row['image_fname']
    measurements = (float(row['steering_angle']), float(row['throttle']), float(row['brake']))
    velocity = float(row['vel'])
    tar_image_name = row['tar_image_fname']
    tar_steering_angle = float(row['tar_steering_angle'])
    tar_vel = float(row['tar_vel'])
    tar_time = float(row['tar_time'])
    return (image_name, measurements, velocity, tar_image_name, tar_steering_angle, tar_vel, tar_time)

def data_generator(data_path):

    if data_path[-1] == '/':
        data_path = data_path[:-1]

    loc_slash = data_path.rfind('/')
    if loc_slash != -1:  # there is '/' in the data path
        model_name = data_path[loc_slash + 1:]  # get folder name
    else:
        model_name = data_path

    csv_path = data_path + '/' + model_name + const.DATA_EXT 

    csv_header = [
        'image_fname', 'steering_angle', 'throttle', 'brake', 'linux_time', 
        'vel', 'vel_x', 'vel_y', 'vel_z', 'pos_x', 'pos_y', 'pos_z', 
        'tar_image_fname', 'tar_steering_angle', 'tar_vel', 'tar_time'
    ]

    df = pd.read_csv(csv_path, names=csv_header, index_col=False)
    num_data = len(df)

    # Create a Pool with the number of available CPU cores
    with Pool(cpu_count()) as p:
        results = p.map(process_row, [df.loc[i] for i in range(num_data)])

    # Separate the results into separate lists
    df_image_names, df_measurements, df_velocities, df_tar_image_names, df_tar_steering_angle, df_tar_vel, df_tar_time = zip(*results)

    # -----
    samples = list(zip(df_image_names, df_velocities, df_measurements, 
                       df_tar_image_names, df_tar_steering_angle, df_tar_vel, df_tar_time))
    
    return samples

traindata_path = "/home2/kdh/vae/new_dataset/2023-08-22-17-26-02/"
train_data = data_generator(traindata_path)

def flipping(img, tar_img, steering, tar_steering):
    flip_image = cv2.flip(img,1)
    flip_tar_image = cv2.flip(tar_img,1)
    flip_steering = steering*-1.0
    flip_tar_steering = tar_steering*-1.0
    return flip_image, flip_tar_image, flip_steering, flip_tar_steering

# -----

def prepare_vae_batch_samples(batch_samples, data_path):
    images = []
    steering_angles = []
    vels = []
    tar_images = []
    tar_steering_angles = []
    tar_vels = []
    tar_times = []


    batch_images = []
    batch_steering_angles = []
    batch_vels = []
    batch_tar_images = []
    batch_tar_steering_angles = []
    batch_tar_vels = []
    batch_tar_times = []
        
    for image_name, velocity, measurement, tar_image_name, tar_steering_angle, tar_vel, tar_time in batch_samples:
        # self.data.image_names, self.data.velocities, self.data.measurements, 
        # self.data.tar_image_names, self.data.tar_steering_angle, self.data.tar_vel, self.data.tar_time
        
        image_path = data_path + '/' + image_name
        # print(data_path, tar_image_name)
        tar_image_path = data_path + '/' + tar_image_name
        image = cv2.imread(image_path)
        tar_image = cv2.imread(tar_image_path)

        # if collected data is not cropped then crop here
        # otherwise do not crop.
        if Config.data_collection['crop'] is not True:
            image = image[Config.data_collection['image_crop_y1']:Config.data_collection['image_crop_y2'],
                        Config.data_collection['image_crop_x1']:Config.data_collection['image_crop_x2']]
            tar_image = tar_image[Config.data_collection['image_crop_y1']:Config.data_collection['image_crop_y2'],
                        Config.data_collection['image_crop_x1']:Config.data_collection['image_crop_x2']]
        image = cv2.resize(image, 
                            (config['input_image_width'],
                            config['input_image_height']))
        tar_image = cv2.resize(tar_image, 
                            (config['input_image_width'],
                            config['input_image_height']))
        image = image_process.process(image)
        tar_image = image_process.process(tar_image)

        batch_vels.append(velocity)
        batch_tar_vels.append(tar_vel)
        batch_tar_times.append(tar_time)
        batch_tar_steering_angles.append(tar_steering_angle)
        # if no brake data in collected data, brake values are dummy
        steering_angle, throttle, brake = measurement
        batch_steering_angles.append(steering_angle)
        batch_images.append(image)
        tar_images.append(tar_image)

        #### flip aug
        flip_image, flip_tar_image, flip_steering_angle, flip_tar_steering_angle = flipping(image, tar_image, steering_angle, tar_steering_angle)
        batch_vels.append(velocity)
        batch_tar_vels.append(tar_vel)
        batch_tar_times.append(tar_time)
        batch_tar_steering_angles.append(flip_tar_steering_angle)
        batch_steering_angles.append(flip_steering_angle)
        batch_images.append(flip_image)
        tar_images.append(flip_tar_image)


    images.append(batch_images)
    steering_angles.append(batch_steering_angles)
    vels.append(batch_vels)
    tar_steering_angles.append(batch_tar_steering_angles)
    tar_vels.append(batch_tar_vels)
    tar_times.append(batch_tar_times)
    return images, vels, steering_angles, tar_images, tar_steering_angles, tar_vels, tar_times


# -----

"""
Data generator
"""
from sklearn.model_selection import train_test_split

def _generator(samples, batch_size=10, data_path=None):
    num_samples = len(samples)
    while True: # Loop forever so the generator never terminates
        for offset in range(0, (num_samples//batch_size)*batch_size, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images, vels, steering_angles, tar_images, tar_steering_angles, tar_vels, tar_times = prepare_vae_batch_samples(batch_samples, data_path)

            X_img = np.array(images).astype("float32")/255.0
            X_tvel = np.array(tar_vels).astype("float32")
            X_tstr = np.array(tar_steering_angles).astype("float32")
            X_ttime = np.array(tar_times).astype("float32")

            X_img = X_img.swapaxes(0,1)
            
            X_tvel = X_tvel.swapaxes(0,1)
            X_tstr = X_tstr.swapaxes(0,1)
            X_ttime = X_ttime.swapaxes(0,1)
            X_tvel = X_tvel.reshape(-1, 1, 1)
            X_tstr = X_tstr.reshape(-1, 1, 1)
            X_ttime = X_ttime.reshape(-1, 1, 1)
            y_train = np.array(tar_images).astype("float32")/255.0
            X_train = [X_img, X_tstr, X_tvel, X_ttime]
            # print(y_train.shape)
            yield X_train, y_train

train_data, valid_data = train_test_split(train_data, test_size=0.2)
# train_data = train_samples
# valid_data = valid_samples
# train_data, valid_data = train_test_split(samples, test_size=0.5)
# train_data = samples
# _, valid_data = train_test_split(samples, test_size=0.2, shuffle=True)
train_generator = _generator(train_data, batch_size, traindata_path)
valid_generator = _generator(valid_data, batch_size, traindata_path)
# valid_generator = _generator(valid_data, batch_size, validdata_path)


# -----

"""
VAE loss
"""
# optimizer=tf.keras.optimizers.Adam(learning_rate=0.00005)
optimizer=tf.keras.optimizers.Adam(learning_rate=0.00005)
#                                  learning_rate=0.000005,decay=0.0000000001
r_loss_factor=160*160*100   # This is a Hyperparameter

def vae_r_loss(y_true, y_pred):    ## MSE
    r_loss = tf.keras.backend.mean(tf.keras.backend.square(y_true-y_pred), axis=[1,2,3])
    
    return r_loss_factor * r_loss

def vae_kl_loss(y_true, y_pred):   ## KL-Divergence
    kl_loss=( -0.5 * tf.keras.backend.sum(1+z_log_var 
                - tf.keras.backend.square(z_mean) 
                - tf.keras.backend.exp(z_log_var), axis=1)
    )
    return kl_loss

def vae_loss(y_true, y_pred): 
    r_loss=vae_r_loss(y_true, y_pred) #Loss of Decoder
    kl_loss = vae_kl_loss(y_true, y_pred) #Loss of Encoder
    return r_loss + kl_loss #Sum of these two


Fwd.compile(optimizer=optimizer,
                loss= vae_loss, 
                metrics=[vae_r_loss, vae_kl_loss])

# -----


"""
## Train the VAE
"""
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

callbacks = []
model_ckpt_name = "/home2/kdh/vae/new_dataset/vae/vae_b64_mse_load_timeadd"
checkpoint = ModelCheckpoint(model_ckpt_name +'_ckpt.{epoch:02d}-{val_loss:.4f}.h5',
                                monitor='val_loss', 
                                verbose=1, save_best_only=True, mode='min')


# Early stopping
patience = 10000
earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=patience, 
                            verbose=1, mode='min')

callbacks = [checkpoint, earlystop]

train_hist = Fwd.fit(train_generator,  
                        steps_per_epoch=len(train_data)//batch_size, 
                        epochs=1000000, 
                        validation_data=valid_generator,
                        validation_steps=len(valid_data)//batch_size,
                        verbose=1, 
                        callbacks=callbacks, 
                        use_multiprocessing=True,
                        workers=48
                        )