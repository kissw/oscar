import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from image_process import ImageProcess
import const, cv2
from config import Config

import sys
config = Config.neural_net
sys.setrecursionlimit(10**7)
tf.compat.v1.experimental.output_all_intermediates(True)
tf.compat.v1.disable_eager_execution()

import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
gpu_config = tf.compat.v1.ConfigProto()
gpu_config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=gpu_config)
tf.compat.v1.keras.backend.set_session(sess)

batch_size = 6

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

class PerspectiveTransformLayer(layers.Layer):
    def __init__(self, **kwargs):
        super(PerspectiveTransformLayer, self).__init__(**kwargs)
    
    def call(self, inputs, v, t):
        # 여기에 원근 변환 로직 구현
        # ...
        transformed_output=0
        return transformed_output
    
img_shape = (batch_size, 1, 160, 160, 3,)
str_shape = (batch_size, 1, 1)
vt_shape  = (batch_size, 1, 1)
latent_dim = 50
# Encoder
img_input = layers.Input(batch_shape=img_shape)  # 시각적 입력
str_input = layers.Input(batch_shape=str_shape)
vt_input = layers.Input(batch_shape=vt_shape)

str_processed = layers.TimeDistributed(layers.Dense(100))(str_input)
vt_processed  = layers.TimeDistributed(layers.Dense(100))(vt_input)

x = layers.ConvLSTM2D(filters=32, kernel_size=(5, 5),strides=2, padding='same', return_sequences=True)(img_input)
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)
x = layers.ConvLSTM2D(filters=64, kernel_size=(5, 5),strides=2, padding='same', return_sequences=True)(x)
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)
x = layers.ConvLSTM2D(filters=128, kernel_size=(3, 3), padding='same', return_sequences=True)(x)
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)
x = layers.ConvLSTM2D(filters=128, kernel_size=(3, 3), padding='same', return_sequences=True)(x)
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)
x = layers.ConvLSTM2D(filters=128, kernel_size=(3, 3), padding='same', return_sequences=True)(x)
x = layers.TimeDistributed(layers.GlobalAveragePooling2D())(x)

x = layers.concatenate([x, str_processed, vt_processed])
x = layers.Bidirectional(layers.LSTM(500, batch_size=batch_size, stateful=True))(x)
z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
z = Sampling()([z_mean, z_log_var])

# Decoder
decoder_input = layers.Input(shape=(latent_dim,))
x = layers.Dense(units=128 * 40 * 40, activation='relu')(decoder_input)
x = layers.Reshape((40, 40, 128))(x)
x = layers.Conv2DTranspose(filters=128, kernel_size=(3, 3), padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.Conv2DTranspose(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
decoder_outputs = layers.Conv2DTranspose(3, kernel_size=(3, 3), padding='same', activation='sigmoid')(x)

# VAE 모델
encoder = Model([img_input, str_input, vt_input], [z_mean, z_log_var, z], name="encoder")
decoder = Model(decoder_input, decoder_outputs, name="decoder")
vae_outputs = decoder(encoder([img_input, str_input, vt_input])[2])
vae = Model([img_input, str_input, vt_input], vae_outputs, name="vae")
vae.summary()


# -----

# load weight
pretrained_path = '/home2/kdh/vae/new_dataset/vae/vae_convlstm_b6_ckpt.01-14749.6760.h5'
vae.load_weights(pretrained_path)
# -----


"""
## Dataset
"""
image_process = ImageProcess()
config = Config.neural_net
from multiprocessing import Pool, cpu_count
def process_row(row):
    image_name = row['image_fname']
    tar_image_name = row['tar_image_fname']
    tar_steering_angle = float(row['tar_steering_angle'])
    tar_vel = float(row['tar_vel'])
    tar_time = float(row['tar_time'])
    return (image_name, tar_image_name, tar_steering_angle, tar_vel, tar_time)

def data_generator(data_path):
    if 'csv' in data_path[-4:]:
        csv_path = data_path
    else:
        if data_path[-1] == '/':
            data_path = data_path[:-1]

        loc_slash = data_path.rfind('/')
        if loc_slash != -1:  # there is '/' in the data path
            model_name = data_path[loc_slash + 1:]  # get folder name
        else:
            model_name = data_path

        csv_path = data_path + '/' + model_name + const.DATA_EXT 

    csv_header = [
        'image_num','image_fname', 'tar_image_fname', 'tar_steering_angle', 'tar_vel', 'tar_time'
    ]

    df = pd.read_csv(csv_path, names=csv_header, index_col=False)
    num_data = len(df)
    print(num_data)

    # Create a Pool with the number of available CPU cores
    with Pool(cpu_count()) as p:
        results = p.map(process_row, [df.loc[i] for i in range(num_data)])

    # Separate the results into separate lists
    df_image_names, df_tar_image_names, df_tar_steering_angle, df_tar_vel, df_tar_time = zip(*results)

    # -----
    samples = list(zip(df_image_names, df_tar_image_names, df_tar_steering_angle, df_tar_vel, df_tar_time))
    
    return samples

traindata_path = "/home2/kdh/vae/new_dataset/2023-08-22-17-26-04/2023-08-22-17-26-04_train.csv"
train_data = data_generator(traindata_path)
validdata_path = "/home2/kdh/vae/new_dataset/2023-08-22-17-26-04/2023-08-22-17-26-04_test.csv"
valid_data = data_generator(validdata_path)

def prepare_vae_batch_samples(batch_samples, data_path):
    images = []
    tar_images = []
    tar_steering_angles = []
    tar_vels = []
    tar_times = []
        
    for image_name, tar_image_name, tar_steering_angle, tar_vel, tar_time in batch_samples:
        
        image_path = data_path + '/' + image_name
        tar_image_path = data_path + '/' + tar_image_name
        image = cv2.imread(image_path)
        tar_image = cv2.imread(tar_image_path)

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

        tar_images.append(tar_image)
        images.append(image)
        tar_steering_angles.append(tar_steering_angle)
        tar_vels.append(tar_vel)
        tar_times.append(tar_time)
    return images, tar_images, tar_steering_angles, tar_vels, tar_times
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

            images, tar_images, tar_steering_angles, tar_vels, tar_times = prepare_vae_batch_samples(batch_samples, data_path)

            X_img = np.array(images).astype("float32")/255.0
            X_tvel = np.array(tar_vels).astype("float32")
            X_ttime = np.array(tar_times).astype("float32")
            X_tstr = np.array(tar_steering_angles).astype("float32")

            X_img = np.expand_dims(X_img,axis=1)
            X_tvel = np.expand_dims(X_tvel,axis=1)
            X_ttime = np.expand_dims(X_ttime,axis=1)
            X_tstr = np.expand_dims(X_tstr,axis=1)
            # X_img = X_img.swapaxes(0,1)
            X_tvel = X_tvel.reshape(-1,1, 1)
            X_ttime = X_ttime.reshape(-1,1, 1)
            X_tstr = X_tstr.reshape(-1,1, 1)
            X_train = [X_img, X_tstr, X_tvel*X_ttime]
            y_train = np.array(tar_images).astype("float32")/255.0
            # print(y_train.shape)
            yield X_train, y_train

train_generator = _generator(train_data, batch_size, '/home2/kdh/vae/new_dataset/2023-08-22-17-26-04')
valid_generator = _generator(valid_data, batch_size, '/home2/kdh/vae/new_dataset/2023-08-22-17-26-04')




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


vae.compile(optimizer=optimizer,
                loss= vae_loss, 
                metrics=[vae_r_loss, vae_kl_loss])


"""
## Train the VAE
"""
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

callbacks = []
model_ckpt_name = "/home2/kdh/vae/new_dataset/vae/vae_convlstm_b6"
checkpoint = ModelCheckpoint(model_ckpt_name +'_ckpt.{epoch:02d}-{val_loss:.4f}.h5',
                                monitor='val_loss', 
                                verbose=1, save_best_only=True, mode='min')


# Early stopping
patience = 10000
earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=patience, 
                            verbose=1, mode='min')

callbacks = [checkpoint, earlystop]

train_hist = vae.fit(train_generator,  
                        steps_per_epoch=len(train_data)//batch_size, 
                        epochs=1000000, 
                        validation_data=valid_generator,
                        validation_steps=len(valid_data)//batch_size,
                        verbose=1, 
                        callbacks=callbacks, 
                        use_multiprocessing=True,
                        workers=48
                        )