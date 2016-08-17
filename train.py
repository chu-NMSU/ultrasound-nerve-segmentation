from __future__ import print_function

import numpy as np
import scipy.misc
from keras.models import Model
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D, Dropout
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from data import load_train_data, load_test_data
import sys
from submission import submission

# original size
# image_rows = 420
# image_cols = 580
img_rows = 64
img_cols = 80
# img_rows = 128
# img_cols = 160
smooth = 1.

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def preprocess(imgs):
    '''shrink image size'''
    imgs_p = np.ndarray((imgs.shape[0], imgs.shape[1], img_rows, img_cols), dtype=np.uint8)
    for i in range(imgs.shape[0]):
        imgs_p[i, 0] = scipy.misc.imresize(imgs[i, 0], size=(img_rows, img_cols), interp='bicubic')
    return imgs_p

def get_unet_drop(conv_size, pool_size, drop_ratio):
    '''add dropout layer to each convolution layer'''
    inputs = Input((1, img_rows, img_cols))
    conv1 = Convolution2D(32, conv_size, conv_size, activation='relu', border_mode='same')(inputs)
    conv1 = Convolution2D(32, conv_size, conv_size, activation='relu', border_mode='same')(conv1)
    conv1 = Dropout(drop_ratio)(conv1)
    pool1 = MaxPooling2D(pool_size=(pool_size,pool_size))(conv1)

    conv2 = Convolution2D(64, conv_size, conv_size, activation='relu', border_mode='same')(pool1)
    conv2 = Convolution2D(64, conv_size, conv_size, activation='relu', border_mode='same')(conv2)
    conv2 = Dropout(drop_ratio)(conv2)
    pool2 = MaxPooling2D(pool_size=(pool_size,pool_size))(conv2)

    conv3 = Convolution2D(128, conv_size, conv_size, activation='relu', border_mode='same')(pool2)
    conv3 = Convolution2D(128, conv_size, conv_size, activation='relu', border_mode='same')(conv3)
    conv3 = Dropout(drop_ratio)(conv3)
    pool3 = MaxPooling2D(pool_size=(pool_size,pool_size))(conv3)

    conv4 = Convolution2D(256, conv_size, conv_size, activation='relu', border_mode='same')(pool3)
    conv4 = Convolution2D(256, conv_size, conv_size, activation='relu', border_mode='same')(conv4)
    conv4 = Dropout(drop_ratio)(conv4)
    pool4 = MaxPooling2D(pool_size=(pool_size,pool_size))(conv4)

    conv5 = Convolution2D(512, conv_size, conv_size, activation='relu', border_mode='same')(pool4)
    conv5 = Convolution2D(512, conv_size, conv_size, activation='relu', border_mode='same')(conv5)
    conv5 = Dropout(drop_ratio)(conv5)

    up6 = merge([UpSampling2D(size=(pool_size,pool_size))(conv5), conv4], mode='concat', concat_axis=1)
    conv6 = Convolution2D(256, conv_size, conv_size, activation='relu', border_mode='same')(up6)
    conv6 = Convolution2D(256, conv_size, conv_size, activation='relu', border_mode='same')(conv6)
    conv6 = Dropout(drop_ratio)(conv6)

    up7 = merge([UpSampling2D(size=(pool_size,pool_size))(conv6), conv3], mode='concat', concat_axis=1)
    conv7 = Convolution2D(128, conv_size, conv_size, activation='relu', border_mode='same')(up7)
    conv7 = Convolution2D(128, conv_size, conv_size, activation='relu', border_mode='same')(conv7)
    conv7 = Dropout(drop_ratio)(conv7)

    up8 = merge([UpSampling2D(size=(pool_size,pool_size))(conv7), conv2], mode='concat', concat_axis=1)
    conv8 = Convolution2D(64, conv_size, conv_size, activation='relu', border_mode='same')(up8)
    conv8 = Convolution2D(64, conv_size, conv_size, activation='relu', border_mode='same')(conv8)
    conv8 = Dropout(drop_ratio)(conv8)

    up9 = merge([UpSampling2D(size=(pool_size,pool_size))(conv8), conv1], mode='concat', concat_axis=1)
    conv9 = Convolution2D(32, conv_size, conv_size, activation='relu', border_mode='same')(up9)
    conv9 = Convolution2D(32, conv_size, conv_size, activation='relu', border_mode='same')(conv9)
    conv9 = Dropout(drop_ratio)(conv9)

    conv10 = Convolution2D(1, 1, 1, activation='sigmoid')(conv9)
    # conv10 = Dropout(drop_ratio)(conv10)

    model = Model(input=inputs, output=conv10)
    model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])

    return model

def get_unet(conv_size, pool_size):
    inputs = Input((1, img_rows, img_cols))
    conv1 = Convolution2D(32, conv_size, conv_size, activation='relu', border_mode='same')(inputs)
    conv1 = Convolution2D(32, conv_size, conv_size, activation='relu', border_mode='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(pool_size,pool_size))(conv1)

    conv2 = Convolution2D(64, conv_size, conv_size, activation='relu', border_mode='same')(pool1)
    conv2 = Convolution2D(64, conv_size, conv_size, activation='relu', border_mode='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(pool_size,pool_size))(conv2)

    conv3 = Convolution2D(128, conv_size, conv_size, activation='relu', border_mode='same')(pool2)
    conv3 = Convolution2D(128, conv_size, conv_size, activation='relu', border_mode='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(pool_size,pool_size))(conv3)

    conv4 = Convolution2D(256, conv_size, conv_size, activation='relu', border_mode='same')(pool3)
    conv4 = Convolution2D(256, conv_size, conv_size, activation='relu', border_mode='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(pool_size,pool_size))(conv4)

    conv5 = Convolution2D(512, conv_size, conv_size, activation='relu', border_mode='same')(pool4)
    conv5 = Convolution2D(512, conv_size, conv_size, activation='relu', border_mode='same')(conv5)

    up6 = merge([UpSampling2D(size=(pool_size,pool_size))(conv5), conv4], mode='concat', concat_axis=1)
    conv6 = Convolution2D(256, conv_size, conv_size, activation='relu', border_mode='same')(up6)
    conv6 = Convolution2D(256, conv_size, conv_size, activation='relu', border_mode='same')(conv6)

    up7 = merge([UpSampling2D(size=(pool_size,pool_size))(conv6), conv3], mode='concat', concat_axis=1)
    conv7 = Convolution2D(128, conv_size, conv_size, activation='relu', border_mode='same')(up7)
    conv7 = Convolution2D(128, conv_size, conv_size, activation='relu', border_mode='same')(conv7)

    up8 = merge([UpSampling2D(size=(pool_size,pool_size))(conv7), conv2], mode='concat', concat_axis=1)
    conv8 = Convolution2D(64, conv_size, conv_size, activation='relu', border_mode='same')(up8)
    conv8 = Convolution2D(64, conv_size, conv_size, activation='relu', border_mode='same')(conv8)

    up9 = merge([UpSampling2D(size=(pool_size,pool_size))(conv8), conv1], mode='concat', concat_axis=1)
    conv9 = Convolution2D(32, conv_size, conv_size, activation='relu', border_mode='same')(up9)
    conv9 = Convolution2D(32, conv_size, conv_size, activation='relu', border_mode='same')(conv9)

    conv10 = Convolution2D(1, 1, 1, activation='sigmoid')(conv9)

    model = Model(input=inputs, output=conv10)
    model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])

    return model

def train_and_predict(conv_size, pool_size, epoch=20, drop_ratio=-1):
    print('-'*30)
    print('Loading and preprocessing train data...')
    print('-'*30)
    imgs_train, imgs_mask_train = load_train_data()

    imgs_train = preprocess(imgs_train)
    imgs_mask_train = preprocess(imgs_mask_train)

    imgs_train = imgs_train.astype('float32')
    mean = np.mean(imgs_train)  # mean for data centering
    std = np.std(imgs_train)  # std for data normalization

    imgs_train -= mean
    imgs_train /= std

    imgs_mask_train = imgs_mask_train.astype('float32')
    imgs_mask_train /= 255.  # scale masks to [0, 1]

    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)

    if drop_ratio==-1:
        model = get_unet(conv_size, pool_size)
    else:
        model = get_unet_drop(conv_size, pool_size, drop_ratio)

    checkpoint_path = 'unet-'+str(img_rows)+'-'+str(img_cols)+'-'+str(conv_size)+'-'+str(pool_size)
    if drop_ratio!=-1:
        checkpoint_path += '-'+str(drop_ratio)
    checkpoint_path += '.hdf5'
    model_checkpoint = ModelCheckpoint(checkpoint_path, monitor='loss', save_best_only=True)

    print('-'*30)
    print('Fitting model...')
    print('-'*30)
    model.fit(imgs_train, imgs_mask_train, batch_size=32, nb_epoch=epoch, verbose=1, shuffle=True,\
            callbacks=[model_checkpoint])

    print('-'*30)
    print('Loading and preprocessing test data...')
    print('-'*30)
    imgs_test, imgs_id_test = load_test_data()
    imgs_test = preprocess(imgs_test)

    imgs_test = imgs_test.astype('float32')
    imgs_test -= mean
    imgs_test /= std

    print('-'*30)
    print('Loading saved weights...')
    print('-'*30)
    model.load_weights(checkpoint_path)

    print('-'*30)
    print('Predicting masks on test data...')
    print('-'*30)
    imgs_mask_test = model.predict(imgs_test, verbose=1)
    output = 'output/Unet-N-'+str(img_rows)+'-M-'+str(img_cols)+'-conv_size-'+str(conv_size)+'-pool_size-'+str(pool_size)
    if drop_ratio!=-1:
        output += '-drop_ratio-'+str(drop_ratio)
    output += '.npy'
    np.save(output, imgs_mask_test)
    submission(output)

if __name__ == '__main__':
    img_rows = int(sys.argv[1])
    img_cols = int(sys.argv[2])
    conv_size = int(sys.argv[3])
    pool_size = int(sys.argv[4])
    epoch = int(sys.argv[5])
    drop_ratio = -1
    if len(sys.argv)>=7:
        drop_ratio = float(sys.argv[6])
    train_and_predict(conv_size, pool_size, epoch, drop_ratio)
