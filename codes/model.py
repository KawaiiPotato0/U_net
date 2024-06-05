from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input, MaxPooling2D, Flatten, Dense, UpSampling2D
from tensorflow.keras.models import Model, Sequential
import numpy as np
from layers import MaxUnpooling2D, MaxPoolingWithArgmax2D

def conv_block(inputs, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x

def encoder_block(inputs, num_filters):
    x = conv_block(inputs, num_filters)
    p = MaxPool2D((2, 2))(x)
    return x, p

def decoder_block(inputs, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, 2, strides=2, padding="same")(inputs)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x

def unet(input_shape):
    inputs = Input(input_shape)

    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)

    # print(s1.shape, s2.shape, s3.shape, s4.shape)
    # print(p1.shape, p2.shape, p3.shape, p4.shape)

    b1 = conv_block(p4, 1024)

    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    outputs = Conv2D(2, 1, padding="same", activation="softmax")(d4)

    model = Model(inputs, outputs, name="UNET")
    return model

def unet_multiclass(input_shape, num_classes):
    inputs = Input(input_shape)

    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)

    b1 = conv_block(p4, 1024)

    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    # Modify the output layer to predict num_classes channels with softmax activation
    outputs = Conv2D(num_classes, 1, padding="same", activation="softmax")(d4)

    model = Model(inputs, outputs, name="UNET")
    return model

def segnet(input_shape, num_classes):
    inputs = Input(shape=input_shape)

    # Encoder
    conv1 = Conv2D(64, (3, 3), padding="same")(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation("relu")(conv1)

    pool1 = MaxPool2D((2, 2))(conv1)

    conv2 =  Conv2D(128, (3, 3), padding="same")(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation("relu")(conv2)

    pool2 = MaxPool2D((2, 2))(conv2)

    conv3 = Conv2D(256, (3, 3), padding="same")(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation("relu")(conv3)

    pool3 = MaxPool2D((2, 2))(conv3)

    conv4 = Conv2D(512, (3, 3), padding="same")(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation("relu")(conv4)

    pool4 = MaxPool2D((2, 2))(conv4)

    conv5 = Conv2D(1024, (3, 3), padding="same")(pool4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation("relu")(conv5)

    # Bottleneck
    bpool = MaxPool2D((2, 2))(conv5)
    bup = UpSampling2D((2, 2))(bpool)

    # Decoder
    conv6 = Conv2D(1024, (3, 3), padding="same")(bup)
    conv6 = BatchNormalization()(conv6)
    conv6 = Activation("relu")(conv6)

    up6 = UpSampling2D(size=(2, 2))(conv6)

    conv7 = Conv2D(512, (3, 3), padding="same")(up6)
    conv7 = BatchNormalization()(conv7)
    conv7 = Activation("relu")(conv7)

    up7 = UpSampling2D(size=(2, 2))(conv7)

    conv8 = Conv2D(256, (3, 3), padding="same")(up7)
    conv8 = BatchNormalization()(conv8)
    conv8 = Activation("relu")(conv8)

    up8 = UpSampling2D(size=(2, 2))(conv8)
    
    conv9 = Conv2D(128, (3, 3), padding="same")(up8)
    conv9 = BatchNormalization()(conv9)
    conv9 = Activation("relu")(conv9)

    up9 = UpSampling2D(size=(2, 2))(conv9)

    conv10 = Conv2D(64, (3, 3), padding="same")(up9)
    conv10 = BatchNormalization()(conv10)

    # Output
    output = Conv2D(num_classes, 1, padding="same", activation="softmax")(conv10)

    model = Model(inputs=inputs, outputs=output)

    return model

def segnet_ind(input_shape, num_classes):
    inputs = Input(shape=input_shape)

    conv1 = Conv2D(64, (3, 3), padding="same")(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation("relu")(conv1)

    pool1, mask1 = MaxPoolingWithArgmax2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, (3, 3), padding="same", kernel_initializer='he_normal')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation("relu")(conv2)

    pool2, mask2 = MaxPoolingWithArgmax2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, (3, 3), padding="same", kernel_initializer='he_normal')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation("relu")(conv3)

    pool3, mask3 = MaxPoolingWithArgmax2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, (3, 3), padding="same", kernel_initializer='he_normal')(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation("relu")(conv4)

    pool4, mask4 = MaxPoolingWithArgmax2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(1024, (3, 3), padding="same", kernel_initializer='he_normal')(pool4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation("relu")(conv5)

    pool5, mask5 = MaxPoolingWithArgmax2D(pool_size=(2, 2))(conv5)
    unpool1 = MaxUnpooling2D(size=(2, 2))([pool5, mask5])

    conv6 = Conv2D(512, (3, 3), padding="same", kernel_initializer='he_normal')(unpool1)
    conv6 = BatchNormalization()(conv6)
    conv6 = Activation("relu")(conv6)

    unpool2 = MaxUnpooling2D(size=(2, 2))([conv6, mask4])

    conv7 = Conv2D(256, (3, 3), padding="same", kernel_initializer='he_normal')(unpool2)
    conv7 = BatchNormalization()(conv7)
    conv7 = Activation("relu")(conv7)

    unpool3 = MaxUnpooling2D(size=(2, 2))([conv7, mask3])

    conv8 = Conv2D(128, (3, 3), padding="same", kernel_initializer='he_normal')(unpool3)
    conv8 = BatchNormalization()(conv8)
    conv8 = Activation("relu")(conv8)

    unpool4 = MaxUnpooling2D(size=(2, 2))([conv8, mask2])

    conv9 = Conv2D(64, (3, 3), padding="same", kernel_initializer='he_normal')(unpool4)
    conv9 = BatchNormalization()(conv9)
    conv9 = Activation("relu")(conv9)

    unpool5 = MaxUnpooling2D(size=(2, 2))([conv9, mask1])

    conv10 = Conv2D(num_classes, (1, 1), padding="same", kernel_initializer='he_normal')(unpool5)
    conv10 = BatchNormalization()(conv10)
    outputs = Activation("softmax")(conv10)

    segnet = Model(inputs=inputs, outputs=outputs)
    return segnet

def create_sliding_window_model(window_size, num_classes=2):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(window_size, window_size, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')])
    return model