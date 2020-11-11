import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D


# create generators for training and validation sets
train_data_path = 'train/'
valid_data_path = 'val/'

target_size = (224,224)

train_data_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(train_data_path, target_size=target_size, classes=['NORMAL', 'PNEUMONIA'], batch_size=64)

valid_data_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(valid_data_path, target_size=target_size, classes=['NORMAL', 'PNEUMONIA'], shuffle=False, batch_size=4)


# define and compile model
base_model = InceptionV3(weights='imagenet', include_top=False)
 
x = base_model.output

x = GlobalAveragePooling2D()(x)

x = Dense(1024, activation='relu')(x)

output_layer = Dense(2, activation='softmax')(x)
 
model = Model(inputs=base_model.input, outputs=output_layer)

for layer in base_model.layers:
    layer.trainable = False
 
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

print(model.summary())

# train and save model
model.fit(
        train_data_gen,
        epochs=10,
        validation_data=valid_data_gen,
        validation_steps=4,
        verbose=1)

model.save('model.h5')