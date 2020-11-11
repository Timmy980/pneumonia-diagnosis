import os, sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import tkinter as tk
from tkinter import filedialog

# load trained model
model = tf.keras.models.load_model('model.h5')

try:
    root = tk.Tk()
    root.withdraw()
    prod_data_path = filedialog.askdirectory()
except:
    prod_data_path = input('Enter path to image(s) folder: ')
file_list = os.listdir(prod_data_path)
image_list = list()

# load, resize, and scale images
for pic in file_list:
    try:
        im = cv2.imread(prod_data_path + '/' + pic)
        im = cv2.resize(im, (224, 224), interpolation=cv2.INTER_AREA)
        im = np.array(im)
        im = im.astype('float32')
        im /= 255
        image_list.append(im)
    except:
        print('Error: ', sys.exc_info()[0],sys.exc_info()[1])
        
image_list = np.array(image_list)

# predict classes
predictions = model.predict(image_list)
predicted_classes = np.argmax(predictions, axis=1)
predicted_classes = ['Normal' if i==0 else 'Pneumonia' for i in predicted_classes]
        
# display images and predicted classes
fig, axes = plt.subplots(1, len(predicted_classes))
fig.set_size_inches(15,4)
fig.tight_layout()
 
for ax, label, count in zip(axes, predicted_classes, range(len(predicted_classes))):
    idx = count
 
    ax.imshow(image_list[idx])
    ax.axis('off')
    ax.set_title(label)
