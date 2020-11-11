import seaborn as sn
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import numpy as np
matplotlib.rcParams['axes.titlepad'] = 20

# load trained model
model = tf.keras.models.load_model('model.h5')

# load test images
target_size = (224,224)
test_data_path = 'test/'
test_data_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(test_data_path, target_size=target_size, classes=['NORMAL', 'PNEUMONIA'], shuffle=False, batch_size=10)

# calculate accuracy
score = model.evaluate(test_data_gen, verbose=1)
print('Model Accuracy:', score[1])

# predict classes of test images
test_data_gen.reset()
predictions = model.predict(test_data_gen)
# Get most likely class
predicted_classes = np.argmax(predictions, axis=1)

# generate classification report
test_data_gen.reset()
true_classes = test_data_gen.classes
class_labels = list(test_data_gen.class_indices.keys())  
report = classification_report(true_classes, predicted_classes, target_names=class_labels)
print(report)


# generate confusion matrix 
cm = confusion_matrix(true_classes, predicted_classes)
df_cm = pd.DataFrame(cm, index = [i for i in class_labels],
                  columns = [i for i in class_labels])
plt.figure(figsize = (5,5))
sn.heatmap(df_cm, annot=True, fmt='d', cbar=True, cmap=['Yellow', 'green', 'red'], annot_kws={"size": 16})
plt.title('Confusion Matrix', fontsize=24)
plt.xlabel('Actual', fontsize=18)
plt.ylabel('Predicted', fontsize=18)
