# In the name of God

# import libraries
import os
import numpy as np
import cv2
import pandas as pd
import glob
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Input
import matplotlib.pyplot as plt

img_path = str(input('enter your image path: '))

# set image size
img_size = 64



model = load_model('cnn_image_classification.keras')


class_labels = ['fier', 'non_fire']

le = LabelEncoder()
le.fit(class_labels)






img = cv2.imread(img_path)
if img is None:
    print('error')

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert image to gray
img = cv2.resize(img, (img_size, img_size))# resize

img = img / 255.0 # normalize


img = np.expand_dims(img, axis=-1)
img = np.expand_dims(img, axis=0)

prediction = model.predict(img)


predicted_class = np.argmax(prediction)


predicted_label = le.inverse_transform([predicted_class])[0]
confidence_rate = prediction[0][predicted_class] * 100

print(f'predicted class: {predicted_label} with confidence of  {confidence_rate:.2f}')




