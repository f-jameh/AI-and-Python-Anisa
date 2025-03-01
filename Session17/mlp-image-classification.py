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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
import matplotlib.pyplot as plt


# set dataset path
dataset_path = '/home/farhad/Desktop/data/datasets/fire_dataset'

# set image size
img_size = 64

# create empty list to store all data
data = []

# create a list of clas names (folder namse)
class_names = os.listdir(dataset_path)
print(class_names)

for class_name in class_names:
    class_path = os.path.join(dataset_path, class_name)
    
    for img_path in glob.glob(os.path.join(class_path, "*.*")):
        img = cv2.imread(img_path)
        
        if img is None:
            print(f'warning!!!, skipping unreadable file {img_path}')
            continue
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert image to gray
        img = cv2.resize(img, (img_size, img_size))# resize
        img = img / 255.0 # normalize
        img = img.flatten()  # flat images
        
        data.append([img, class_name]) # store images in list
    
# Convert list to DataFrame
df = pd.DataFrame(data, columns=['data', 'label'])

print(len(df))

#%% label encoding & train test split

# extract images (x) and labels (y)
X = np.array(df['data'].tolist())
y = np.array(df['label'].tolist())

# encode labels
le = LabelEncoder()
y = le.fit_transform(y)
y = to_categorical(y)

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y,  test_size=0.2, random_state=42)

#%% Define model

model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(len(le.classes_), activation='softmax')
    ])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()


#%% Train

history = model.fit(X_train, y_train, epochs=15, batch_size=32, validation_data=(X_test, y_test))

#%% plotting

# Plot Training & Validation Accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')

plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot Training & Validation Loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')

plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


