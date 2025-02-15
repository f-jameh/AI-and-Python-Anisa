# In the name of God

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt


############################## Pre-process ####################################

# load the dataset
file_path = '/home/farhad/Desktop/AI-and-Python-Anisa/Session13/iris_labeled.csv'
data = pd.read_csv(file_path)

# separate features (x) and labels (y)
x = data [['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']]
y = data [['Species']]

# spliting train from test data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# one-host encoding
encoder =OneHotEncoder(sparse_output=False)
y_train_encode = encoder.fit_transform(y_train)
y_test_encode = encoder.transform(y_test)

####################### Build Neural Network Architucture #####################

model = tf.keras.Sequential([
     tf.keras.layers.Dense(24, activation='relu', input_shape=(x_train.shape[1],)),
     tf.keras.layers.Dense(32, activation='relu'),
     tf.keras.layers.Dense(3, activation='softmax')
     ])

####################### Compiling The model  ##################################

model.compile(
    optimizer='SGD',    # or adam
    loss='categorical_crossentropy',
    metrics = ['accuracy']
    )
model.summary()

####################### Training The model  ##################################

history = model.fit(x_train, y_train_encode,
                    epochs=30, batch_size=16,
                    validation_split=0.2,
                    verbose=1)
# optional
print(history.history)

####################### Evaluting The model  ##################################




#######################  Ploting (reporting)  ##################################




#######################   Save Model  ##################################







