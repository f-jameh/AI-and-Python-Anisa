# In the name of God



# import library

import os
import pandas as pd
import re
import pickle
import numpy as np
import tensorflow as tf
from bs4 import BeautifulSoup
from flask import Flask, request, render_template
from tensorflow.keras.preprocessing.sequence import pad_sequences


########################### Load Model & Tokenizer ##############################

# Load the trained model
model = tf.keras.models.load_model('spam_classification.h5')

# Load the tokenizer
with open('tokenizer.pkl', 'rb') as file:
    tokenizer = pickle.load(file)


########################### import Dataset ################################ 
ham_dir = '/home/farhad/Desktop/AI-and-Python-Anisa/Session16/easy_ham_2_fine_tune'
spam_dir = "/home/farhad/Desktop/AI-and-Python-Anisa/Session16/spam_2_fine_tune"

# create 2 empty list to add emails (ham & spam) to them
spam_emails = []
ham_emails = []

# read the spam emails and adding them to list
for filename in os.listdir(spam_dir):
    print(filename)
    with open(os.path.join(spam_dir, filename), 'r', encoding='latin-1') as file:
        spam_emails.append(file.read())

# read the ham emails and adding them to list
for filename in os.listdir(ham_dir):
    print(filename)
    with open(os.path.join(ham_dir, filename), 'r', encoding='latin-1') as file:
        ham_emails.append(file.read())

# convert lists to dataframe
data = pd.DataFrame({'email': ham_emails + spam_emails, 'label': [0] * len(ham_emails) + [1] * len(spam_emails)})

print(f'loaded {len(data)} new emails')

#%%
################################# pre-processing and Data cleaning ############
#   Remove HTML ## only for emails
#   Remove Email-headers  ## only for emails
#   Remove Special characters (" ; , >, ...)  # for all texts
#   Convert to Lowercase                      # for all texts
#   Remove extera spaces                      # for all texts

import re
from bs4 import BeautifulSoup

# create an empty list to store cleaned emails
clean_emails = []

# create a loop for reading all emails to pre-processing
for i in range(len(data)):
    email_text = data['email'][i]
    email_text = BeautifulSoup(email_text, 'html.parser').get_text()
    email_text = re.sub(r'From:.*|To:.*|Subject:.*', '', email_text)
    email_text = re.sub(r'[^a-zA-Z\s]', '', email_text)
    email_text = email_text.lower()
    email_text = re.sub(r'\s+',  ' ', email_text).strip()

    clean_emails.append(email_text)

data['clean_email'] = clean_emails
print('finish')

#%%
############################ tokenization, padding, label array#####################################

from tensorflow.keras.preprocessing.text import Tokenizer

email_sequences = tokenizer.texts_to_sequences(data['clean_email'])

max_lengh = 200
padded_sequences = pad_sequences(email_sequences, maxlen=max_lengh, padding='post')

labels = np.array(data['label'])

print('finish')

#%% ##########################################  Defining the Model ###################

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, GRU, RNN

# summary of model
model.summary()

# compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#Train the model
history = model.fit(padded_sequences, labels, epochs=5, batch_size=32, validation_split=0.2)

print('finish')

#%% 
# save the model
model.save('updated_model.keras')
print('model has been saved in updated_model.keras')

#%% ploting
# import library
import matplotlib.pyplot as plt

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
