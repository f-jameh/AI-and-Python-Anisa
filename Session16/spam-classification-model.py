# In the name of God

############################ import libraries ##############################
import os
import pandas as pd

########################### import Dataset ################################ 
ham_dir = '/home/farhad/Desktop/AI-and-Python-Anisa/Session16/balance-ham'# ham dir that contains spam hams
spam_dir = '/home/farhad/Desktop/AI-and-Python-Anisa/Session16/balance-spam'   # spam dir that contains spam samples

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

print(data.head())


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
    
    # remove html tags
    email_text = BeautifulSoup(email_text, 'html.parser').get_text()
    
    # remove email headers
    email_text = re.sub(r'From:.*|To:.*|Subject:.*', '', email_text)
    
    # remove special characters
    email_text = re.sub(r'[^a-zA-Z\s]', '', email_text)
    
    # convert to lowrrcase
    email_text = email_text.lower()
    
    # remove extra white spaces
    email_text = re.sub(r'\s+',  ' ', email_text).strip()
    
    # append to clean_emails list
    clean_emails.append(email_text)

# add cleane_emails list to dataframe
data['clean_email'] = clean_emails

print(data.head(5))
data.to_csv('final_email_dataset.csv')


#%%
############################ tokenization #####################################

# import library for tokenization
from tensorflow.keras.preprocessing.text import Tokenizer

# create a tokenizer (object)
tokenizer = Tokenizer(num_words=10000) # num_words is number of most frequent words in emails

# Fit (tokeniz) on cleaned emails (learn word-to-number mapping)
tokenizer.fit_on_texts(data['clean_email'])

# conver words (cleaned emails) to sequence of tokens (numbers)
email_sequences = tokenizer.texts_to_sequences(data['clean_email'])

# [optinal] print some sequences from entir daraset
print(email_sequences[:5])

############################ Padding #####################################

# import libraries
from tensorflow.keras.preprocessing.sequence import pad_sequences

# define a maximum lengh for each sequence (tokenized email)
max_lengh = 200

# pad all sequences to the same lengh
padded_sequences = pad_sequences(email_sequences, maxlen=max_lengh, padding='post')

# [optinal] print first 5 padded sequences
print(email_sequences[:5])

#%%

########################################## spliting dataset ###################

#import libraries
from sklearn.model_selection import train_test_split
import numpy as np

labels = np.array(data['label'])

# split dataset (padded sequences) into train and test parts
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, labels, test_size=0.2, random_state=42)

#%%
##########################################  Defining the Model ###################

# import libraries
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, GRU, RNN

# define model for binary classification (only 1 neuron in output)

# create an object for model
model = Sequential([
    Embedding(input_dim=10000, output_dim=64, input_length=200),
    LSTM(64, dropout=0.2, recurrent_dropout=0.2),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
    ])

# compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# [optional] print the summary
model.summary()



#%%
##########################################  training model ###################

history = model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2)

#%%

# evaluate model

loss, accuracy = model.evaluate(X_test, y_test)

print(f'Test Loss is: {loss:.4f}')
print(f'Test Accuracy is: {accuracy:.4f}')


#%%
##########################################  plotting section ###################

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


#%%
##########################################  save model ###################

# import library
import pickle

# save model in h5 format
model.save('spam_classification.h5')

# save model in keras format
model.save('spam_classification.keras')

# save the tokenizer (to use in predection latar)
with open('tokenizer.pkl', 'wb') as file:
    pickle.dump(tokenizer, file)











