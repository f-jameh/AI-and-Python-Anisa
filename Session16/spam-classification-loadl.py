# In the name of God

# import libraries

import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from bs4 import BeautifulSoup
import re
import tensorflow as tf
import pickle

# load the model
model = tf.keras.models.load_model('/home/farhad/Desktop/AI-and-Python-Anisa/Session16/spam_classification.h5')

# load tokenizer
with open('/home/farhad/Desktop/AI-and-Python-Anisa/Session16/tokenizer.pkl', 'rb') as file:
    tokenizer = pickle.load(file)

# create sample emails
email1 = "URGENT!!! your bank account is hacked and in at risk. click link beloww to verify"
email2 = 'your tomorow metting is canceled'


email_text = email2

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

# convert email to sequence using the trained tokenizer
email_text = tokenizer.texts_to_sequences([email_text])

# pad all sequences to the same lengh
padded_email = pad_sequences(email_text, maxlen=200, padding='post')

# make prediction
predict = model.predict(padded_email)

print(predict)

if predict > 0.7:
    print("this is not spam")
else:
    print('this is spam')










