# In the name of God

import os
import re
import pickle
import numpy as np
import tensorflow as tf
from bs4 import BeautifulSoup
from flask import Flask, request, render_template
from tensorflow.keras.preprocessing.sequence import pad_sequences

########################### Load Model & Tokenizer ##############################

# Load the trained model
model = tf.keras.models.load_model('updated_model.keras')

# Load the tokenizer
with open('tokenizer.pkl', 'rb') as file:
    tokenizer = pickle.load(file)

# Define max sequence length (same as training)
max_length = 200

########################### Flask Web API ##############################

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    email_text = request.form['email']

    # Preprocessing
    email_text = BeautifulSoup(email_text, 'html.parser').get_text()
    email_text = re.sub(r'From:.*|To:.*|Subject:.*', '', email_text)
    email_text = re.sub(r'[^a-zA-Z\s]', '', email_text)
    email_text = email_text.lower()
    email_text = re.sub(r'\s+', ' ', email_text).strip()

    # Tokenize & Pad
    email_sequence = tokenizer.texts_to_sequences([email_text])
    email_padded = pad_sequences(email_sequence, maxlen=max_length, padding='post')

    # Predict
    prediction = model.predict(email_padded)[0][0]

    # Classify as Spam or Not Spam (Threshold 0.7 for better accuracy)
    result = "Spam ❌" if prediction > 0.5 else "Not Spam ✅"

    return render_template('result.html', result=result)
    print(result)

########################### Run Flask App ##############################
if __name__ == '__main__':
    app.run(host='172.16.1.1', debug=True, use_reloader=False)
    