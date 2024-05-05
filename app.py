from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Flatten, Dropout
from sklearn.metrics import classification_report
from transformers import TFDistilBertModel, DistilBertTokenizer
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

app = Flask(__name__)

# Load the pre-trained classification model
classification_model = load_model('classification_model.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']

    # Preprocess the input text
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([text])
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=1000, padding='post', truncating='post')

    # Concatenate the LSTM features with the input data
    lstm_features = classification_model.predict(padded_sequence)
    input_data = np.concatenate((padded_sequence, lstm_features), axis=1)

    # Make the prediction
    prediction = classification_model.predict(input_data)
    predicted_class = 'Fake' if prediction[0][0] > 0.5 else 'Real'

    return render_template('result.html', prediction=predicted_class)

if __name__ == '__main__':
    app.run(debug=True)