from flask import Flask,request,render_template
import numpy as np
import pandas as pd
import pickle
import dill
import keras
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

application = Flask(__name__)
app = application

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        model = keras.models.load_model('saved_model.keras')
        text = request.form.get('text')
        with open("tokenizer.pkl", "rb") as handle:
          tokenizer = pickle.load(handle) 
        # print(tokenizer.texts_to_sequences(text))
        print(text)
        text = [text]
        seq = tokenizer.texts_to_sequences(text)
        padded = pad_sequences(seq, maxlen=3000)
        pred = model.predict(padded)
        labels = ['Business','Entertainment','Politics','Sports','Tech']
        # print(pred, labels[np.argmax(pred)])
        return render_template('index.html',result=labels[np.argmax(pred)])

if __name__ == '__main__':
    app.run(host="0.0.0.0",debug=True)