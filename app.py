from flask import Flask,request,render_template
import numpy as np
import pandas as pd
import pickle
import dill
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
        pickle_in =  open("artifacts/bcc_classification.pkl","rb")
        model = pickle.load(pickle_in)
        text = request.form.get('text')
        tokenizer = Tokenizer(num_words=17727, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~',lower=True)
        tokenizer.fit_on_texts([text])
        encoded_text = tokenizer.texts_to_sequences([text])[0]
        encoded_text = pad_sequences([encoded_text],maxlen=3000)
        labels = ['Business','Entertainment','Politics','Sports','Tech']
        predict = model.predict(encoded_text)
        result = labels[np.argmax(predict)]
        print(result)
        return render_template('index.html',result=result)

if __name__ == '__main__':
    app.run(host="0.0.0.0",debug=True)