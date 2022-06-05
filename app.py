import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import tensorflow as tf

app = Flask(__name__)
model = tf.keras.models.load_model('ANN')
model = pickle.load(open('IrisFlowerPredictor.mdl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    sepalLength = float(request.form['sepalLength'])
    sepalWidth = float(request.form['sepalWidth'])
    petalLength = float(request.form['petalLength'])
    petalWidth = float(request.form['petalWidth'])
    
    finalFeatures = np.array([[sepalLength,sepalWidth,petalLength,petalWidth]])
    prediction = model.predict(finalFeatures)


    

    return render_template('index.html', prediction_text='Expected Profit from the Startup is  $ {}'.format(prediction[0]))


if __name__ == "__main__":
    app.run(debug=True)