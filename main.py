from flask import Flask , render_template , request , jsonify
from keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model  
from keras.preprocessing.image import img_to_array
import numpy as np
import os 

import tensorflow as tf


model = tf.keras.models.load_model('model/Alzheimer_Final_Model_forReal.keras')

    
app = Flask(__name__) 

@app.route('/' , methods=['GET'])
def Load_front():
    return render_template('index.html')


@app.route('/' , methods=['POST'])
def predict():


    app.config['UPLOAD_FOLDER'] = './images/'

    if 'imageFile' not in request.files or request.files['imageFile'].filename == '':
        prediction_message = "You must upload a photo."
        return render_template('index.html', prediction=prediction_message)
    
    
    imageFile = request.files['imageFile']
    imgPath = './images/' + imageFile.filename
    imageFile.save(imgPath)


   

    img = load_img(imgPath , target_size = (256 , 256))
    image = img_to_array(img)
    image = np.expand_dims(image, axis=0) 
    image /= 255.0
    


    predictions = model.predict(image)
    predicted_class = np.argmax(predictions, axis=1)[0]

    
    if predicted_class == 0:
        prediction_message = "Mild Alzheimer"
    elif predicted_class == 1:
        prediction_message = "Moderate Alzheimer"
    elif predicted_class == 2:
        prediction_message = "Don't have Alzheimer"
    elif predicted_class == 3:
        prediction_message = "Very mild Alzheimer"
    elif predicted_class == 4:
        prediction_message = "Unknown class , Please make sure that you uploaded a mri brain photo"


    return render_template('index.html' ,  prediction = prediction_message)


if __name__ == '__main__':
    app.run(port=3000, debug=True, use_reloader=False)
