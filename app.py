from flask import Flask, request, render_template, url_for, jsonify
from tensorflow.keras.models import load_model
import tensorflow as tf
from PIL import Image
import numpy as np

app = Flask(__name__)

def preprossing(image):
    image=Image.open(image)
    image = image.resize((224, 224))
    image_arr = np.array(image.convert('RGB'))
    image_arr.shape = (1, 224, 224, 3)
    return image_arr

classes = ['Blight' ,'Common Rust', 'Fall Armyworm' ,'Gray Leaf Spot' ,'Maize Streak Virus' ,'Health']
model=load_model("maize_leaf_disease_model.h5")


@app.route('/')
def index():

    return render_template('index.html', appName="Maize Disease Detector")


@app.route('/predictApi', methods=["POST"])
def api():
    # Get the image from post request
    try:
        if 'fileup' not in request.files:
            return "Please try again. The Image doesn't exist"
        image = request.files.get('fileup')
        image_arr = preprossing(image)
        print("Model predicting ...")
        result = model.predict_on_batch(image_arr)
        print("predicted ...")
        classification = np.where(result == np.amax(result))[1][0]
        print(str(result[0][classification]*100) + '% Confidence ' + names(classification))
        ind = np.argmax(result)
        prediction = str(result[0][classification]*100) + '% Confidence ' + names(classification)

        print(prediction)
        return jsonify({'prediction': prediction})
    except:
        return jsonify({'Error': 'Error occur'})

def names(number):
    if number==0:
        return "It's a leaf with BLIGHT disease"
    elif number==1:
        return "It's a leaf with Common Rust disease"
    elif number==2:
        return "It's a leaf with Gray Leaf Spot disease"
    elif number==3:
        return "It's a leaf with Fall armyworm disease"
    elif number==4:
        return "It's a leaf with Maize Streak Virus disease"
    elif number==5:
        return "It's a Healthy leaf"


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    print("run code")
    if request.method == 'POST':
        # Get the image from post request
        print("image loading....")
        image = request.files['fileup']
        print("image loaded....")
        image_arr= preprossing(image)
        print("predicting ...")
        result = model.predict_on_batch(image_arr)
        print("predicted ...")

    
        classification = np.where(result == np.amax(result))[1][0]
        print(str(result[0][classification]*100) + '% Confidence ' + names(classification))
        ind = np.argmax(result)
        prediction = str(result[0][classification]*100) + '% Confidence ' + names(classification)

        print(prediction)
        



        return render_template('index.html', prediction=prediction, image='static/IMG/', appName="Maize Disease Detector")
    else:
        return render_template('index.html',appName="Maize Disease Detector")


if __name__ == '__main__':
    app.run(debug=True)