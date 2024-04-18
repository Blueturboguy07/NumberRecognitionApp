from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.vgg16 import preprocess_input
import os
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
model = load_model('NumRecognition.h5')
target_img = os.path.join(os.getcwd() , 'static/images')
@app.route('/')
def index_view():
    return render_template('index.html')
#Allow files with extension png, jpg and jpeg
ALLOWED_EXT = set(['jpg' , 'jpeg' , 'png'])
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXT
           
# Function to load and prepare the image in right shape
def read_image(filename):
    img = load_img(filename, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x


@app.route('/predict',methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            raise FileNotFoundError("File not found in the request.")
        
        file = request.files['file']
        
        if file.filename == '':
            raise ValueError("No file selected.")
        
        if not allowed_file(file.filename):
            raise ValueError("Invalid file format.")
        
        filename = file.filename
        file_path = os.path.join('static/images', filename)
        file.save(file_path)
        
        img = read_image(file_path)  # Assuming this function is defined elsewhere
        if img is None:
            os.remove(file_path)  # Remove the file if image reading failed
            raise ValueError("Failed to read the image.")
        
        class_prediction = model.predict(img)  # Assuming 'model' is defined elsewhere
        if class_prediction is None:
            os.remove(file_path)  # Remove the file if prediction failed
            raise ValueError("Failed to predict.")
        
        fruit = np.argmax(class_prediction, axis=1)
        
        # Render the template with the results
        return render_template('predict.html', fruit=fruit, prob=class_prediction, user_image=file_path)
    else:
        raise ValueError("Invalid request method. Only POST requests are allowed.")

if __name__ == '__main__':
    app.run(debug=True,use_reloader=False, port=8000)