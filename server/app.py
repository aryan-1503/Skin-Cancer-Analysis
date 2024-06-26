from flask import Flask, request, jsonify
from PIL import Image
from flask_cors import CORS
from tensorflow.keras.models import load_model
import numpy as np
import pickle

app = Flask(__name__)
CORS(app)

model = load_model('skin_cancer_model.h5')



classes = {4: ('nv', ' Melanocytic Nevi'),
           6: ('mel', 'Melanoma'),
           2: ('bkl', 'Benign keratosis-like lesions'),
           1: ('bcc', ' Basal cell carcinoma'),
           5: ('vasc', ' Pyogenic granulomas and hemorrhage'),
           0: ('akiec', 'Actinic keratoses and intraepithelial carcinomae'),
           3: ('df', 'Dermatofibroma')}

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image found in the request'})

    img = request.files['image']
    image = Image.open(img)
    image = image.resize((28, 28))
    img_array = np.array(image)
    img_array = img_array.reshape(-1, 28, 28, 3)

    result = model.predict(img_array)
    max_prob = max(result[0])
    class_ind = result[0].tolist().index(max_prob)

    prediction = {'class': classes[class_ind][0], 'description': classes[class_ind][1]}
    return jsonify(prediction)


if __name__ == '__main__':
    app.run(debug=True)
