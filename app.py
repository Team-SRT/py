import os
import tensorflow as tf
from PIL import Image
import numpy as np
from flask import Flask, request, jsonify

#TODO create api

# 모델 불러오기
model = tf.keras.models.load_model('keras_model.h5')

app = Flask(__name__)

def preprocess(image_path):
    # 이미지 불러오기 및 전처리
    image = Image.open(image_path)
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    return np.expand_dims(image_array, axis=0)

def classify_image(image_path):
    preprocessed_image = preprocess(image_path)

    prediction = model.predict(preprocessed_image)

    category_index = tf.argmax(prediction, axis=-1).numpy()[0]

    threshold = 0.5
    if prediction.max() < threshold:
        category_index = 6

    return category_index

@app.route('/classify_image', methods=['POST'])
def classify_image_endpoint():
    if not request.form['file'] == '':
        print('error')
        return jsonify({'error': 'No file part in the request'})

    file = request.form['file']
    print(file)
    if file == '':
        print('error2')
        return jsonify({'error': 'No file selected'})
    image_path = "/tmp/" + file
    file.save(image_path)#save add

    category_index = classify_image(image_path)
    os.remove(image_path)

    category_index = int(category_index)

    print('success')
    return jsonify({'category_index': category_index})


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=1234)

#tt