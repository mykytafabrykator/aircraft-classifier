import tensorflow as tf
import numpy as np


IMAGE_SIZE = (180, 180)

CLASS_NAMES = ["F16", "Mi24", "B1", "C130"]


def processing_image(image):
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = tf.expand_dims(img_array, 0)

    return img_array


def load_and_preprocess_image(path):
    image = tf.keras.preprocessing.image.load_img(
        path,
        target_size=IMAGE_SIZE
    )

    return processing_image(image)


def classify(model, image_path):
    preprocessed_image = load_and_preprocess_image(image_path)

    predictions = model.predict(preprocessed_image)
    predicted_index = np.argmax(predictions[0])
    predicted_label = CLASS_NAMES[predicted_index]
    predicted_prob = float(predictions[0][predicted_index])

    return predicted_label, round(predicted_prob * 100, 2)
