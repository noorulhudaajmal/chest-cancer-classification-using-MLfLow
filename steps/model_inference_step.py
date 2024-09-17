import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os



def predict(image_file):
    model = load_model(os.path.join("../model", "model.keras"))
    test_image = image.load_img(image_file, target_size = (224,224))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = np.argmax(model.predict(test_image), axis=1)
    print(result)

    if result[0] == 1:
        prediction = 'Normal'
        return [{ "image" : prediction}]
    else:
        prediction = 'Adenocarcinoma Cancer'
        return [{ "image" : prediction}]

