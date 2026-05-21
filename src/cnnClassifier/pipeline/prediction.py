import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os


class PredictionPipeline:

    def __init__(self, filename):
        self.filename = filename


    def predict(self):

        # Load trained model
        model = load_model(
            os.path.join("model", "model.h5")
        )

        # Load image
        test_image = image.load_img(
            self.filename,
            target_size=(224, 224)
        )

        # Convert image to array
        test_image = image.img_to_array(test_image)

        # Normalize image
        test_image = test_image / 255.0

        # Expand dimensions
        test_image = np.expand_dims(test_image, axis=0)

        # Prediction
        result = np.argmax(model.predict(test_image), axis=1)

        print(result)

        # Class labels
        if result[0] == 1:
            prediction = "Tumor"
        else:
            prediction = "Normal"

        return [{"image": prediction}]