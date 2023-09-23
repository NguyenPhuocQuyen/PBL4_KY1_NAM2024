import numpy as np 
import tensorflow as tf 
import tensorflow_hub as hub 
import os 
import cv2
import matplotlib.pyplot as plt
from enum import Enum  

class Categories(Enum):
    CARDBOARD = 0 
    GLASS = 1 
    METAL = 2 
    PAPER = 3 
    PLASTIC = 4 
    TRASH = 5 
    OTHER = -1 
IMAGE_SIZE = 256 
RESCALING = True
COLOR_MODE = "rgb"
# COLOR_MODE = "grayscale"
if COLOR_MODE == "rgb":
    CHANNELS = 3
else: 
    CHANNELS = 1 
def classifyObject(img):
    model_load = tf.keras.models.load_model(filepath = r"D:\Learning\Ky1_Nam4\PBL4\Trash_Classification\ai_models\model_trashclassification2.h5",
                                            custom_objects = None, compile = True, options = None)
    if COLOR_MODE == "grayscale":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img,(IMAGE_SIZE,IMAGE_SIZE))
    if RESCALING == True:
        img_preprocessed = img/255.0
    else:
        img_preprocessed = img
    outputs_prediction = model_load.predict(img_preprocessed.reshape((1,IMAGE_SIZE,IMAGE_SIZE,CHANNELS)))
    print(outputs_prediction[0])
    prediction = np.argmax(outputs_prediction)
    print(prediction)
    if outputs_prediction[0][prediction] <= 0.35:
        prediction = -1 
    print(prediction)
    return prediction
def testIAwithImage():
    img = cv2.imread(r"D:\Learning\Ky1_Nam4\PBL4\Trash_Classification\archive\Garbage_classification\Garbage_classification\paper\paper73.jpg")
    cv2.imshow("TEST_IMAGE", img)
    prediction = classifyObject(img)
    print(f"Class with the highest probability: {Categories(prediction)}")
    print(f"Oject belongs to category number: {prediction}")
    cv2.imshow("TEST_IMAGE", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
if __name__ == "__main__":
    testIAwithImage()

