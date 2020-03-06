import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import matplotlib.pyplot as plt
import models
import math

# This file is used to predict the class to which the predicted emotion belongs to

def prediction(image):
    class_labels = {0: 'Anger', 1: 'Neutral', 2: 'Disgust', 3: 'Happy', 4: 'Sad',
                    5: 'Surprise'}

    model = models.model_h6()
    model.load_weights('Trained_ckplus_model.h5')

    img = np.array(cv2.resize(image, (48,48))).reshape(1,48,48,3)

    predictions = model.predict_classes(img)
    print('Predictions:', class_labels[predictions[0]])

    return class_labels[predictions[0]]
