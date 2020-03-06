import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import matplotlib.pyplot as plt
import models

base_dir = 'datasets/ck_data'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')


model = models.model_h9()


# The code in the comments is used to generate the model for Sentiment Analysis based on the ckplus data
# The plot functions are used to plot a graph between the training accuracy and the testing accuracy


model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001,
                                                 beta_1=0.15,
                                                 beta_2=0.85),
              metrics=['acc'])


train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(48, 48),
        batch_size=26,
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(48, 48),
        batch_size=10,
        class_mode='categorical')

history = model.fit_generator(
      train_generator,
      steps_per_epoch=35,  # 2000 images = batch_size * steps
      epochs=80,
      validation_data=validation_generator,
      validation_steps=6,  # 1000 images = batch_size * steps
      verbose=1
)
model.save_weights('Trained_ckplus_model.h5')

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()

plt.show()
