import numpy as np
import pandas as pd
import os
from re import search
import shutil
import natsort
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorflow as tf
import cv2


DIR = r'C:\Desktop\Foliar-diseases-in-Apple-Trees-Prediction-master\images\Original Dataset'

# Loading Train and Test Data
train = pd.read_csv(r'C:\Desktop\Foliar-diseases-in-Apple-Trees-Prediction-master\labels\train.csv')
test = pd.read_csv(r'C:\Desktop\Foliar-diseases-in-Apple-Trees-Prediction-master\labels\test.csv')

# Adding the Label column for Excel File
class_names = train.loc[:, 'healthy':].columns
print(class_names)
number = 0
train['label'] = 0
for i in class_names:
    train['label'] = train['label'] + train[i] * number
    number = number + 1

# Advanced Image Processing Functions
def apply_edge_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

def adjust_contrast(image, factor=1.5):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = cv2.multiply(hsv[:, :, 2], factor)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def reduce_noise(image):
    return cv2.GaussianBlur(image, (5, 5), 0)

def get_label_img(img):
    if search("Train", img):
        img = img.split('.')[0]
        label = train.loc[train['image_id'] == img]['label']
        return label


# Creating Train and test folders
def create_train_data():
    images = natsort.natsorted(os.listdir(DIR))
    for img in tqdm(images):
        label = get_label_img(img)
        path = os.path.join(DIR, img)
                # Read the image
        image = cv2.imread(path, cv2.IMREAD_COLOR)

        # Apply advanced image processing
        image = apply_edge_detection(image)
        image = adjust_contrast(image)
        image = reduce_noise(image)
        if search("Train", img):
            if (img.split("_")[1].split(".")[0]) and label.item() == 0:
                shutil.copy(path, r'C:\Desktop\Foliar-diseases-in-Apple-Trees-Prediction-master\images\train\healthy')
            elif (img.split("_")[1].split(".")[0]) and label.item() == 1:
                shutil.copy(path, r'C:\Desktop\Foliar-diseases-in-Apple-Trees-Prediction-master\images\train\multiple_disease')
            elif (img.split("_")[1].split(".")[0]) and label.item() == 2:
                shutil.copy(path, r'C:\Desktop\Foliar-diseases-in-Apple-Trees-Prediction-master\images\train\rust')
            elif (img.split("_")[1].split(".")[0]) and label.item() == 3:
                shutil.copy(path, r'C:\Desktop\Foliar-diseases-in-Apple-Trees-Prediction-master\images\train\scab')
        elif search("Test", img):
            shutil.copy(path, r'C:\Desktop\Foliar-diseases-in-Apple-Trees-Prediction-master\images\test')
    shutil.os.mkdir(r'C:\Desktop\Foliar-diseases-in-Apple-Trees-Prediction-master\images\train')
    shutil.os.mkdir(r'C:\Desktop\Foliar-diseases-in-Apple-Trees-Prediction-master\images\train\healthy')
    shutil.os.mkdir(r'C:\Desktop\Foliar-diseases-in-Apple-Trees-Prediction-master\images\train\multiple_disease')
    shutil.os.mkdir(r'C:\Desktop\Foliar-diseases-in-Apple-Trees-Prediction-master\images\train\rust')
    shutil.os.mkdir(r'C:\Desktop\Foliar-diseases-in-Apple-Trees-Prediction-master\images\train\scab')
    shutil.os.mkdir(r'C:\Desktop\Foliar-diseases-in-Apple-Trees-Prediction-master\images\test')
    train_dir = create_train_data()
    Train_DIR = r'C:\Desktop\Foliar-diseases-in-Apple-Trees-Prediction-master\images\train'
    Categories = ['healthy', 'multiple_disease', 'rust', 'scab']

    # Preprocessing
    for j in Categories:
        path = os.path.join(Train_DIR, j)

        for img in os.listdir(path):
            old_image = cv2.imread(os.path.join(path, img), cv2.COLOR_BGR2RGB)
            plt.imshow(old_image)
            plt.show()
            break
        break

    IMG_SIZE = 224
    new_image = cv2.resize(old_image, (IMG_SIZE, IMG_SIZE))
    plt.imshow(new_image)
    plt.show()

# Implementation of CNN
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.layers import Input, Reshape, UpSampling2D
from tensorflow.keras.models import Model



class CustomLayer(tf.keras.layers.Layer):
    def init(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(CustomLayer, self).init(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)

    def call(self, inputs):
        return tf.matmul(inputs, self.kernel)

def autoencoder_generator(datagen):
    for batch, _ in datagen:
        yield (batch, batch)


IMG_SIZE = 224
# Define the encoder part of the autoencoder
inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# Define the decoder part of the autoencoder
x = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

# Define the full autoencoder model
autoencoder = Model(inputs, decoded)
    
datagen = ImageDataGenerator(rescale=1./255,
                           shear_range=0.2,
                           zoom_range=0.2,
                           horizontal_flip=True,
                           vertical_flip=True,
                           validation_split=0.2)

train_datagen = datagen.flow_from_directory(r'C:\Desktop\Foliar-diseases-in-Apple-Trees-Prediction-master\images\train',
                                            target_size=(IMG_SIZE, IMG_SIZE),
                                            batch_size=16,
                                            class_mode='categorical',
                                            subset='training'
                                            )

val_datagen = datagen.flow_from_directory(r'C:\Desktop\Foliar-diseases-in-Apple-Trees-Prediction-master\images\train',
                                          target_size=(IMG_SIZE, IMG_SIZE),
                                          batch_size=16,
                                          class_mode='categorical',
                                          subset='validation'
                                          )

# Convert your existing data generators
train_autoencoder_gen = autoencoder_generator(train_datagen)
val_autoencoder_gen = autoencoder_generator(val_datagen)

model = Sequential()
model.add(Conv2D(64,(3,3),activation='relu',padding='same',input_shape=(IMG_SIZE,IMG_SIZE,3)))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(2, 2))
model.add(Flatten())
model.add(Dense(4, activation='softmax'))

# Compile the Model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

checkpoint = ModelCheckpoint(r'C:\Desktop\Foliar-diseases-in-Apple-Trees-Prediction-master\models\apple-224.h5',
                             monitor='val_loss',
                             mode='min',
                             save_best_only=True,
                             verbose=1)

earlystop = EarlyStopping(monitor='val_loss',
                          min_delta=0,
                          patience=10,
                          verbose=1,
                          restore_best_weights=True)

callbacks = [checkpoint, earlystop]

# autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Now you can train your autoencoder on the processed images
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.fit_generator(train_autoencoder_gen, 
                          validation_data=val_autoencoder_gen,
                          epochs=30,
                          steps_per_epoch=train_datagen.samples // 16,
                          validation_steps=val_datagen.samples // 16,
                          callbacks=callbacks)


encoder = Model(inputs, encoded)

model_history = model.fit_generator(train_datagen, validation_data=val_datagen,
                                    epochs=30,
                                    steps_per_epoch=train_datagen.samples // 16,
                                    validation_steps=val_datagen.samples // 16,
                                    callbacks=callbacks)

# Training and Validation Accuracy
acc_train = model_history.history['accuracy']
acc_val = model_history.history['val_accuracy']
epochs = range(1, 31)
plt.plot(epochs, acc_train, 'g', label='Training Accuracy')
plt.plot(epochs, acc_val, 'b', label='Validation Accuracy')
plt.title("Training and Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# Training and Validation loss
loss_train = model_history.history['loss']
loss_val = model_history.history['val_loss']
epochs = range(1, 31)
plt.plot(epochs, loss_train, 'g', label='Training Loss')
plt.plot(epochs, loss_val, 'b', label='Validation Loss')
plt.title("Training and Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()