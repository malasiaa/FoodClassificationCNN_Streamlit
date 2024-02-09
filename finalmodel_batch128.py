import pathlib

# current working directory
print(pathlib.Path().absolute())

# import necessary modules and libraries
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import RMSprop
from sklearn.utils import class_weight
from tensorflow.keras.metrics import F1Score
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
import tensorflow_addons as tfa
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
from tensorflow.keras import regularizers
from sklearn.metrics import f1_score
import tensorflow as tf
from tensorflow.keras import backend as K
import os
import splitfolders
import warnings
import zipfile

# Metrics used for model evalution
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

# Set environment variables for Kaggle API. We are getting the dataset directly from Kaggle
os.environ["KAGGLE_USERNAME"] = get_kaggle_credentials('kaggle_credentials.txt', 'KAGGLE_USERNAME')
os.environ["KAGGLE_KEY"] = get_kaggle_credentials('kaggle_credentials.txt', 'KAGGLE_KEY')

# Authenticate and initialize Kaggle API
from kaggle.api.kaggle_api_extended import KaggleApi
warnings.filterwarnings("ignore")
api = KaggleApi()
api.authenticate()

# Download dataset from Kaggle
api.dataset_download_files("harishkumardatalab/food-image-classification-dataset", path=".")

# Unzip the downloaded dataset 
with zipfile.ZipFile("food-image-classification-dataset.zip", "r") as z:
    z.extractall(".")

# Define path to storage the dataset
path = './Food Classification dataset'

import shutil
# Function to copy folders from source to destination directory
def copy_folders(source_directory, destination_directory):
    try:
        # Copy the entire directory tree from source to destination
        shutil.copytree(source_directory, destination_directory)
        print(f"Folders from '{source_directory}' successfully copied to '{destination_directory}'.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Define source and destination directories for dataset
source_directory = './Food Classification dataset'
destination_directory = './Food Classification dataset partial'

# Copy folders
copy_folders(source_directory, destination_directory)

# Split dataset. The dataset folder is constituted by 34 category folders. The split is being applyed to each of those folders.
splitfolders.ratio(destination_directory, seed=1337, output="partial-food-dataset-splitted", ratio=(0.6, 0.2, 0.2))

# Part 1 - Building the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(64, (3, 3), input_shape=(256, 256, 3), activation='relu', padding='same'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Add batch normalization after convolutional layers
classifier.add(BatchNormalization())

# Adding a second convolutional layer
classifier.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Add batch normalization after convolutional layers
classifier.add(BatchNormalization())

# Adding a third convolutional layer
classifier.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Add batch normalization after convolutional layers
classifier.add(BatchNormalization())

# Adding a third convolutional layer
classifier.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Add batch normalization after convolutional layers
classifier.add(BatchNormalization())

# Step 3 - Flattening
classifier.add(Flatten())
classifier.add(Dropout(0.2))

# Step 4 - Full connection
classifier.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
classifier.add(Dense(34, activation='softmax'))

# Compiling the CNN
optimizer = Adam(learning_rate=0.0001)
classifier.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy', f1_m, 'Precision', 'Recall'])


# Part 2 - Fitting the CNN to the images
# Define image data generators for training and validation sets
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, rotation_range = 20)
test_datagen = ImageDataGenerator(rescale=1./255)

# Load and preprocess the training set
training_set = train_datagen.flow_from_directory(
    r'./partial-food-dataset-splitted/train',
    target_size=(256, 256),
    batch_size=128,
    class_mode='categorical'
)

# Load and preprocess the validation set
validation_set = test_datagen.flow_from_directory(
    r'./partial-food-dataset-splitted/val',
    target_size=(256, 256),
    batch_size=128,
    class_mode='categorical'
)

# Print data generator parameters
datagen = ImageDataGenerator()
print("Rotation Range: ", datagen.rotation_range)
print("Width Shift Range: ", datagen.width_shift_range)
print("Height Shift Range: ", datagen.height_shift_range)
print("Shear Range: ", datagen.shear_range)
print("Zoom Range: ", datagen.zoom_range)

# Define class weights to address the dataset imbalancement 
class_weights = class_weight.compute_class_weight("balanced", classes = np.unique(training_set.classes), y=training_set.classes)
class_weights = dict(enumerate(class_weights))

# Define early stopping callback to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Fit the model
hist = classifier.fit(training_set,
                      epochs=30,
                      validation_data=validation_set,
                      class_weight = class_weights,
                      callbacks = [early_stopping]
                      )

test_set = test_datagen.flow_from_directory(
    r'./partial-food-dataset-splitted/test',
    target_size=(256, 256),
    batch_size=64,
    class_mode='categorical',
    shuffle=False
)

classifier.save('final_batch128.h5')

predictions = classifier.predict(test_set, steps=len(test_set))

# Evaluate the model on the test set
evaluation_results = classifier.evaluate(test_set)
print(f'Test Loss: {evaluation_results[0]}, Test Accuracy: {evaluation_results[1]}, F1: {evaluation_results[2]}, Precision: {evaluation_results[3]}, Recall: {evaluation_results[4]}')

# Plot training and validation accuracy
accuracy = hist.history['accuracy']
val_accuracy = hist.history['val_accuracy']
epochs = range(1, len(accuracy) + 1)

plt.ylim([0, 1])
plt.plot(epochs, accuracy, 'g', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot training and validation accuracy
accuracy = hist.history['f1_m']
val_accuracy = hist.history['val_f1_m']
epochs = range(1, len(accuracy) + 1)

plt.ylim([0, 1])
plt.plot(epochs, accuracy, 'g', label='Training f1')
plt.plot(epochs, val_accuracy, 'b', label='Validation f1')
plt.title('Adding ')
plt.xlabel('Epochs')
plt.ylabel('f1_m')
plt.legend()
plt.show()

# Plot training and validation accuracy
loss = hist.history['loss']
val_loss = hist.history['val_loss']
epochs = range(1, len(accuracy) + 1)

plt.ylim([0, 15])
plt.plot(epochs, loss, 'g', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Assuming you have more than two classes, use argmax to get predicted class indices
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
Y_pred = classifier.predict_generator(test_set)
predicted_classes = np.argmax(Y_pred, axis=1)

# Print confusion matrix
print('Confusion Matrix')
print(confusion_matrix(test_set.classes, predicted_classes))

# Print classification report
target_names = ['apple_pie', 'Baked Potato', 'burger', 'butter_naan', 'chai', 'chapati', 'cheesecake', 'chicken_curry', 'chole_bhature', 'Crispy Chicken', 'dal_makhani', 'dhokla', 'Donut', 'fried_rice', 'Fries', 'Hot Dog', 'ice_cream', 'idli', 'jalebi', 'kaathi_rolls', 'kadai_paneer', 'kulfi', 'masala_dosa', 'momos', 'omelette', 'paani_puri', 'pakode', 'pav_bhaji', 'pizza', 'samosa', 'Sandwich', 'sushi', 'Taco', 'Taquito']
print('Classification Report')
print(classification_report(test_set.classes, predicted_classes, target_names=target_names))

from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(test_set.classes, predicted_classes)
cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
plt.figure(figsize=(15,15))
plt.title('Food Confusion Matrix')
sns.heatmap(cmn, cmap="Blues", annot=True, fmt=".2f", linewidths=.5, xticklabels=target_names, yticklabels=target_names)