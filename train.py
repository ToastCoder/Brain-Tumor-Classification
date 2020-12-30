# BRAIN TUMOR CLASSIFICATION

# FILE NAME: train.py

# DEVELOPED BY: Vigneshwar Ravichandar

# TOPICS: Multi-Class Classification, Deep Learning, TensorFlow, Convolutional Neural Networks

# DISABLE TF DEBUGGING INFORMATION
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# IMPORTING REQUIRED LIBRARIES
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# IMPORTING IMAGES OF TRAINING SET
dir_train = './dataset/train'
datagen_train = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1./255.)
train_data = datagen_train.flow_from_directory(dir_train,
                                               target_size = (200,200), 
                                               color_mode = 'grayscale', 
                                               class_mode = 'binary', 
                                               batch_size = 10)

# IMPORTING IMAGES OF TEST SET
dir_test = './dataset/test'
datagen_test = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1./255.)
test_data = datagen_test.flow_from_directory(dir_test,
                                               target_size = (200,200), 
                                               color_mode = 'grayscale', 
                                               class_mode = 'binary', 
                                               batch_size = 10)

# FUNCTION FOR NEURAL NETWORK
def tumorModel():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D( input_shape = (200,200,1), activation = 'relu', filters = 32, kernel_size = (5,5) ))
    model.add(tf.keras.layers.MaxPool2D( pool_size = (3,3), strides = (3,3) ))
    model.add(tf.keras.layers.Dropout(rate= 0.25))
    model.add(tf.keras.layers.Conv2D( filters = 64, kernel_size = (5,5), activation = 'relu'))
    model.add(tf.keras.layers.MaxPool2D(pool_size = (3,3), strides = (3,3)))
    model.add(tf.keras.layers.Dropout(rate= 0.25))
    model.add(tf.keras.layers.Conv2D( filters = 128, kernel_size = (3,3), activation = 'relu'))
    model.add(tf.keras.layers.MaxPool2D(pool_size = (2,2), strides = (2,2)))
    model.add(tf.keras.layers.Dropout(rate= 0.25))
    model.add(tf.keras.layers.Conv2D( filters = 256, kernel_size = (3,3), activation = 'relu'))
    model.add(tf.keras.layers.MaxPool2D(pool_size = (2,2), strides = (2,2)))
    model.add(tf.keras.layers.Dropout(rate= 0.25))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(512, activation = 'relu'))
    model.add(tf.keras.layers.Dense(512, activation = 'relu'))
    model.add(tf.keras.layers.Dense(4, activation = 'softmax'))
    return model

ACC_THRESHOLD = 0.97
# CALLBACK CLASS
class Callback(tf.keras.callbacks.Callback): 
    def on_epoch_end(self, epoch, logs={}): 
        if(logs.get('accuracy') > ACC_THRESHOLD):   
            print("\nReached %2.2f%% accuracy, so stopping training!!" %(ACC_THRESHOLD*100))   
            self.model.stop_training = True

model = tumorModel()
callback = Callback()
# FITTING THE DATA AND TRAINING THE MODEL
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
history = model.fit(train_data, validation_data = test_data, epochs = 30,verbose = 1, batch_size = 1,callbacks = [callback])

# PLOTTING THE GRAPH FOR TRAIN-LOSS AND VALIDATION-LOSS
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

# SAVING THE TRAINED MODEL
tf.keras.models.save_model(model,'./model/tumor-model')

