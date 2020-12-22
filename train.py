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

# DEFINING THE NEURAL NETWORK
model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Conv2D( input_shape = (200,200,1),
                                  activation = 'relu',
                                  filters = 32,
                                  kernel_size = 3 ))

model.add(tf.keras.layers.MaxPool2D( pool_size = 2, strides = 2 ))

model.add(tf.keras.layers.Conv2D( filters = 64,
                                  kernel_size = 3,
                                  activation = 'relu'))

model.add(tf.keras.layers.MaxPool2D(pool_size = 2, strides = 2))

model.add(tf.keras.layers.Conv2D( filters = 128,
                                  kernel_size = 3,
                                  activation = 'relu'))

model.add(tf.keras.layers.MaxPool2D(pool_size = 2, strides = 2))

model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(256, activation = 'relu'))

model.add(tf.keras.layers.Dense(4, activation = 'softmax'))

# FITTING THE DATA AND TRAINING THE MODEL
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
history = model.fit(train_data, validation_data = test_data, epochs = 10,verbose = 1)

# SAVING THE TRAINED MODEL
tf.keras.models.save_model(model,'./model/tumor-model')

