import tensorflow as tf
tf.config.list_physical_devices('GPU')
from datetime import datetime
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_loss(history):
  fig=plt.subplot()
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  plt.xlabel('Epoch')
  plt.ylabel('Error')
  plt.legend()
  plt.grid(True)
  plt.show()

def plot_evaluation_of_data(x, real_y, predicted_y):
    fig = plt.subplot()
    plt.plot(x, real_y, label='SQRT')
    plt.plot(x, predicted_y, label='predicted_y')
    plt.xlabel('Number')
    plt.ylabel('SQRT')
    plt.legend()
    plt.grid(True)
    plt.show()

logdir=os.getcwd()+"/logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
models_saved_dir = os.getcwd()+"/saved_models/"

'Lets build the data input'
data_numbers = np.linspace(0,  10000, 2048)
targets = np.sqrt(data_numbers)
FIRST_LAYER_SIZE = 128
SECOND_LAYER_SIZE = 256
THIRD_LAYER_SIZE = 128
LEARNING_RATE = 0.003
EPOCHS =100

#Define input layer (features)
inputs = tf.keras.Input(shape=(1,), dtype=tf.float32)

#Weights initialised
initial_weights = tf.keras.initializers.RandomUniform(-0.05, 0.05)

#Define first dense layer
dense1 = tf.keras.layers.Dense(FIRST_LAYER_SIZE, activation='elu',
                               kernel_initializer=initial_weights, use_bias=True)(inputs)
#Second dense layer
dense2 = tf.keras.layers.Dense(SECOND_LAYER_SIZE, activation ='elu',
                               kernel_initializer=initial_weights)(dense1)

dense3 = tf.keras.layers.Dense(THIRD_LAYER_SIZE, activation ='elu',
                               kernel_initializer=initial_weights)(dense2)

#Output layer
output = tf.keras.layers.Dense(1, activation='elu',
                               kernel_initializer=initial_weights)(dense3)

#Define a functional model
model = tf.keras.Model(inputs=inputs, outputs=output)

#Compile model
model.compile(optimizer=tf.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='mean_absolute_error')
#Train model
history = model.fit(data_numbers, targets, validation_split=0.20, epochs=EPOCHS,
                    callbacks=[tensorboard_callback])

#print results
print(model.summary())

#Evaluate the model with new data
data_test = np.linspace(0, 50000, 1000)
y_test = np.sqrt(data_test)
model_test = model.evaluate(data_test, y_test)
predictions = model.predict(data_test)
plot_evaluation_of_data(data_test, y_test, predictions)

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()
plot_loss(history)

#Save model to use it later
tf.saved_model.save(model, models_saved_dir+'sqrt_model/' )
