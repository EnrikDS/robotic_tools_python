import tensorflow as tf
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

logdir=os.getcwd()+"/logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

'Lets build the data input'

data_numbers = np.linspace(100, 200, 3000)
targets = np.sqrt(data_numbers)
FIRST_LAYER_SIZE = 500
SECOND_LAYER_SIZE = 200
THIRD_LAYER_SIZE = 50
LEARNING_RATE = 0.004
EPOCHS =300
#Define input layer (features)
inputs = tf.keras.Input(shape=(1,), dtype=tf.float32)

#Weights initialised
initial_weights = tf.keras.initializers.RandomUniform(-0.1, 0.1)

#Define first dense layer
dense1 = tf.keras.layers.Dense(FIRST_LAYER_SIZE, activation='relu',
                               kernel_initializer=initial_weights)(inputs)
#Second dense layer
dense2 = tf.keras.layers.Dense(SECOND_LAYER_SIZE, activation ='relu',
                               kernel_initializer=initial_weights)(dense1)

dense3 = tf.keras.layers.Dense(THIRD_LAYER_SIZE, activation ='relu',
                               kernel_initializer=initial_weights)(dense2)

#Output layer
output = tf.keras.layers.Dense(1, activation='relu',
                               kernel_initializer=initial_weights)(dense3)

#Define a functional model
model = tf.keras.Model(inputs=inputs, outputs=output)
#compile model
model.compile(optimizer=tf.optimizers.SGD(learning_rate=LEARNING_RATE),
    loss='mean_absolute_error')
#Train model
history = model.fit(data_numbers, targets, validation_split=0.20, epochs=EPOCHS,
                    callbacks=[
    tensorboard_callback])
#print results
print(model.summary())

#Evaluate the model with new data
data_test = np.linspace(1E2, 1E6, 10)
y_test = np.sqrt(data_test)
model_test = model.evaluate(data_test, y_test)
for layer in model.layers:
      try:
        print(layer.get_weights()[0]) # weights
        print(layer.get_weights()[1]) # biases
      except:
          pass

print(model.predict([1,2,4,9]))
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()
plot_loss(history)
