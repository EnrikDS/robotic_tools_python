import tensorflow as tf
import os
import numpy as np


models_saved_dir = os.getcwd()+"/saved_models/"
new_model = tf.keras.models.load_model(models_saved_dir+'sqrt_model/')
# Check its architecture
new_model.summary()
#Make predictions
test_numbers = np.array([1, 4, 9, 16])
print(new_model.predict(test_numbers))