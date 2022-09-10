
import matplotlib
import tensorflow as tf
from tensorflow import keras 
import matplotlib.pyplot as plt
import numpy as np

# Load the data
(x_train,y_train),(x_test,y_test) = tf.keras.datasets.mnist.load_data()

#x_train has 60000 images of 28x28 pixels
#y_train has 60000 labels

#normalize the data so it is in a range from 0 to 1
x_train = x_train/255.0
x_test = x_test/255.0

#build the model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(100,activation='relu'),
    tf.keras.layers.Dense(10,activation='sigmoid')
])

#compile the model
model.compile(optimizer='adam',loss = 'sparse_categorical_crossentropy',metrics=['accuracy'])
#train the model
model.fit(x_train,y_train,epochs=10)
#evaluate the model / test the model with the test data
model.evaluate(x_test,y_test)

#predict a number using the trained model
y_predicted = model.predict(x_test)
#print the predicted number
print(np.argmax(y_predicted[0]))