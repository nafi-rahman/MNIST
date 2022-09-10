
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

model.compile(optimizer='adam',loss = 'sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(x_train,y_train,epochs=10)

model.evaluate(x_test,y_test)

plt.matshow(x_test[0])
y_predicted = model.predict(x_test)
print(np.argmax(y_predicted[0]))