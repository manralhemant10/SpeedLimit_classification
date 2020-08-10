Deep learning is used for classifying images by building a convolutional neural network (CNN). 

Library Used :
Keras(https://keras.io/):
The Keras library in Python makes it pretty simple to build a CNN. 

training.py:
this file is used for training the model with the help of keras and it gives us a trained model file in pickel format

predicting.py:
this file is used for predicting the output in real time , for which we use camera attached to raspberry pi(Picam) and after predicting it classifies the speed and output it using serial output.