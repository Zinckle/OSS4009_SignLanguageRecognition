import cv2
import mediapipe as mp
import time
import math
import numpy as np
from rembg import remove
from keras.datasets import mnist
from keras.src.utils import img_to_array
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Flatten
import tensorflow as tf
import visualkeras

model1 = tf.keras.models.load_model('smnist.h5')

visualkeras.layered_view(model1).show() # display using your system viewer

visualkeras.layered_view(model1, to_file='output.png') # write to disk
