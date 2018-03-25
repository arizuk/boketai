import tensorflow as tf
from tensorflow.python.keras.applications.vgg16 import VGG16
from tensorflow.python.keras.layers import Input

x = Input(shape=(224,224,3))
model = VGG16(
    weights='imagenet',
    include_top=False,
    input_tensor=x)

model.summary()