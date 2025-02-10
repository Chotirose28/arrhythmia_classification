import tensorflow as tf
from Parameters import Parameters
from tensorflow.keras import layers
from tensorflow.keras.models import Model

initializer = tf.initializers.he_normal()

class CNN(Model):
    
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1 = layers.Conv2D(32,3,1,padding='same', activation='relu', kernel_initializer=initializer)
        self.conv2 = layers.Conv2D(32,3,1,padding='same', activation='relu', kernel_initializer=initializer)
        self.conv3 = layers.Conv2D(64,3,1,padding='same', activation='relu', kernel_initializer=initializer)
        self.conv4 = layers.Conv2D(64,3,1,padding='same', activation='relu', kernel_initializer=initializer)
        self.conv5 = layers.Conv2D(64,3,1,padding='same', activation='relu', kernel_initializer=initializer)
        self.conv6 = layers.Conv2D(64,3,1,padding='same', activation='relu', kernel_initializer=initializer)
        self.conv7 = layers.Conv2D(128,3,1,padding="same", activation='relu', kernel_initializer=initializer)
        self.conv8 = layers.Conv2D(128,3,1,padding="same", activation='relu', kernel_initializer=initializer)


        self.pool1 = layers.MaxPooling2D((2,2),strides=2,padding='same')
        self.pool2 = layers.MaxPooling2D((2,2),strides=2,padding='same')
        self.pool3 = layers.MaxPooling2D((2,2),strides=2,padding='same')
        self.pool4 = layers.MaxPooling2D((2,2),strides=2,padding="same")

        self.flatten = layers.Flatten()

        self.bn1 = layers.BatchNormalization(axis = 3)
        self.bn2 = layers.BatchNormalization(axis = 3)
        self.bn3 = layers.BatchNormalization(axis = 3)
        self.bn4 = layers.BatchNormalization(axis = 3)
        self.bn5 = layers.BatchNormalization(axis = 3)
        self.bn6 = layers.BatchNormalization(axis = 3)
        self.bn7 = layers.BatchNormalization(axis = 3)
        self.bn8 = layers.BatchNormalization(axis = 3)
        self.lbn1 = layers.BatchNormalization(axis = 1)
        self.lbn2 = layers.BatchNormalization(axis = 1)

        self.dense1 = layers.Dense(2048,activation='relu')
        self.dense2 = layers.Dense(512,activation='relu')
        self.dense3 = layers.Dense(Parameters.NumofClass,activation='softmax')

    def call(self,input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.pool2(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.conv6(x)
        x = self.bn6(x)
        x = self.pool3(x)
        x = self.conv7(x)
        x = self.bn7(x)
        x = self.conv8(x)
        x = self.bn8(x)
        x = self.pool4(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.lbn1(x)
        x = self.dense2(x)
        x = self.lbn2(x)
        x = self.dense3(x)

        return x