'''
link for dataset : https://www.kaggle.com/datamunge/sign-language-mnist#amer_sign2.png
'''
#sign language recognition
import tensorflow as tf
import keras
import keras.layers as k
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.initializers import glorot_uniform

#importing train and test data
train_data = pd.read_csv('sign_dataset/sign_mnist_train.csv')
test_data = pd.read_csv('sign_dataset/sign_mnist_test.csv')

from keras.models import Model
#CNN model
def sign_model(X_shape = (28,28,1)):
    X_input = k.Input(shape = X_shape)
    x = k.Conv2D(64,(3,3),activation = 'relu',kernel_initializer = glorot_uniform())(X_input)
    x = k.BatchNormalization()(x)
    x = k.Conv2D(64,(3,3),activation = 'relu',kernel_initializer = glorot_uniform())(x)
    x = k.BatchNormalization()(x)
    x = k.MaxPooling2D(pool_size = (2,2))(x)
    
    x = k.Conv2D(128,(3,3),activation = 'relu',kernel_initializer = glorot_uniform())(x)
    x = k.BatchNormalization()(x)
    x = k.Conv2D(128,(3,3),activation = 'relu',kernel_initializer = glorot_uniform())(x)
    x = k.BatchNormalization()(x)
    x = k.MaxPooling2D(pool_size = (2,2))(x)
    
    x = k.Flatten()(x)
    x = k.BatchNormalization()(x)
    x = k.Dense(256,activation = 'relu')(x)
    x = k.BatchNormalization()(x)
    x = k.Dense(26,activation = 'softmax')(x)
      
    model = Model(inputs = X_input,outputs = x,name = 'sign_model')
    return model


class myCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch,logs = {}):
        if logs.get('val_accuracy') > 0.995:
            print('reached 99.5% accuracy,so stop training!!')
            self.model.stop_training = True

#separating the labels into train_label and image pixel into train_set
train_label = train_data['label'].values
del train_data['label']
train_set = train_data.values
train_set = train_set.reshape(-1,28,28,1)

#separating the labels into test_label and image pixel into test_set
test_label = test_data['label'].values
del test_data['label']
test_data = test_data.values
test_data = test_data.reshape(-1,28,28,1)

from keras.preprocessing.image import ImageDataGenerator
#spilliting the data into 80% training and 20% validation
train_gen = ImageDataGenerator(rescale = 1/255,horizontal_flip = True,
                               shear_range = 0.1,zoom_range = 0.2,
                               width_shift_range = 0.1,height_shift_range = 0.1,
                               rotation_range = 20,validation_split = 0.2)
training_set = train_gen.flow(train_set,train_label,subset = 'training')
validation_set = train_gen.flow(train_set,train_label,subset = 'validation')

test_gen = ImageDataGenerator(rescale = 1/255)
test_set = test_gen.flow(test_data,test_label)


model = sign_model(X_shape = (28,28,1))
#to show The details od CNN
model.summary()
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy',metrics = ['accuracy'])

            
callbacks = myCallback()

history = model.fit(training_set,epochs = 25,validation_data = validation_set,verbose = 1,callbacks = [callbacks])

#plotting graphs
val_loss = history.history['val_loss']
val_accuracy = history.history['val_accuracy']
loss = history.history['loss']
accuracy = history.history['accuracy']

#training and validation acuracy
plt.plot(val_accuracy,label = 'validation accuracy')
plt.plot(accuracy,label = 'training accuracy')
plt.title('training and validation acuracy')
plt.legend()
plt.show()
#training and validation loss
plt.plot(val_loss,label = 'validation loss')
plt.plot(loss,label = 'training loss')
plt.title('training and validation loss')
plt.legend()
plt.show()
#loss and accuracy on test set
loss,accuracy = model.evaluate_generator(test_set)
print('loss on test set :',loss)
print('accuracy on test set :',accuracy)


