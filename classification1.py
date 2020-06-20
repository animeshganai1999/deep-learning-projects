#Convoulational Neural Network
#part-1 Building The CNNs
#importing the required libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import  Dense

#Initialising The CNNs
classifier = Sequential()

#Step-1 Convolution
classifier.add(Conv2D(32,(3,3),padding="same",input_shape = (64,64,3),activation = 'relu'))

#Step -2 Poolong
classifier.add(MaxPooling2D(pool_size = (2,2)))

#Adding a second convoulation layer
classifier.add(Conv2D(32,(3,3),padding="same",activation = 'relu')) #Here we don't need to specify the input image dimenstion because it's already known by the keras from the 1st maxPooling stage

classifier.add(MaxPooling2D(pool_size = (2,2)))

#Step -3 Flattenning
classifier.add(Flatten())

#setp-4 Full connection
classifier.add(Dense(output_dim = 128,activation = 'relu')) #We already use flatten function so input_dim parameter for 1st hidden layer is already known
classifier.add(Dense(output_dim = 1,activation = 'sigmoid'))

#Compiling The cnn
classifier.compile(optimizer = 'adam',loss = 'binary_crossentropy',metrics = ['accuracy'])

#part --2 Fitting the CNN to The Images
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
                                    rescale=1./255,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory( 'dataset/training_set',
                                                        target_size=(64, 64),
                                                        batch_size=32,
                                                        class_mode='binary')

test_set = test_datagen.flow_from_directory( 'dataset/test_set',
                                                        target_size=(64, 64),
                                                        batch_size=32,
                                                        class_mode='binary')

classifier.fit_generator(
        training_set,
        steps_per_epoch=8000,
        epochs=25,
        validation_data=test_set,
        validation_steps=2000)

result = classifier.predict_generator(test_set,verbose = 1)
