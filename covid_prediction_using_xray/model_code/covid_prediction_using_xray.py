import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.layers import *
from keras.models import *
from keras.preprocessing import image

Train_Path="CovidDataset/Train"
Val_Path="CovidDataset/Val"

model=Sequential()
model.add(Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=(224,224,3)))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(.25))

model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(.25))

model.add(Conv2D(128,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(.25))

model.add(Flatten())
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss=keras.losses.binary_crossentropy, optimizer='adam', metrics=['accuracy'])

model.summary()

train_datagen= image.ImageDataGenerator(
    rescale=1/255.,
    shear_range=.2,
    zoom_range=.2,
    horizontal_flip=True,
)

test_dataset=image.ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    Train_Path,
    target_size=(224,224),
    batch_size=32,
    class_mode='binary')

train_generator.class_indices

val_generator = train_datagen.flow_from_directory(
    Val_Path,
    target_size=(224,224),
    batch_size=32,
    class_mode='binary')

history=model.fit_generator(
    train_generator,
    steps_per_epoch=8,
    epochs=10,
    validation_data=val_generator,
    validation_steps=2
)

model.save('model.h5')

model2=load_model('model.h5')

model2.summary()

