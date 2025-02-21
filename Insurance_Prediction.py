import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras 
from keras import layers, models, optimizers

# loading the dataset
dataset = pd.read_csv('insurance_data.csv')

#Normalise the dataset
dataset['age'] = dataset['age']/100.0

train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

train_ages = train_dataset['age'].values
train_labels = train_dataset['insurance_status'].values
test_ages = test_dataset['age'].values
test_labels = test_dataset['insurance_status'].values

#Build the Model
model = models.Sequential([
    layers.Dense(10, activation='relu', input_shape=(1,)),
    layers.Dense(1, activation='sigmoid')
])

#Compile the Model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_ages, train_labels, epochs=10)


#Evaluate
test_loss, test_acc = model.evaluate(test_ages,test_labels)
print('\nTestAccuracy: ', test_acc)

#Predict for a certain age

age_to_predict = 50/100.0 
prediction = model.predict(np.array([age_to_predict]))
print('\nPrediction for age 50: ', prediction)




