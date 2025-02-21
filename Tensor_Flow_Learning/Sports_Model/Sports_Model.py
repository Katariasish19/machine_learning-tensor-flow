import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, models
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

#Loading the text files

with open('Cricket.txt', 'r') as file:
    cricket_text = file.read()

with open('Soccer_Article.txt', 'r') as file:
    soccer_text = file.read()

with  open('Formula1_Article.txt', 'r') as file:
    formula1_text = file.read()

with open('Hockey_Article.txt', 'r') as file:
    hockey_text = file.read()

# Create a DataFrame
df = pd.DataFrame({'text': [cricket_text, soccer_text, formula1_text, hockey_text],
                   'label':['Cricket', 'Soccer', 'Formula1', 'Hockey']
})

'''
Convert labels to numerical values, as the  model requires numerical values for labels because they cannot process
categorial data directly
'''

label_map = {'Cricket':0, 'Soccer': 1, 'Formula1': 2, 'Hockey': 3}
df['label'] = df['label'].map(label_map)

'''

1. Tokenizer is used to  convert text into sequences of integers, 10000 specifies the max number of words to keep,
based on the frequency.

2. oov_token='<OOV>' is used to handle words that are not in the tokenizer's vocabulary. 
Any word not in the top 10,000 will be replaced with this token. This ensures that the model can handle 
unseen words during training and inference without errors.

'''

tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')
tokenizer.fit_on_texts(df['text'])

# Converts each texts into sequence of integers.This transforms the text data into a numerical format that the model can process.
sequences = tokenizer.texts_to_sequences(df['text'])


'''
Ensures that all sequences have the same length by adding zeros (padding) at the end (padding='post').
Neural networks require input sequences to be of uniform length for batch processing.
Padding ensures consistency in input shape.
'''
padded_sequences = pad_sequences(sequences, padding='post')

# Split the data
train_sequences, test_sequences, train_labels, test_labels = train_test_split(
    padded_sequences, df['label'], test_size=0.2, random_state=42)

# Building the model
model = models.Sequential([
    layers.Embedding(input_dim=10000, output_dim=16, input_length= padded_sequences.shape[1]), # 16 Dimensional vector
    layers.GlobalAveragePooling1D(), #This layer computes the average of all elements in each feature map. It reduces the dimensionality of the data.
    layers.Dense(64, activation='relu'),
    layers.Dense(4, activation='softmax')
])


# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(padded_sequences, df['label'], epochs=10)

# Evaluate the model
loss, accuracy = model.evaluate(test_sequences, test_labels)
print(f'Test Accuracy: {accuracy}')











