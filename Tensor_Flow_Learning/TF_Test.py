import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow import keras 
from keras import layers, models
from keras_preprocessing.text import  Tokenizer
from keras_preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

with open('..', 'Sports_Model', 'Cricket.txt', 'r') as file:
    cricket_article = file.read()

with open('..', 'Sports_Model', 'Soccer_Article.txt', 'r') as file:
    soccer_article = file.read()

with open('..', 'Sports_Model', 'Formula1_Article.txt', 'r') as file:
    formula1_article = file.read()

with open('..', 'Sports_Model', 'Hockey_Article.txt', 'r') as file:
    hockey_article = file.read()


df = pd.DataFrame({'text':[cricket_article, soccer_article,  formula1_article, hockey_article],
                    'label':['cricket', 'soccer', 'formula1', 'hockey']})

label_map = {'Cricket': 0, 'Soccer': 1, 'Formula1':2, 'Hockey':3}
df['label'] = df['label'].map(label_map)


tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')
tokenizer.fit_on_texts(df['text'])

sequences =  tokenizer.texts_to_sequences(df['text'])
padded_sequences = pad_sequences(sequences, padding='post')

# Split the data into training and testing sets
train_sequences , train_labels, test_sequences, test_labels = train_test_split(
    padded_sequences,  df['label'], test_size=0.2, random_state=42)


# Build the model

model = models.Sequential[(
    layers.Embedding(input_dim=10000, output_dim=16, input_length = padded_sequences.shape[1]),
    layers.GlobalAveragePooling1D(),
    layers.Dense(64, activation='relu'),
    layers.Dense(4, activation='softmax')
)]

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(padded_sequences, df['label'], epochs=10)

loss, accuracy = model.evaluate(test_sequences, test_labels)
print(f'Test Accuracy: {accuracy}')




