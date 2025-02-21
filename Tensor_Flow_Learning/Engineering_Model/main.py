import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras 
from keras import layers, models
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

# load the files
current_dir = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(current_dir, 'Civil.txt'), 'r') as file:
    civil_eng = file.read()

with open(os.path.join(current_dir,'Computers.txt'), 'r') as file:
    cse_eng = file.read()

with open(os.path.join(current_dir,'Electrical.txt'), 'r') as file:
    elect_eng = file.read()

with open(os.path.join(current_dir,'Mechanical.txt'), 'r') as file:
    mech_eng = file.read()


# Creating the dataframe
df = pd.DataFrame({
    'text': [civil_eng, cse_eng, elect_eng, mech_eng],
    'label':['civil','computers','electrical','mechanical']
})


# Label the data
label_map = {'civil':0,'computers':1, 'electrical':2, 'mechanical':3}
df['label'] = df['label'].map(label_map)


tokenizer = Tokenizer(num_words=15000, oov_token='<OOV>')
tokenizer.fit_on_texts(df['text'])

sequences = tokenizer.texts_to_sequences(df['text'])
padded_sequences = pad_sequences(sequences, padding='post')

X = np.array(padded_sequences)
y = np.array(df['label'])

# Split the data
train_sequences, test_sequences, train_labels, test_labels = train_test_split(
    X,y, test_size=0.2,  random_state=42)

# Build the model

model = models.Sequential([
    layers.Embedding(input_dim=15000, output_dim=16, input_length=padded_sequences.shape[1]),
    layers.GlobalAveragePooling1D(),
    layers.Dense(64, activation='relu'),
    layers.Dense(4, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(padded_sequences, df['label'],epochs=10)

loss, accuracy = model.evaluate( test_sequences,test_labels)
print(f'Test Accuracy: {accuracy}')