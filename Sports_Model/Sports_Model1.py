import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, models, optimizers, metrics
from keras_preprocessing.text import  Tokenizer
from keras_preprocessing.sequence  import pad_sequences
from sklearn.model_selection  import train_test_split

#Loading the text files
current_dir = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(current_dir,'Cricket.txt'), 'r') as file:
    cricket_text = file.read()

with open(os.path.join(current_dir,'Soccer_Article.txt'), 'r') as file:
    soccer_text = file.read()

with  open(os.path.join(current_dir,'Formula1_Article.txt'), 'r') as file:
    formula1_text = file.read()

with open(os.path.join(current_dir,'Hockey_Article.txt'), 'r') as file:
    hockey_text = file.read()

# Create a DataFrame
df = pd.DataFrame({'text': [cricket_text, soccer_text, formula1_text, hockey_text],
                    'label':['Cricket', 'Soccer', 'Formula1', 'Hockey']
})

# Convert labels to numerical values
label_map = {'Cricket':0, 'Soccer': 1, 'Formula1': 2, 'Hockey': 3}
df['label'] = df['label'].map(label_map)

tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')
tokenizer.fit_on_texts(df['text'])

sequences = tokenizer.texts_to_sequences(df['text'])

padded_sequences = pad_sequences(sequences, padding='post')


# Building the model
model = models.Sequential([
    layers.Embedding(input_dim=10000, output_dim=16, input_length= padded_sequences.shape[1]),
    layers.GlobalAveragePooling1D(),
    layers.Dense(64, activation='relu'),
    layers.Dense(4, activation='softmax')
])

# Split the data
train_sequences, test_sequences, train_labels, test_labels = train_test_split(
    padded_sequences, df['label'], test_size=0.2, random_state=42)

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(padded_sequences, df['label'], epochs=10)

# Evaluate the model
loss, accuracy = model.evaluate(test_sequences, test_labels)
print(f'Test Accuracy: {accuracy}')


# New text data
new_text = "This is a new article about Cristiano Ronaldo. It discusses the latest match results and player performances."

# Tokenize the new text
new_sequence = tokenizer.texts_to_sequences([new_text])

# Pad the sequence
new_padded_sequence = pad_sequences(new_sequence, padding='post', maxlen=padded_sequences.shape[1])

# Make a prediction
prediction = model.predict(new_padded_sequence)

# Get the predicted class
predicted_class = np.argmax(prediction, axis=1)[0]

# Map the predicted class to the label
label_map_reverse = {0: 'Cricket', 1: 'Soccer', 2: 'Formula1', 3: 'Hockey'}
predicted_label = label_map_reverse[predicted_class]

print(f'The predicted category for the new article is: {predicted_label}')











