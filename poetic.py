import random
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation, Dropout
from keras.optimizers import RMSprop

# importing the training dataset
filepath = tf.keras.utils.get_file('shakespeare.txt',
        'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

text = open(filepath, 'rb').read().decode(encoding='utf-8').lower()

# Converting the Text to numerical format
chars = sorted(set(text))
chars_to_nums = dict((c, i) for i, c in enumerate(chars))
nums_to_chars = dict((i, c) for i, c in enumerate(chars))

seq_length = 40
step_size = 3

sentences = []  # features
next_chars = [] # the generated ones
for i in range(0, len(text) - seq_length, step_size):
    sentences.append(text[i:i + seq_length])
    next_chars.append(text[i + seq_length])

# Initialize x and y outside of the loop
x = np.zeros((len(sentences), seq_length, len(chars)), dtype=np.bool_)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool_)

# Fill x and y
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, chars_to_nums[char]] = 1
    y[i, chars_to_nums[next_chars[i]]] = 1


model = tf.keras.models.load_model('shakespere.h5')

def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def generate_text(length, temperature):
    start_index = random.randint(0, len(text) - seq_length - 1)
    generated = ''
    sentence = text[start_index: start_index + seq_length]
    generated += sentence
    for i in range(length):
        x = np.zeros((1, seq_length, len(chars)))
        for t, char in enumerate(sentence):
            x[0, t, chars_to_nums[char]] = 1
        predictions = model.predict(x, verbose=0)[0]
        next_index = sample(predictions, temperature)
        next_char = nums_to_chars[next_index]

        generated += next_char
        sentence = sentence[1:] + next_char
    return generated
print (generate_text(300, 0.2))