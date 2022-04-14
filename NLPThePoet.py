from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Embedding, GlobalAveragePooling1D, Dense, Bidirectional, LSTM
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

sentences = [
                "new, lighter iphone hailed by exhausted, humpbacked iphone 4 users",
                "ohio police chief: senseless killings by cops 'making us all look bad'",
                "eye surgery lets abused dog see his rescuer for the very first time",
                "nation unsure which candidate's plan to destroy the environment will create more jobs",
                "wild-eyed sears ceo convinced these the flannel pajama pants that will turn everything around",
                "new facebook feature allows user to cancel account",
                "jimmy fallon six tantalizing months from disappearing forever",
                "determined ant requires second flicking",
                "comey memoir claims trump was obsessed with disproving 'pee tape' allegation",
                "nation's sanitation workers announce everything finally clean",
                "new law determines bullets no longer responsibility of owner once fired from gun",
                "candy purchase puts yet more money in raisinets' bloated coffers",
                "bunch of numbers from where daddy works means no trip to disney world",
                "jennifer garner makes first public appearance since ben affleck split",
                "africa is inspiring these chinese transplants to reflect on their culture",
                "why erlich on 'silicon valley' is the best and the worst",
                "exhausted florida resident returns home after weathering harrowing week with family out of state",
                "senate can't pass methane rollback so interior decides to do it anyway",
                "why the deadly attacks against foreigners in south africa come as no surprise",
                "immigration backlash at the heart of british push to leave the e.u.",
                "area ladder never thought it would end up a bookcase",
                "little pussy has to take phone call in other room",
                "romney: democrats lost because they weren't 'proud' enough of obama",
                "mosquitoes don't even need to bite us, study shows",
                "Don't act like a silly boy",
                "Be wise boy"
             ]

tokenizer = Tokenizer(num_words=100, oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)
training_words_index = tokenizer.word_index
training_sequences = tokenizer.texts_to_sequences(sentences)
padded_training_sequences = pad_sequences(training_sequences, padding='pre')
padded_training_sequences = np.array(padded_training_sequences)
# print(f"padded_training_sequences shape {padded_training_sequences}")
# print(f"padded_training_sequences shape {padded_training_sequences.shape}")
# print(f"y_train shape {sentences.shape}")

# create labels
labels = []
for i in range(1, len(padded_training_sequences)):
    label = padded_training_sequences[:i]
    labels.append(label)

# labels and features
labels = np.array(labels, dtype="object")

# print(f" this is X data {X}")
# print(f" this is y data {y}")
# vocab_size = len(data)
# embedding_dim = 240
# max_length = 15
#
# model = Sequential()
# model.add(Embedding(vocab_size, embedding_dim, input_length=max_length))
# model.add(Bidirectional(LSTM(150)))
# model.add(Dense(vocab_size, activation='softmax'))
#
# model.compile(loss='categorical_crossentropy',
#               optimizer='adam',
#               metrics=['accuracy']
#               )
#
# model.fit(X, y)
