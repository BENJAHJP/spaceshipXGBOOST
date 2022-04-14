from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Embedding, GlobalAveragePooling1D, Dense, Bidirectional, LSTM
import numpy as np

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

labels = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1]
labels = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(sentences, labels, test_size=0.2)

tokenizer = Tokenizer(num_words=100, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)
training_words_index = tokenizer.word_index
training_sequences = tokenizer.texts_to_sequences(X_train)
padded_training_sequences = pad_sequences(training_sequences, padding='post')
padded_training_sequences = np.array(padded_training_sequences)
print(f"padded_training_sequences shape {padded_training_sequences}")
print(f"padded_training_sequences shape {padded_training_sequences.shape}")
print(f"y_train shape {y_train.shape}")

tokenizer.fit_on_texts(X_test)
test_words_index = tokenizer.word_index
testing_sequences = tokenizer.texts_to_sequences(X_test)
padded_testing_sequences = pad_sequences(testing_sequences, padding='post', maxlen=15)
padded_testing_sequences = np.array(padded_testing_sequences)
print(f"padded_testing_sequences {padded_testing_sequences}")
print(f"padded_testing_sequences {padded_testing_sequences.shape}")
print(f"y_test shape {y_test.shape}")
print(f"test words index {test_words_index}")

vocab_size = len(training_words_index)
embedding_dim = 126
max_length = 15
print(vocab_size)

model = Sequential()
model.add(Embedding(vocab_size, embedding_dim, input_length=max_length))
model.add(GlobalAveragePooling1D())
model.add(Dense(24, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy']
              )

num_epochs = 10
model.fit(padded_training_sequences, y_train,
          epochs=num_epochs,
          batch_size=10,
          validation_data=(padded_testing_sequences, y_test),
          verbose=2
          )

predict = model.predict(padded_testing_sequences)
