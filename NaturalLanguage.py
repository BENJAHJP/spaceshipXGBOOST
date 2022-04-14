import keras
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences

sentences = ["first sentence", "second one", "third sentence"]
test_data = ["test sentence one", "test sentence two", "test sentence three"]
tokenizer = Tokenizer(num_words=100, oov_token='<OOV>')
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
test_seq = tokenizer.texts_to_sequences(test_data)

sequences = tokenizer.texts_to_sequences(sentences)

padded_text = pad_sequences(sequences)
print(word_index)
print(sequences)
print(test_seq)
print(padded_text)