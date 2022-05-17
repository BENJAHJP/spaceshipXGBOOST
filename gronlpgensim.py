import gensim.models
import pandas as pd
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import remove_stopwords

df1_data = pd.read_csv("/home/b3njah/Downloads/gro dataset/gro_nlp_homework/source_1.csv")
df2_data = pd.read_csv("/home/b3njah/Downloads/gro dataset/gro_nlp_homework/source_2.csv")

print(df1_data.head())
print(df2_data.head())
print(df2_data.shape)
print(df1_data.shape)

# data preprocess

df1 = df1_data.name.apply(simple_preprocess)
df2 = df2_data.name.apply(simple_preprocess)


print(df1)
print(df2)

# model

model = gensim.models.Word2Vec(window=10, min_count=1, workers=4)
model.build_vocab(df1, progress_per=100)

# train
model.train(df1, epochs=3, total_examples=model.corpus_count)

# model test
print("cattle", model.wv.most_similar("cattle"))
print("buffalo", model.wv.most_similar("buffalo"))
print("poultry", model.wv.most_similar("poultry"))
print("mammals", model.wv.most_similar("mammals"))

# model
print("similarity between buffalo and poultry", model.wv.similarity(w1="buffalo", w2="poultry"))
