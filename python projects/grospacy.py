import pandas as pd
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity

# model instance
nlp = spacy.load("en_core_web_lg")

# data loading
# df1_data = pd.read_csv("/home/b3njah/Downloads/gro dataset/gro_nlp_homework/source_1.csv")
# df2_data = pd.read_csv("/home/b3njah/Downloads/gro dataset/gro_nlp_homework/source_2.csv")

# data parsing
#
#
# def parse_data(dataframe, dataframe_value='name'):
#     dataframe['parsed'] = dataframe[dataframe_value].apply(nlp)
#     return dataframe['parsed']
#
#
# df1_data['parsed'] = parse_data(dataframe=df1_data)
#
# print(df1_data['parsed'])
#
# # data cleaning and normalization
#
#
# def lemmatize(text):
#     """
#     REMOVE STOP WORDS GET NOUNS AND THEIR LEMMAS
#     """
#     doc = nlp(text)
#     lemma_list = [str(token.lemma_).lower() for token in doc if token.is_alpha and token.text.lower() not in STOP_WORDS
#                   and token.pos_ == 'NOUN']
#     return lemma_list
#
#
# df1_data['proc'] = df1_data['parsed'].apply(lemmatize)
#
# preprocessed = pd.DataFrame({
#     'id': df1_data['id'],
#     'preprocessed': df1_data['proc']
# })
#
# preprocessed.to_csv('source1_proc.csv')


# df1_data = pd.read_csv("/home/b3njah/PycharmProjects/numpy/python projects/source1_proc.csv")
# df2_data = pd.read_csv("/home/b3njah/PycharmProjects/numpy/python projects/source2_proc.csv")
#
#
# def get_vectors(text):
#     tokens = nlp(text)
#     for word in tokens:
#         return word.vector
#
#
# df2_data['word_vectors'] = df2_data['preprocessed'].apply(get_vectors)
#
# vectors = pd.DataFrame({
#     'id': df2_data['id'],
#     'vectors': df2_data['word_vectors']
# })
#
# vectors.to_csv('vectors2.csv', index_label=False)

df1 = pd.read_csv("/home/b3njah/PycharmProjects/numpy/python projects/source1_proc.csv")
df2 = pd.read_csv("/home/b3njah/PycharmProjects/numpy/python projects/source2_proc.csv")


similarities = []
id1 = []
id2 = []
for i in range(df1.shape[0]):
    for k in range(df2.shape[0]):
        base = nlp(df1.preprocessed[i])
        compare = nlp(df2.preprocessed[k])
        similarity = base.similarity(compare)
        if similarity >= 0.94:
            similarities.append(similarity)
            id1.append(df1.id[i])
            id2.append(df2.id[k])

submission = pd.DataFrame({
    "source_1": id1,
    "source_2": id2
})

submission.to_csv("Benjahjp@gmail.com.csv", index=False)

