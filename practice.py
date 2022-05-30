import pandas as pd
import spacy
from sklearn.metrics.pairwise import cosine_similarity

# model instance
nlp = spacy.load("en_core_web_lg")

df1 = pd.read_csv("/home/b3njah/Documents/data1.csv")
df2 = pd.read_csv("/home/b3njah/Documents/data2.csv")

print(df1['name'][1])

similarity = []
id1_name = []
id2_name = []
simi = []
id_1 = []
id_2 = []
for i in range(df1.shape[0]):
    for k in range(df2.shape[0]):
        base = nlp(df1.name[i])
        compare = nlp(df2.name[k])
        similar = base.similarity(compare)
        if similar > 0.5:
            simi.append("similar")
        else:
            simi.append("Not similar")
        id_1.append(df1.id[i])
        id_2.append(df2.id[k])
        similarity.append(similar)
        id1_name.append(df1['name'][i])
        id2_name.append(df2['name'][k])


submission = pd.DataFrame({
    "id1_name": id1_name,
    "id2_name": id2_name,
    "id_1":id_1,
    "id_2":id_2,
    "similarity":similarity,
    "similar":simi
})

submission.to_csv("sub.csv", index=False)
