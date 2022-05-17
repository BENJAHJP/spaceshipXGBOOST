from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')
col1 =[]
col2 =[]

vec1 = model.encode(col1, convert_to_tensor=True)
vec2 = model.encode(col2, convert_to_tensor=True)

cosine_scores = util.cos_sim(vec1, vec2)

for i, (sent1, sent2) in enumerate(zip(vec1, vec2)):
    if cosine_scores[i][i] >= 0.5:
        label = "similar"
    else:
        label = "not similar"
    print()