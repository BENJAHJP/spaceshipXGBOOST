import torch
import torch.nn as nn
import torch.nn.functional as F
from spacy.lang.en import STOP_WORDS
from torch.utils.data.dataset import T_co
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import spacy

nlp = spacy.load('en_core_web_sm')

df = pd.read_csv("/home/b3njah/Downloads/nlp/nlp-getting-started/train.csv")


def data_preparation(dataframe=df):
    dataframe = dataframe.dropna()
    dataframe['context'] = dataframe['keyword']+"."+dataframe['text']
    dataframe = dataframe.drop(labels=['keyword', 'text', 'location'], axis=1)
    dataframe = dataframe[['id', 'context', 'target']]
    return dataframe


df = data_preparation()

print(df.head())


def text_normalisation(text):
    doc = nlp(text)
    lemma_list = [str(token.lemma_).lower() for token in doc if token.is_alpha and token.text.lower() not in STOP_WORDS]
    return lemma_list


df['context'] = df['context'].apply(text_normalisation)

print(df['context'])


def tokenizer(text):
    pass


class WordData(Dataset):
    def __init__(self):
        pass

    def __len__(self):
        pass

    def __getitem__(self, index):
        pass

