import tensorflow_hub as hub
import tensorflow as tf


bert_encoder_url = "/home/b3njah/Documents/Bert/bert/bertmodel/bert_model/"
bert_preprocess_url = "/home/b3njah/Documents/Bert/bert/preprocess/bert_preprocess/"

bert_encoder = tf.saved_model.load(bert_encoder_url)
bert_preprocess = tf.saved_model.load(bert_preprocess_url)

# bert_encoder = hub.KerasLayer(bert_encoder_url)
# bert_preprocess = hub.KerasLayer(bert_preprocess_url)

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


def preprocess_embedding(sentence):
    preprocessed_sentence = bert_preprocess(sentence)
    output = bert_encoder(preprocessed_sentence)
    return output


print(preprocess_embedding(sentences))
