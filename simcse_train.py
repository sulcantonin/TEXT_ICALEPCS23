from sentence_transformers import SentenceTransformer, InputExample
from sentence_transformers import models, losses, util
from torch.utils.data import DataLoader
import torch, gzip, pickle, re
from pprint import pprint
import pandas as pd
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
from core import *
from unidecode import unidecode


IMG_TAG = re.compile('<image:.*>')
NEW_LINE = re.compile(r'-\n')
gd = lambda x,i : x[list(x.keys())[i]]

# Define your sentence transformer model using CLS pooling
data_folder = '../data_local/'
model_name = 'roberta-base'
num_epochs = 2
max_seq_length = 256
batch_size = 128
# model = SentenceTransformer(model_name, device = 'cuda')
# model_name = 'distilroberta-base'
word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device = 'cuda')

#
# PREPARING DATA
#
with gzip.open(data_folder + 'text.pickle.gzip','rb') as f:
    data = pickle.load(f)
    
if True:
    with open(data_folder + 'dictionaries/words_alpha.txt', 'r') as f:
        EN = f.read().split('\n')
        EN = set([w.lower() for w in EN if len(w) > 1])

    # not using book corpus
    abstracts = get_abstracts(data, word_dict = EN)
    train_sentences = list(abstracts.values())
else:
    data = {k : "".join([x_[4] for x_ in data[k] if not re.match(IMG_TAG,x_[4])]).lower() for k in data}
    data = {k : re.sub(NEW_LINE,'\n', data[k]) for k in data}
    train_sentences = list(data.values())
    data = {k : data[k].encode('ascii','ignore') for k in data}
    print(gd(data,-1))
# Putting on the masks on the sentences
# # 
train_data = [InputExample(texts=[s, s]) 
              for b in tqdm(train_sentences)
              for s in sent_tokenize(b)]

# Preparing the dataset
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

# MultipleNegativesRankingLoss: takes a pair (a,p) anochor and positive, which are modified (with dropout_mask), 
# and all other are negative exmaples 
train_loss = losses.MultipleNegativesRankingLoss(model)

if True:
    warmup_steps = int(num_epochs*len(train_dataloader)*0.1)
    print(warmup_steps)
    # Call the fit method
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=num_epochs,
        warmup_steps=warmup_steps,
        optimizer_params = {'lr': 1e-05},
        show_progress_bar=True)
    model.save('output/simcse-model')
