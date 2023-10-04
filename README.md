# Textual Analysis of ICALEPCS and IPAC Conference Proceedings: Revealing Research Trends, Topics, and Collaborations for Future Insights and Advanced Search
## Abstract
In this paper, we show a textual analysis of past ICALEPCS and IPAC conference proceedings to gain insights into the research trends and topics discussed in the field. We use natural language processing techniques to extract meaningful information from the abstracts and papers of past conference proceedings. We extract topics to visualize and identify trends, analyze their evolution to identify emerging research directions and highlight interesting publications based solely on their content with an analysis of their network. Additionally, we will provide an advanced search tool to better search in the existing papers to prevent duplication and easier reference findings. Our analysis provides a comprehensive overview of the research landscape in the field and helps researchers and practitioners to better understand the state-of-the-art and identify areas for future research. 

Keywords: research trends, search, conference proceedings, natural language processing, topic modeling

## Data

To download data and trained models, run following commands
```bash
# Make sure you have git-lfs installed (https://git-lfs.com)
git lfs install
git clone https://huggingface.co/sulcan/TEXT_ICALEPCS
```

To load the downloaded dataset, you need an English word dictionary (*words_alpha.txt*) and *gzip* and *pickle* libraries to open the *text_public.pickle.gzip* file. We mostly work with abstracts, which can be extracted via function *core.get_abstracts*

```python
import pickle, gzip, core

# English dictionary, to eventually eliminate badly readable pdfs
with open('words_alpha.txt', 'r') as f:
         EN = f.read().split('\n')

with gzip.open('TEXT_ICALEPCS23/text_public.pickle.gzip','rb') as f:
         data = pickle.load(f)
         abstracts = core.get_abstracts(data,EN)
```

## Semantic Search Tool

You need following Python libraries 
```bash
pip install sentence_transformers==2.2.2 gensim==4.1.2 nltk==3.6.7
```


### SimCSE
SimCSE finetuning https://arxiv.org/abs/2104.08821.

Source code is partially based on sample script https://github.com/UKPLab/sentence-transformers/blob/master/examples/unsupervised_learning/SimCSE/train_simcse_from_file.py

#### Search
Simple demo from the paper

```python
# loads library
from sentence_transformers import SentenceTransformer
# loads the model weights
model = SentenceTransformer('TEXT_ICALEPCS/simcse')
# sample texts from paper
texts = ["DESY radio frequency cavities detuned.",
         "XFEL cavities detuned.",
         "My teeth have frequent cavities.",
         "Please tune radio at low frequency.",
         "DESY is following a radio at low volumes."]
# sentences transformed to the embedding
e = model.encode(texts)
print(e.shape) # (5, 768)
```
You can replace texts variable with arbitrary content, e.g. abstracts. 

#### Fine-Tuning
The model was trained with script [simcse_train.py](simcse_train.py), see paper for exact parameters. 

### Word2Vec
Word embedding for all unique tokens was found with [Gensim 4.1.2](https://github.com/RaRe-Technologies/gensim), 

To search through all papers (abstracts):
```python

```

## Topics
TBD
## Knowledge Extraction
TBD
### Citation Graph
TBD
### Bipartite Graph and Projected Graphs
TBD

## Citation

```
TBD
```
