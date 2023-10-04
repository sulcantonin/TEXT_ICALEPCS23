# Textual Analysis of ICALEPCS and IPAC Conference Proceedings: Revealing Research Trends, Topics, and Collaborations for Future Insights and Advanced Search
## Abstract
In this paper, we show a textual analysis of past ICALEPCS and IPAC conference proceedings to gain insights into the research trends and topics discussed in the field. We use natural language processing techniques to extract meaningful information from the abstracts and papers of past conference proceedings. We extract topics to visualize and identify trends, analyze their evolution to identify emerging research directions and highlight interesting publications based solely on their content with an analysis of their network. Additionally, we will provide an advanced search tool to better search in the existing papers to prevent duplication and easier reference findings. Our analysis provides a comprehensive overview of the research landscape in the field and helps researchers and practitioners to better understand the state-of-the-art and identify areas for future research. 

Keywords: research trends, search, conference proceedings, natural language processing, topic modeling

## Data

Source corpus data is available at https://huggingface.co/datasets/sulcan/TEXT_ICALEPCS23

## Semantic Search Tool

### SimCSE
SimCSE finetuning https://arxiv.org/abs/2104.08821, partially based on sample script https://github.com/UKPLab/sentence-transformers/blob/master/examples/unsupervised_learning/SimCSE/train_simcse_from_file.py, trained our dataset

To get the SimCSE pretrained weights, run following commands
```bash
# Make sure you have git-lfs installed (https://git-lfs.com)
git lfs install
git clone https://huggingface.co/sulcan/TEXT_ICALEPCS
```


#### Search
Trained moddel shown in available at [https://huggingface.co/sulcan/TEXT_ICALEPCS/tree/main/simcse](https://huggingface.co/sulcan/TEXT_ICALEPCS/tree/main/simcse)


To search through all papers (abstracts):

```python
TBD
```

#### Fine-Tuning
The model was trained with script [simcse_train.py](simcse_train.py), see paper for exact parameters. 

### Word2Vec
Word embedding for all unique tokens was found with [Gensim 4.1.2](https://github.com/RaRe-Technologies/gensim), 

To search through all papers (abstracts):
```python
TBD
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
