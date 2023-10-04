import re
from nltk.tokenize import word_tokenize
from tqdm import tqdm
from collections import defaultdict
gd = lambda x,i : x[list(x.keys())[i]]

def get_abstract(fitz_paper_list):
    
    paper = "\n".join([block[4] for block in fitz_paper_list if not block[4].startswith('<')]) # image
    result = re.findall(r'Abstract(.*?)(?:INTRODUCTION)',paper, re.DOTALL)
    if len(result) > 0:
        return result[0]
    return None

def get_abstracts(data, min_len = 128, max_len = 3072, word_dict = []):
    abstracts = {}
    for paper_key in tqdm(data):
        result = get_abstract(data[paper_key])
        if result:
            # sort to be an abstract, not too short 
            if len(result) <= max_len and len(result) >= min_len:             
                # some abstracts are messed, measure ratio of english words in them, remove 
                words = word_tokenize(result)
                en_words = sum([1 for w in words if w.lower() in word_dict])
                if en_words / len(words) > 0.5: # at least 25% words are probably english
                    result = re.sub(r'-\n','',result)
                    result = re.sub('\n',' ',result)
                    abstracts[paper_key] = result
                else:
                    continue
    return abstracts


def get_titles(data : dict):
    titles = defaultdict(lambda : None)
    for k in tqdm(data):
        if len(data[k]) > 0:
            title = data[k][0][4]
            title = re.sub('\s+', ' ', title)
            title = re.sub('[^A-Z\-\s]', '', title)
            tokens = word_tokenize(title)
            num_of_more_than_one_letter_tokens = sum([1 for token in tokens if len(token) > 1])
            # at least half of all tokens is more than one letter
            if 2 * num_of_more_than_one_letter_tokens > len(tokens):
                titles[k] = title
    # pprint(list(zip(titles.keys(), titles.values()))[:100])
    return titles 

def process_references_of_a_paper(text : str):
    references = re.split('\[\d+\]',text)
    references = [re.findall(r"“(.*)”",text)[0] for text in references if re.findall(r"“(.*)”",text)]
    references = [text for text in references if len(text) > 0]
    return references

def get_references(data : dict):
    references = defaultdict(lambda : None)

    for k in tqdm(data):
        if len(data[k]) > 0:
            text = "\n".join([d[4] for d in data[k]])
            # print(text)
            references_start = text.find('REFERENCES')
            text = text[references_start:]
            text = re.sub(r'\s+', ' ', text)
            text = re.sub('<image:.*>','', text)
            references[k] = process_references_of_a_paper(text)
    return references

def remove_path_prefix_in_key(data_dict : dict, prefix : str):
    '''
    this function removes prefixes from the keys containing the path
    '''
    
    return {re.sub(prefix, '', k) : data_dict[k] for k in data_dict}

def replace_key_with_title(data_dict : dict, title_dict : dict):
    '''
    this function takes every key and replaces it with a title (the value of title_dict)
    keys of data_dict and title_dict must be same
    '''
    replaced = {}
    for k in data_dict:
        if k in title_dict:
            replaced[title_dict[k]] = data_dict[k]
    return replaced