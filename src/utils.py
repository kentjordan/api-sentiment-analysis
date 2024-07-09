import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

import preprocessor as p
import numpy as np
import torch
import re

nltk.download("punkt")
nltk.download("stopwords")

stop_words = set(stopwords.words('english'))
ps =  PorterStemmer()

def tokenize_text(text: str):
    text = p.clean(text) # Remove URLs, Mentions, Emojis, Smileys and Hashtags
    text = re.sub(r"[^\w]+", " ", text) # Remove Punctuation
    text = nltk.word_tokenize(text) # Tokenize into array
    text = [token.lower() for token in text if token.lower() not in stop_words] # Remove stop words, and make lower-case
    text = [ps.stem(token) for token in text] # Stem each token
    return text

def scale(scaling: str, pos_total: int, neg_total: int, pos_freqs: list[str], neg_freqs: list[str]):
    # Positive Frequency
    max_pos_freqs = max(pos_freqs)
    min_pos_freqs = min(pos_freqs)
    
    mean_pos_freqs = np.array(pos_freqs).mean()
    std_pos_freqs = np.array(pos_freqs).std()

    # Negative Frequency
    max_neg_freqs = max(neg_freqs)
    min_neg_freqs = min(neg_freqs)

    mean_neg_freqs = np.array(neg_freqs).mean()
    std_neg_freqs = np.array(neg_freqs).std()

    if scaling == 'normalize':
        pos_total = (pos_total - min_pos_freqs) / (max_pos_freqs - min_pos_freqs)
        neg_total =  (neg_total - min_neg_freqs) / (max_neg_freqs - min_neg_freqs)
    
    if scaling == 'standardize':
        pos_total = (pos_total - mean_pos_freqs) / std_pos_freqs
        neg_total = (neg_total - mean_neg_freqs) / std_neg_freqs
    
    if scaling == 'log':
        pos_total = np.log(pos_total + 1)
        neg_total = np.log(neg_total + 1)

    if scaling == 'sqrt':
        pos_total = np.sqrt(pos_total)
        neg_total = np.sqrt(neg_total)

    return pos_total, neg_total

def make_input(text: str, pos_freqs: list[str], neg_freqs: list[str], vocabulary: set[str], scaling: str):
    
    pos_total = 0
    neg_total = 0

    for token in tokenize_text(text):
        for j, vtoken in enumerate(vocabulary):
            if token == vtoken:
                pos_total += pos_freqs[j]
                neg_total += neg_freqs[j]
    
    pos_total, neg_total = scale(scaling=scaling, pos_total=pos_total, neg_total=neg_total, pos_freqs=pos_freqs, neg_freqs=neg_freqs)
    
    return torch.as_tensor([1, pos_total, neg_total]).to(dtype=torch.float32)

__all__ = ['tokenize_text', 'scale', 'make_input']