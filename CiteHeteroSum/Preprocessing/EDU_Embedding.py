import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer_embedding = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
model_embedding = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased").to(device)

def getPositionEncoding(pos, d=768, n=10000):
    P = np.zeros(d)
    for i in np.arange(int(d/2)):
        denominator = np.power(n, 2*i/d)
        P[2*i] = np.sin(pos/denominator)
        P[2*i+1] = np.cos(pos/denominator)
    return torch.tensor(P, dtype=torch.float32).to(device)

def phrase_embedding(phrase):
    token_embed_tuple = compute_word_embeddings(phrase)
    token_embeddings = torch.stack([val[1] for val in token_embed_tuple])
    edu_embedding = token_embeddings.mean(dim=0)
    return edu_embedding

def compute_word_embeddings(text):
    inputs = tokenizer_embedding(text, return_tensors="pt", truncation=True, max_length=512, padding=True).to(device)
    with torch.no_grad():
        outputs = model_embedding(**inputs)

    word_embeddings = outputs.last_hidden_state.squeeze(0)
    tokens = tokenizer_embedding.convert_ids_to_tokens(inputs['input_ids'].squeeze(0).to(device))
    token_embeddings = [(token, word_embeddings[idx]) for idx, token in enumerate(tokens)]
    return token_embeddings

