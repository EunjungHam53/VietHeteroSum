from rouge_score import rouge_scorer
import numpy as np
import torch
import torch.nn.functional as F
from bert_score import score as bert_score
import random
import os

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
scorer_unstemer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_rouge(ref, pred, use_stemmer=True):
    if use_stemmer:
        score = scorer.score(ref, pred)
    else:
        score = scorer_unstemer.score(ref, pred)
    return score

def r1p(score): return score["rouge1"].precision

def r1r(score): return score["rouge1"].recall

def r1f(score): return score["rouge1"].fmeasure

def r2p(score): return score["rouge2"].precision

def r2r(score): return score["rouge2"].recall

def r2f(score): return score["rouge2"].fmeasure

def rlp(score): return score["rougeL"].precision

def rlr(score): return score["rougeL"].recall

def rlf(score): return score["rougeL"].fmeasure

def get_rouges(goldens, predicts):
    rouge_scores = {"rouge1": {"p": [], "r": [], "f": []},
                    "rouge2": {"p": [], "r": [], "f": []},
                    "rougeL": {"p": [], "r": [], "f": []}}

    for golden, predict in zip(goldens, predicts):
        scores = get_rouge(golden, predict)
        for rouge_type in ["rouge1", "rouge2", "rougeL"]:
            rouge_scores[rouge_type]["p"].append(scores[rouge_type].precision)
            rouge_scores[rouge_type]["r"].append(scores[rouge_type].recall)
            rouge_scores[rouge_type]["f"].append(scores[rouge_type].fmeasure)

    rouge_means = {rouge_type: {metric: sum(values) / len(values) if values else 0.0
                                for metric, values in metrics.items()}
                   for rouge_type, metrics in rouge_scores.items()}

    return rouge_means


def get_bert_score(refs, preds, lang="en"):
    P, R, F1 = bert_score(preds, refs, lang=lang, verbose=True)
    score = {
        "precision": P.mean().item(),
        "recall": R.mean().item(),
        "fmeasure": F1.mean().item()
    }
    return score
def bertp(score): return score["precision"]

def bertr(score): return score["recall"]

def bertf(score): return score["fmeasure"]


def tensor_similarity(h1: torch.Tensor, h2: torch.Tensor):
    """Calculate similarity between two sets of vectors."""
    h1 = F.normalize(h1, dim=1)
    h2 = F.normalize(h2, dim=1)
    return h1 @ h2.t()

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

def get_cosine_similarity(text1, text2):
    try:
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform([text1, text2])
        cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
        return cosine_sim[0][0]
    except:
        return 0

if __name__ == '__main__':
    scores = scorer.score('The quick brown fox jumps over the lazy dog',
                          'The quick brown dog jumps on the log.')
    print(scores)

    reference = 'The quick brown fox jumps over the lazy dog'
    predict =  'The quick brown dog jumps on the log.'
    scores = get_rouge(reference, predict, use_stemmer=True)
    print(r1f(scores))

    score = get_bert_score([reference], [predict])
    print(score)
    print(bertf(score))