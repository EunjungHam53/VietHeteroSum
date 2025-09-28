import numpy as np
import torch
import torch.nn.functional as F
from bert_score import score as bert_score
import random
import os
from rouge import Rouge
from pyvi import ViTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize Rouge scorer
rouge = Rouge()

def get_rouge(ref, pred, use_stemmer=True):
    """Get Rouge scores for a single reference-prediction pair"""
    try:
        # Tokenize with ViTokenizer if use_stemmer is True
        if use_stemmer:
            ref_processed = ViTokenizer.tokenize(ref.replace('_', ' ')).lower()
            pred_processed = ViTokenizer.tokenize(pred.replace('_', ' ')).lower()
        else:
            ref_processed = ref.lower()
            pred_processed = pred.lower()
        
        scores = rouge.get_scores(pred_processed, ref_processed)[0]
        
        # Convert to the original format
        score_dict = {
            "rouge1": type('obj', (object,), {
                'precision': scores['rouge-1']['p'],
                'recall': scores['rouge-1']['r'],
                'fmeasure': scores['rouge-1']['f']
            })(),
            "rouge2": type('obj', (object,), {
                'precision': scores['rouge-2']['p'],
                'recall': scores['rouge-2']['r'],
                'fmeasure': scores['rouge-2']['f']
            })(),
            "rougeL": type('obj', (object,), {
                'precision': scores['rouge-l']['p'],
                'recall': scores['rouge-l']['r'],
                'fmeasure': scores['rouge-l']['f']
            })()
        }
        return score_dict
    except ValueError:
        # Return zero scores if calculation fails
        zero_score = type('obj', (object,), {'precision': 0.0, 'recall': 0.0, 'fmeasure': 0.0})()
        return {
            "rouge1": zero_score,
            "rouge2": zero_score,
            "rougeL": zero_score
        }

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
    """Calculate Rouge scores for multiple reference-prediction pairs"""
    rouge_scores = {"rouge1": {"p": [], "r": [], "f": []},
                    "rouge2": {"p": [], "r": [], "f": []},
                    "rougeL": {"p": [], "r": [], "f": []}}

    for golden, predict in zip(goldens, predicts):
        scores = get_rouge(golden, predict)
        for rouge_type in ["rouge1", "rouge2", "rougeL"]:
            rouge_scores[rouge_type]["p"].append(getattr(scores[rouge_type], 'precision'))
            rouge_scores[rouge_type]["r"].append(getattr(scores[rouge_type], 'recall'))
            rouge_scores[rouge_type]["f"].append(getattr(scores[rouge_type], 'fmeasure'))

    rouge_means = {rouge_type: {metric: sum(values) / len(values) if values else 0.0
                                for metric, values in metrics.items()}
                   for rouge_type, metrics in rouge_scores.items()}

    return rouge_means

# Additional Rouge calculation functions from the new code
def getRouge2(ref, pred, kind):  # tokenized input
    try:
        ref_processed = ViTokenizer.tokenize(ref.replace('_', ' ')).lower()
        pred_processed = ViTokenizer.tokenize(pred.replace('_', ' ')).lower()
        return round(rouge.get_scores(pred_processed, ref_processed)[0]['rouge-2'][kind], 4)
    except ValueError:
        return 0.0

def getRouge1(ref, pred, kind):  # tokenized input
    try:
        ref_processed = ViTokenizer.tokenize(ref.replace('_', ' ')).lower()
        pred_processed = ViTokenizer.tokenize(pred.replace('_', ' ')).lower()
        return round(rouge.get_scores(pred_processed, ref_processed)[0]['rouge-1'][kind], 4)
    except ValueError:
        return 0.0

def getRougeL(ref, pred, kind):  # tokenized input
    try:
        ref_processed = ViTokenizer.tokenize(ref.replace('_', ' ')).lower()
        pred_processed = ViTokenizer.tokenize(pred.replace('_', ' ')).lower()
        return round(rouge.get_scores(pred_processed, ref_processed)[0]['rouge-l'][kind], 4)
    except ValueError:
        return 0.0

def cal_rouge(goldens, predicts, avg=True):
    """Calculate Rouge scores using ViTokenizer preprocessing"""
    ppredicts = list(map(lambda x: ViTokenizer.tokenize(x.replace('_', ' ')).lower(), predicts))
    pgoldens = list(map(lambda x: ViTokenizer.tokenize(x.replace('_', ' ')).lower(), goldens))
    
    scores = rouge.get_scores(ppredicts, pgoldens, avg=avg)
    if not avg:
        return scores
    return list(map(
        lambda x: round(x, 4),
        (
            scores['rouge-1']['p'], scores['rouge-1']['r'], scores['rouge-1']['f'],
            scores['rouge-2']['p'], scores['rouge-2']['r'], scores['rouge-2']['f'],
            scores['rouge-l']['p'], scores['rouge-l']['r'], scores['rouge-l']['f'],
        )
    ))

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

def get_cosine_similarity(text1, text2):
    try:
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform([text1, text2])
        cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
        return cosine_sim[0][0]
    except:
        return 0

if __name__ == '__main__':
    # Test with Vietnamese text
    reference = 'Con cáo nâu nhanh nhẹn nhảy qua chú chó lười biếng'
    predict = 'Con cáo nâu nhanh chóng nhảy lên khúc gỗ'
    
    # Test individual Rouge scores
    scores = get_rouge(reference, predict, use_stemmer=True)
    print(f"Rouge-1 F1: {r1f(scores)}")
    print(f"Rouge-2 F1: {r2f(scores)}")
    print(f"Rouge-L F1: {rlf(scores)}")
    
    # Test batch Rouge calculation
    goldens = [reference]
    predicts = [predict]
    batch_scores = cal_rouge(goldens, predicts)
    print(f"Batch Rouge scores: {batch_scores}")
    
    # Test BERT score
    bert_scores = get_bert_score([reference], [predict], lang="en")
    print(f"BERT F1: {bertf(bert_scores)}")