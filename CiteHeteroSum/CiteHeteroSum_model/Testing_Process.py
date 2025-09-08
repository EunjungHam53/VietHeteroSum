import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from CiteHeteroSum_model.Training_Process import get_summary
from torch.amp import autocast
from torch.amp import GradScaler

scaler = GradScaler('cuda')
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def test_e2e(val_dataloader, model, summary_max_word_num, edu_num, kappa):
    model[0].eval()
    model[1].eval()
    all_predicts = []
    all_gt = []

    for i, data in enumerate(val_dataloader):
        scores = test_e2e_batch(data, model)
        summary_text = get_summary(scores[0], data.edus, summary_max_word_num, edu_num, kappa)
        all_gt.append(data.golden)
        all_predicts.append(summary_text)

    return all_predicts, all_gt

def test_e2e_batch(data_batch, model):
    with autocast('cuda'):
        c_model = model[0]
        s_model = model[1]
        feature = data_batch.feature.unsqueeze(0)
        adj = data_batch.adj.unsqueeze(0)
        sect_num = data_batch.sect_num
        sent_num = data_batch.sent_num
        scores = data_batch.score.unsqueeze(0).float()

        with torch.no_grad():
            pg = c_model(feature.cuda(), adj.cuda(), sect_num, sent_num)
            x = s_model(pg.cuda(), adj.cuda(), sect_num, sent_num)
            scores = torch.sigmoid(x.squeeze(-1))

    return scores