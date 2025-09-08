import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from CiteHeteroSum_model.Loss import pairwise_distances
from utils import *
from torch.amp import autocast
from torch.amp import GradScaler

scaler = GradScaler('cuda')
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def concat_with_overlap(str1, str2):
    tokens1 = str1.split()
    tokens2 = str2.split()

    overlap_length = 0
    for i in range(1, min(len(tokens1), len(tokens2)) + 1):
        if tokens1[-i:] == tokens2[:i]:
            overlap_length = i

    merged_tokens = tokens1 + tokens2[overlap_length:]
    return ' '.join(merged_tokens)

def get_summary(scores, edus, max_word_num, edu_num, kappa):
    assert len(scores) == len(edus)
    ranked_score_idxs = torch.argsort(scores, dim=0, descending=True)

    wordCnt = 0
    summEduIDList = []
    for i in ranked_score_idxs:
        if wordCnt >= max_word_num and edu_num == 0:
            break
        elif edu_num > 0 and len(summEduIDList) == edu_num:
            break
        s = edus[i]

        replicated = False
        if scores.squeeze(0)[i].item() < 0:
            replicated = True

        for chosedID in summEduIDList:
            eval_score = get_rouge(edus[chosedID], s)
            if r2p(eval_score) >= kappa:
                replicated = True
                break
        if replicated: continue

        wordCnt += len(s.split(' '))
        summEduIDList.append(i)
    summEduIDList = sorted(summEduIDList)

    selected_edu = [s for i, s in enumerate(edus) if i in summEduIDList]
    predicted_summary = ""
    for edu in selected_edu:
        predicted_summary = concat_with_overlap(predicted_summary, edu)

    return predicted_summary

def train_e2e(train_dataloader, model, optimizer, loss_method, summary_max_word_num, edu_num, kappa):
    model[0].train()
    model[1].train()

    c_loss, s_loss, loss, batch_num = 0, 0, 0, 0
    print_epo = 20
    rouge2_score = []

    for i, data in enumerate(train_dataloader):
        batch_loss, bc_loss, bs_loss, scores = train_e2e_batch(data, model, optimizer, loss_method)
        loss += batch_loss
        c_loss += bc_loss
        s_loss += bs_loss
        batch_num += 1

        predicted_text = get_summary(scores[0], data.edus, summary_max_word_num, edu_num=edu_num, kappa=kappa)
        eval_score = get_rouge(data.golden, predicted_text)
        rouge2_score.append(r2f(eval_score))

        if i % print_epo == 0:
            print(f"Batch {i}, Loss: {loss / batch_num}")
            print(f"Batch {i}, C-Loss: {c_loss / batch_num}")
            print(f"Batch {i}, S-Loss: {s_loss / batch_num}")

    return loss / batch_num, np.mean(rouge2_score)

def val_e2e(val_dataloader, model, loss_method, summary_max_word_num, edu_num, kappa):
    model[0].eval()
    model[1].eval()

    loss, c_loss, s_loss = 0, 0, 0
    batch_num = 0
    rouge2_score = []

    all_summaries = []
    all_gt = []

    for i, data in enumerate(val_dataloader):
        cur_loss, c_loss_b, s_loss_b, scores = val_e2e_batch(data, model, loss_method)
        loss += cur_loss
        c_loss += c_loss_b
        s_loss += s_loss_b

        predicted_summary = get_summary(scores[0], data.edus, summary_max_word_num, edu_num=edu_num, kappa=kappa)
        all_gt.append(data.golden)
        all_summaries.append(predicted_summary)
        eval_score = get_rouge(data.golden, predicted_summary)
        rouge2_score.append(r2f(eval_score))
        batch_num += 1

    rouge2_score_mean = np.mean(rouge2_score)
    loss = loss / batch_num
    c_loss /= batch_num
    s_loss /= batch_num

    return rouge2_score_mean, loss, c_loss, s_loss


def train_e2e_batch(data_batch, model, optimizer, loss_method):
    with autocast('cuda'):
        c_model = model[0]
        s_model = model[1]

        optimizer.zero_grad()
        feature = data_batch.feature.unsqueeze(0)
        adj = data_batch.adj.unsqueeze(0)
        sect_num = data_batch.sect_num
        sent_num = data_batch.sent_num
        labels = data_batch.score_onehot.unsqueeze(0)
        labels_neg = data_batch.score_onehot_neg.unsqueeze(0)
        golden_vec = data_batch.golden_vec

        pg = c_model(feature.cuda(), adj.cuda(), sect_num, sent_num)
        x = s_model(pg.cuda(), adj.cuda(), sect_num, sent_num)

        x = torch.clamp(x.squeeze(-1), min=-700, max=700)
        s_loss = F.binary_cross_entropy_with_logits(x, labels.cuda(), pos_weight=torch.tensor(10).cuda())
        pg = pg.squeeze(0)

        pos_mask = torch.zeros(1, feature.shape[1])
        pos_mask[:, :-sect_num - sent_num] = labels
        pos_mask[:, -sect_num - sent_num:] = 1

        neg_mask = torch.zeros(1, feature.shape[1])
        neg_mask[:, :-sect_num - sent_num] = labels_neg
        neg_mask[:, -sect_num - sent_num:] = 1

        if torch.any(pos_mask):
            pos_distances, neg_distances = pairwise_distances(golden_vec.cuda(), pg, pos_mask.cuda(), neg_mask.cuda())
            c_loss = loss_method(pos_distances, neg_distances)
        else:
            c_loss = torch.tensor(0.0, device=device)

        loss = s_loss + 1. * c_loss
        
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    return loss.item(), c_loss.item(), s_loss.item(), torch.sigmoid(x.squeeze(-1))


def val_e2e_batch(data_batch, model, loss_method):
    with autocast('cuda'):
        c_model = model[0]
        s_model = model[1]
        feature = data_batch.feature.unsqueeze(0)
        adj = data_batch.adj.unsqueeze(0)
        sect_num = data_batch.sect_num
        sent_num = data_batch.sent_num
        labels = data_batch.score_onehot.unsqueeze(0)
        labels_neg = data_batch.score_onehot_neg.unsqueeze(0)
        golden_vec = data_batch.golden_vec

        with torch.no_grad():
            pg = c_model(feature.cuda(), adj.cuda(), sect_num, sent_num)
            x = s_model(pg.cuda(), adj.cuda(), sect_num, sent_num)

            pg = pg.squeeze(0)

            pos_mask = torch.zeros(1, feature.shape[1])
            pos_mask[:, :-sect_num - sent_num] = labels
            pos_mask[:, -sect_num - sent_num:] = 1

            neg_mask = torch.zeros(1, feature.shape[1])
            neg_mask[:, :-sect_num - sent_num] = labels_neg
            neg_mask[:, -sect_num - sent_num:] = 1

            if torch.any(pos_mask):
                pos_distances, neg_distances = pairwise_distances(golden_vec.cuda(), pg, pos_mask.cuda(), neg_mask.cuda())
                c_loss = loss_method(pos_distances, neg_distances)
            else:
                c_loss = torch.tensor(0.0, device=device)

            s_loss = F.binary_cross_entropy_with_logits(x.squeeze(-1), labels.cuda(), pos_weight=torch.tensor(10).cuda())

            loss = c_loss * 1. + s_loss
            scores = torch.sigmoid(x.squeeze(-1))

    return loss.item(), c_loss.item(), s_loss.item(), scores
