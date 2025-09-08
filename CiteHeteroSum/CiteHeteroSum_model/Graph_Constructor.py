import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import *

def mask_to_adj(sect_sent_mask, sent_edu_mask):
    sect_sent_mask = np.array(sect_sent_mask)
    sent_edu_mask = np.array(sent_edu_mask)

    edu_num = sent_edu_mask.shape[1]
    sent_num = sent_edu_mask.shape[0]
    sect_num = sect_sent_mask.shape[0]

    adj = np.zeros((edu_num + sent_num + sect_num, edu_num + sent_num + sect_num))
    adj[-sent_num - sect_num:-sect_num, 0:-sent_num - sect_num] = sent_edu_mask
    adj[0:-sent_num - sect_num, -sent_num - sect_num:-sect_num] = sent_edu_mask.T

    for i in range(0, sect_num):
        sect_mask = sect_sent_mask[i]
        if sect_mask.ndim == 1:
            sect_mask = sect_mask.reshape((1, -1))
        elif sect_mask.ndim == 0:
            sect_mask = np.array([sect_mask])

        adj[edu_num:-sect_num, edu_num:-sect_num] += sect_mask * sect_mask.T

    adj[-sect_num:, -sent_num - sect_num:-sect_num] = sect_sent_mask
    adj[-sent_num - sect_num :-sect_num, -sect_num:] = sect_sent_mask.T
    adj[-sect_num:, -sect_num:] = 1

    for i in range(0, sent_num):
        sent_mask = sent_edu_mask[i]
        if sent_mask.ndim == 1:
            sent_mask = sent_mask.reshape((1, -1))
        elif sent_mask.ndim == 0:
            sent_mask = np.array([sent_mask])
        adj[:edu_num, :edu_num] += sent_mask * sent_mask.T

    return adj

class Graph:
    def __init__(self, edus, edu_vectors, sent_vectors, sec_vectors, scores, sect_sent_mask, sent_edu_mask, golden, golden_vec, threds):
        assert len(edu_vectors) == len(scores) == len(edus), "ERROR"
        self.sect_num = len(sect_sent_mask)
        self.sent_num = len(sent_edu_mask)

        self.adj = torch.from_numpy(mask_to_adj(sect_sent_mask, sent_edu_mask)).float()
        self.feature = torch.cat((torch.stack(edu_vectors), torch.stack(sent_vectors), torch.stack(sec_vectors)), dim=0)

        neg_thred = threds[0]
        pos_thred = threds[1]

        self.score = torch.from_numpy(np.array(scores)).float()
        self.score_onehot = (self.score >= pos_thred).float()
        self.score_onehot_neg = (self.score <= neg_thred).float()


        self.edus = np.array(edus)
        self.golden = golden
        self.golden_vec = golden_vec.unsqueeze(0)


def graph_construction(sample, args):
    sec_num = len(sample['section_list'])
    sent_num = len(sample['sent_list'])
    edu_num = sample['number_edus']

    edus = []
    eduVecs = []
    sentVecs = []
    secVecs = []
    scores = []

    sect_sent_mask = np.zeros((sec_num, sent_num))
    sent_edu_mask = np.zeros((sent_num, edu_num))

    sent_count = 0
    edu_count = 0
    for secidx, secID in enumerate(sample["section_list"]):
        secVecs.append(sample["units"][secID]["embedding"])

        sentidx = 0
        for sentID in sample["units"][secID]["children"]:
            sect_sent_mask[secidx, sentidx + sent_count] = 1
            sentVecs.append(sample["units"][sentID]["embedding"])

            for eduidx, eduID in enumerate(sample["units"][sentID]["children"]):
                sent_edu_mask[sentidx + sent_count, edu_count + eduidx] = 1
                edus.append(sample["units"][eduID]["text"])
                eduVecs.append(sample["units"][eduID]["embedding"])
                scores.append(sample["units"][eduID]['golden_label']['r2p_thres'])

            edu_count += len(sample["units"][sentID]["children"])
            sentidx += 1
        sent_count += sentidx

    label_data = sample["label"][0]
    label_embedding = sample["label_embedding"]

    tmp_graph = Graph(edus, eduVecs, sentVecs, secVecs, scores, sect_sent_mask, sent_edu_mask, label_data, label_embedding, args['triplet_threds'])
    return tmp_graph