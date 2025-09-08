import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils import  *
from Preprocessing.EDU_Extracting import load_model, inference as EDU_inference, extract_spans
from Preprocessing.EDU_Cleaning import filter_invalid_edus, model_error_fix, merge_edus

def edu_cleaning(sent, edus):
    edus = filter_invalid_edus(sent, edus)
    edus = model_error_fix(sent, edus)
    edus = merge_edus(edus)
    return edus

# File `rst_parser_EDU.torchsave` is available on GitHub: https://github.com/seq-to-mind/DMRST_Parser.
model_path = 'data/checkpoint/rst_parser_EDU.torchsave'
EDU_model_batch_size = 1
EDU_tokenizer, EDU_extract_model = load_model(model_path)
def extract_edus(input_sent):
    Test_InputSentences = [input_sent]
    input_sentences, all_segmentation_pred, all_tree_parsing_pred = EDU_inference(EDU_extract_model, EDU_tokenizer, Test_InputSentences, EDU_model_batch_size)
    return extract_spans(input_sentences[0], all_segmentation_pred[0])


def EDU_process(sample, create_extractive_label=True, is_contain_sentence=True):
    label = sample['label'][0]

    for part in [sample["document"], sample["citation"]]:
        tmp_units = dict()
        for sentID in part["sent_list"]:
            sent = part["units"][sentID]
            sent_text = sent["text"]
            # all_edu_embedding = []
            if create_extractive_label:
                sent_golden_score = get_rouge(label, sent_text)
                sent["golden_rouge"] =  {"2p": r2p(sent_golden_score), "fmean": (r1f(sent_golden_score) + r2f(sent_golden_score) ) /2}

            num_token_sent = len(sent_text.split())
            if num_token_sent <= 1:
                print(sample["ID"], "SENT ERROR")
                continue

            edu_texts =  extract_edus(sent["text"])
            if len(edu_texts) == 1:
                edu_texts = [sent["text"]]
            else:
                if is_contain_sentence: 
                    edu_texts.append(sent["text"])

            preID = sentID.split("|")[:-1]
            preID = "|".join(preID)
            edu_pos = 0

            for edu_text in edu_texts:
                edu_pos += 1
                eduID = "{}|{}".format(preID, edu_pos)
                if create_extractive_label:
                    edu_golden_score =  get_rouge(label, edu_text)

                edu_unit = {"ID": eduID,
                            "unit_type": "edu",
                            "is_citation": sent["is_citation"],
                            "parent": sentID,
                            "children": [],
                            "cite": None,
                            "embedding": None,
                            "text":  edu_text,
                            "cl_id": None,
                            "golden_rouge": {"2p": r2p(edu_golden_score), "fmean": (r1f(edu_golden_score) + r2f
                                (edu_golden_score) ) /2} if create_extractive_label else {},
                            }
                part["units"][sentID]["children"].append(eduID)
                tmp_units[eduID] = edu_unit

        for eduID in tmp_units:
            part["units"][eduID] = tmp_units[eduID]
    return sample