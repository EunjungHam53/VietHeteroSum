import torch
from tqdm import tqdm

def independence_citation_section(ID, doc, citation, input_sections=None):
    rs = dict()
    rs["units"] = dict()
    rs['section_list'] = []
    rs["sent_list"] = []
    rs_edu_total =  0
    rs["units"] = dict()

    for secID in doc["section_list"]:
        sec = doc["units"][secID]
        if sec["text"] in input_sections:
            rs['section_list'].append(secID)
            rs["units"][secID] = sec

            for sentID in sec["children"]:
                sent = doc["units"][sentID]
                if len(sent["children"]) == 0:
                    continue
                rs["sent_list"].append(sentID)
                rs["units"][sentID] = sent
                sent["children"].append(sentID)
                rs_edu_total += len(sent["children"])

                for eduID in sent["children"]:
                    rs["units"][eduID] = doc["units"][eduID]

    sec_cite_unit_ID =  "{}|{}|{}|{}".format(1, 0, 0, 0)
    sec_cite_unit = {"ID": sec_cite_unit_ID,
                     "unit_type":  "section",
                    "is_citation":True,
                    "parent": ID,
                    "children": [],
                    "cite": None,
                    "embedding": None,
                    "text": None,
                    "cl_id": None}

    all_sec_cite_sent_embeddings = []
    for old_secID in citation["section_list"]:
        old_sec = citation["units"][old_secID]

        for sentID in old_sec["children"]:
            sent = citation["units"][sentID]
            if len(sent["children"]) == 0:
                    continue
            rs["sent_list"].append(sentID)
            sent["children"].append(sentID)
            sent["parent"] =  sec_cite_unit_ID
            sec_cite_unit["children"].append(sentID)
            all_sec_cite_sent_embeddings.append(sent["embedding"])
            rs["units"][sentID] = sent
            rs_edu_total += len(sent["children"])

            for eduID in sent["children"]:
                rs["units"][eduID] = citation["units"][eduID]

    sec_cite_unit["embedding"] =  torch.stack(all_sec_cite_sent_embeddings).mean(dim=0)
    rs['section_list'].append(sec_cite_unit_ID)
    rs["units"][sec_cite_unit_ID] = sec_cite_unit

    return rs, rs_edu_total

def citation_filtering(sample, sim = [0.2, 0.3], length_range = [15, 30]):
    for sentID in sample["citation"]["sent_list"]:
        sent = sample["citation"]["units"][sentID]

        if not isinstance(sent.get("cite"), dict):
            continue

        if "cosine" not in sent["cite"] or "ltoken" not in sent["cite"]:
            sent["cite"] = "skip"
            continue

        top_sim = max(sent["cite"]["cosine"].values())
        length = sent["cite"]["ltoken"]

        if (sim is None or length_range is None) or (top_sim > sim[0] and top_sim <= sim[1] and length > length_range[0] and length < length_range[1]):
            top_cite = max(sent["cite"]["cosine"], key=sent["cite"]["cosine"].get)
            top_cite = "{}|{}|{}|{}".format(0, top_cite.split("|")[1], 1, 0)
            sent["cite"] = [top_cite]
        else:
            sent["cite"] = "skip"
    return sample


def merge_citation_to_doc_section(input_data, input_sections, have_label_embedding=False):
    doc = input_data["document"]
    citation = input_data["citation"]

    rs = dict()
    rs["ID"] = input_data["ID"]
    rs["label"] = input_data["label"]
    rs["units"] = dict()
    rs['section_list'] = []
    rs["sent_list"] = []
    rs["number_edus"] = 0

    if have_label_embedding:
        rs["label_embedding"] = input_data["label_embedding"]

    for secID in doc["section_list"]:
        sec = doc["units"][secID]

        if input_sections is not None:
            if sec["text"] not in input_sections: continue
            
        if len(sec["children"]) == 0:
            continue

        rs['section_list'].append(secID)
        rs["units"][secID] = sec

        for sentID in sec["children"]:
            sent = doc["units"][sentID]
            if len(sent["children"]) == 0:
                print("ERROR")
                continue

            rs["sent_list"].append(sentID)
            rs["units"][sentID] = sent
            rs["number_edus"] += len(sent["children"])
            for eduID in sent["children"]:
                rs["units"][eduID] = doc["units"][eduID]

    for citeID in citation["sent_list"]:
        csent = citation["units"][citeID]
        if len(csent["children"]) == 0 or csent["cite"] is None:
            print(citeID, "Citation ERROR")
            continue

        new_csent_children = []
        for eduID in csent["children"]:
            edu = citation["units"][eduID]
            if edu["cite"] != "skip":
                new_csent_children.append(eduID)

        csent["children"] = new_csent_children
        if len(new_csent_children) == 0:
            csent["cite"] = "skip"

        if csent["cite"] == "skip":
            continue

        for dsentID in csent["cite"]:
            dsent = doc["units"][dsentID]
            dsec = doc["units"][dsent["parent"]]
            if input_sections is not None:
                if dsec["text"] not in input_sections: continue

            csent["parent"] = dsec["ID"]
            dsec["children"].append(citeID)

            rs["sent_list"].append(citeID)
            rs["units"][citeID] = csent

            rs["number_edus"] += len(csent["children"])
            for eduID in csent["children"]:
                rs["units"][eduID] = citation["units"][eduID]

    return rs


def skip_merge_citation(input_data, input_sections, have_label_embedding=False):
    doc = input_data["document"]
    citation = input_data["citation"]

    rs = dict()
    rs["ID"] = input_data["ID"]
    rs["label"] = input_data["label"]
    rs["units"] = dict()
    rs['section_list'] = []
    rs["sent_list"] = []
    rs["number_edus"] = 0

    if have_label_embedding:
        rs["label_embedding"] = input_data["label_embedding"]

    for secID in doc["section_list"]:
        sec = doc["units"][secID]

        if input_sections is not None:
            if sec["text"] not in input_sections: continue

        if len(sec["children"]) == 0:
            continue

        rs['section_list'].append(secID)
        rs["units"][secID] = sec

        for sentID in sec["children"]:
            sent = doc["units"][sentID]
            if len(sent["children"]) == 0:
                print("ERROR")
                continue

            rs["sent_list"].append(sentID)
            rs["units"][sentID] = sent
            rs["number_edus"] += len(sent["children"])
            for eduID in sent["children"]:
                rs["units"][eduID] = doc["units"][eduID]
    return rs

def citation_excluding(input_data):
    output_data = dict()
    for sampleID, sample in tqdm(input_data.items()):
        new_sample = dict()
        new_sample["ID"] = sampleID
        new_sample["label"] = sample["label"]
        new_sample["label_embedding"] = sample["label_embedding"]
        new_sample["units"] = dict()
        new_sample["section_list"] = []
        new_sample["sent_list"] = []
        new_sample["number_edus"] = 0

        for secID in sample['section_list']:
            sec = sample["units"][secID]
            sec["children"]  = [child_id for child_id in sec["children"] if not(sample["units"][child_id]["is_citation"])]
            new_sample["units"][secID] = sec
            new_sample["section_list"].append(secID)

            for sentID in sec["children"]:
                sent = sample["units"][sentID]
                new_sample["units"][sentID] = sent
                new_sample["sent_list"].append(sentID)

                for eduID in sent["children"]:
                    new_sample["units"][eduID] = sample["units"][eduID]
                    new_sample["number_edus"] += 1

        if len(new_sample["section_list"]) > 0:
            output_data[sampleID] = new_sample
    return output_data