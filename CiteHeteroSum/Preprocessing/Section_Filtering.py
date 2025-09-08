from tqdm import tqdm

def section_filtering(sample, sampleID, input_section_names, label_rouge_threshold):
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
        if sec["text"] not in input_section_names:
            continue

        new_sample["units"][secID] = sec
        new_sample["section_list"].append(secID)

        for sentID in sec["children"]:
            sent = sample["units"][sentID]
            new_sample["units"][sentID] = sent
            new_sample["sent_list"].append(sentID)

            for eduID in sent["children"]:
                edu = sample["units"][eduID]
                new_sample["units"][eduID] = edu
                new_sample["number_edus"] += 1

                rouge2p_score = edu["golden_rouge"]["2p"]
                edu["golden_label"] = {"r2p_thres": 1 if rouge2p_score > label_rouge_threshold else 0}
    return new_sample

def section_list(data):
    section_set = set()

    for sampleID, sample in data.items():
        for secID in sample['section_list']:
            sec = sample["units"][secID]
            section_set.add(sec["text"])

    print(section_set)
