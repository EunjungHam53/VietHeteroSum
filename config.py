args = {
    'seed': 42,
    'label_rouge_threshold': 0.5,
    'input_section_names': ['Abstract', 'Conclusion'],
    'input': 768,
    'hidden': 128,
    'heads': 64,
    'epochs': 48,
    'lr': 0.0003,
    'dropout': 0.3,
    'c_patient': 30,
    'summary_max_word_num': 165,
    'kappa': 0.3,
    'triplet_threds': [0.4, 0.5],
    'triplet_margin': 2.0,
    'triplet_topk': 3
}