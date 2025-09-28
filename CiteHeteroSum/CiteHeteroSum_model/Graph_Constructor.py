import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import *
import numpy as np
import torch

def mask_to_adj(doc_sect_mask, sect_sent_mask):
    """
    Create adjacency matrix for the hierarchical graph
    doc_sect_mask: (doc_num, sect_num) - connection between documents and sections
    sect_sent_mask: (sect_num, sent_num) - connection between sections and sentences
    """
    doc_sect_mask = np.array(doc_sect_mask)
    sect_sent_mask = np.array(sect_sent_mask)

    sent_num = sect_sent_mask.shape[1]
    sect_num = sect_sent_mask.shape[0]
    doc_num = doc_sect_mask.shape[0]

    total_nodes = sent_num + sect_num + doc_num
    adj = np.zeros((total_nodes, total_nodes))
    
    # Connect sections to sentences
    # sent: [0:sent_num], sect: [sent_num:sent_num+sect_num], doc: [sent_num+sect_num:]
    adj[sent_num:sent_num+sect_num, 0:sent_num] = sect_sent_mask
    adj[0:sent_num, sent_num:sent_num+sect_num] = sect_sent_mask.T
    
    # Connect documents to sections
    adj[sent_num+sect_num:, sent_num:sent_num+sect_num] = doc_sect_mask
    adj[sent_num:sent_num+sect_num, sent_num+sect_num:] = doc_sect_mask.T
    
    # Self-connections within same level (sections within same document)
    for i in range(doc_num):
        doc_mask = doc_sect_mask[i]
        if doc_mask.ndim == 1:
            doc_mask = doc_mask.reshape((1, -1))
        elif doc_mask.ndim == 0:
            doc_mask = np.array([doc_mask])
        
        # Connect sections that belong to the same document
        adj[sent_num:sent_num+sect_num, sent_num:sent_num+sect_num] += doc_mask * doc_mask.T
    
    # Self-connections within same level (sentences within same section)
    for i in range(sect_num):
        sect_mask = sect_sent_mask[i]
        if sect_mask.ndim == 1:
            sect_mask = sect_mask.reshape((1, -1))
        elif sect_mask.ndim == 0:
            sect_mask = np.array([sect_mask])
        
        # Connect sentences that belong to the same section
        adj[0:sent_num, 0:sent_num] += sect_mask * sect_mask.T
    
    # Self-connections at document level
    adj[sent_num+sect_num:, sent_num+sect_num:] = 1

    return adj

class Graph:
    def __init__(self, sentences, sent_vectors, sect_vectors, doc_vectors, scores, doc_sect_mask, sect_sent_mask, golden, golden_vec, threds):
        assert len(sent_vectors) == len(scores) == len(sentences), "ERROR: Mismatch in dimensions"
        
        self.doc_num = len(doc_sect_mask)
        self.sect_num = len(sect_sent_mask)
        self.sent_num = len(sent_vectors)

        self.adj = torch.from_numpy(mask_to_adj(doc_sect_mask, sect_sent_mask)).float()
        
        # Feature concatenation: [sentences, sections, documents]
        self.feature = torch.cat((
            torch.stack(sent_vectors), 
            torch.stack(sect_vectors), 
            torch.stack(doc_vectors)
        ), dim=0)

        neg_thred = threds[0]
        pos_thred = threds[1]

        self.score = torch.from_numpy(np.array(scores)).float()
        self.score_onehot = (self.score >= pos_thred).float()
        self.score_onehot_neg = (self.score <= neg_thred).float()

        self.sentences = np.array(sentences)
        self.golden = golden
        self.golden_vec = golden_vec.unsqueeze(0)


def graph_construction(sample, args):
    """
    Construct graph from sample data with new structure:
    cluster -> document -> section -> sentence
    """
    doc_list = sample['cluster']['document_list']
    sect_list = sample['cluster']['section_list'] 
    sent_list = sample['cluster']['sentence_list']
    units = sample['cluster']['units']
    
    doc_num = len(doc_list)
    sect_num = len(sect_list)
    sent_num = len(sent_list)

    sentences = []
    sentVecs = []
    sectVecs = []
    docVecs = []
    scores = []

    # Create masks for hierarchical connections
    doc_sect_mask = np.zeros((doc_num, sect_num))
    sect_sent_mask = np.zeros((sect_num, sent_num))

    # Process documents
    for doc_idx, doc_id in enumerate(doc_list):
        doc_unit = units[doc_id]
        docVecs.append(doc_unit["embedding"])
        
        # Find sections belonging to this document
        doc_sections = doc_unit["children"]
        for sect_id in doc_sections:
            if sect_id in sect_list:
                sect_idx = sect_list.index(sect_id)
                doc_sect_mask[doc_idx, sect_idx] = 1

    # Process sections  
    for sect_idx, sect_id in enumerate(sect_list):
        sect_unit = units[sect_id]
        sectVecs.append(sect_unit["embedding"])
        
        # Find sentences belonging to this section
        sect_sentences = sect_unit["children"]
        for sent_id in sect_sentences:
            if sent_id in sent_list:
                sent_idx = sent_list.index(sent_id)
                sect_sent_mask[sect_idx, sent_idx] = 1

    # Process sentences
    for sent_idx, sent_id in enumerate(sent_list):
        sent_unit = units[sent_id]
        sentences.append(sent_unit["text"])
        sentVecs.append(sent_unit["embedding"])
        
        # Get score (assuming it exists in the unit)
        if 'golden_rouge' in sent_unit:
            scores.append(sent_unit['golden_rouge']['2p'])
        else:
            scores.append(0.0)  # Default score if not available

    # Get label information
    label_data = sample["label"]
    label_embedding = sample.get("label_embedding", torch.zeros(768))  # Default if not available

    tmp_graph = Graph(
        sentences=sentences,
        sent_vectors=sentVecs, 
        sect_vectors=sectVecs,
        doc_vectors=docVecs,
        scores=scores, 
        doc_sect_mask=doc_sect_mask,
        sect_sent_mask=sect_sent_mask, 
        golden=label_data, 
        golden_vec=label_embedding, 
        threds=args['triplet_threds']
    )
    
    return tmp_graph

def validate_graph_structure(sample):
    """
    Validate that the sample has the expected structure
    """
    required_keys = ['cluster']
    cluster_keys = ['document_list', 'section_list', 'sentence_list', 'units']
    
    for key in required_keys:
        if key not in sample:
            raise ValueError(f"Missing key: {key}")
    
    for key in cluster_keys:
        if key not in sample['cluster']:
            raise ValueError(f"Missing key in cluster: {key}")
    
    # Check that all IDs in lists exist in units
    units = sample['cluster']['units']
    all_ids = (sample['cluster']['document_list'] + 
               sample['cluster']['section_list'] + 
               sample['cluster']['sentence_list'])
    
    for unit_id in all_ids:
        if unit_id not in units:
            raise ValueError(f"Unit ID {unit_id} not found in units")
    
    print("Graph structure validation passed!")
    return True

# Example usage
def main():
    # Example of how to use the updated graph construction
    sample_data = {
        'ID': 0,
        'label': 'Sample summary text...',
        'cluster': {
            'document_list': ['0|1|0|0', '0|2|0|0'],
            'section_list': ['0|1|1|0', '0|1|2|0', '0|2|1|0'],
            'sentence_list': ['0|1|1|1', '0|1|1|2', '0|1|2|1', '0|2|1|1'],
            'units': {
                # Document units
                '0|1|0|0': {
                    'unit_type': 'document',
                    'children': ['0|1|1|0', '0|1|2|0'],
                    'embedding': torch.randn(768),
                    'text': 'Document 1 text...'
                },
                '0|2|0|0': {
                    'unit_type': 'document', 
                    'children': ['0|2|1|0'],
                    'embedding': torch.randn(768),
                    'text': 'Document 2 text...'
                },
                # Section units
                '0|1|1|0': {
                    'unit_type': 'section',
                    'children': ['0|1|1|1', '0|1|1|2'],
                    'embedding': torch.randn(768),
                    'text': 'Section 1 text...'
                },
                '0|1|2|0': {
                    'unit_type': 'section',
                    'children': ['0|1|2|1'],
                    'embedding': torch.randn(768),
                    'text': 'Section 2 text...'
                },
                '0|2|1|0': {
                    'unit_type': 'section',
                    'children': ['0|2|1|1'],
                    'embedding': torch.randn(768),
                    'text': 'Section 3 text...'
                },
                # Sentence units
                '0|1|1|1': {
                    'unit_type': 'sentence',
                    'children': [],
                    'embedding': torch.randn(768),
                    'text': 'First sentence...',
                    'golden_rouge': {'2p': 0.8}
                },
                '0|1|1|2': {
                    'unit_type': 'sentence',
                    'children': [],
                    'embedding': torch.randn(768),
                    'text': 'Second sentence...',
                    'golden_rouge': {'2p': 0.6}
                },
                '0|1|2|1': {
                    'unit_type': 'sentence',
                    'children': [],
                    'embedding': torch.randn(768),
                    'text': 'Third sentence...',
                    'golden_rouge': {'2p': 0.4}
                },
                '0|2|1|1': {
                    'unit_type': 'sentence',
                    'children': [],
                    'embedding': torch.randn(768),
                    'text': 'Fourth sentence...',
                    'golden_rouge': {'2p': 0.9}
                }
            }
        },
        'label_embedding': torch.randn(768)
    }
    
    args = {'triplet_threds': [0.3, 0.7]}
    
    # Validate structure
    validate_graph_structure(sample_data)
    
    # Create graph
    graph = graph_construction(sample_data, args)
    
    print(f"Graph created successfully!")
    print(f"Documents: {graph.doc_num}")
    print(f"Sections: {graph.sect_num}")
    print(f"Sentences: {graph.sent_num}")
    print(f"Adjacency matrix shape: {graph.adj.shape}")
    print(f"Feature matrix shape: {graph.feature.shape}")

if __name__ == "__main__":
    main()