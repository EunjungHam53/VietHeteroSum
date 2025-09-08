# Beyond the Scientific Document: A Citation-Aware Multi-Granular Summarization Approach with Heterogeneous Graphs
___

This repository contains the implementation of a citation-aware multi-granular summarization approach using heterogeneous graphs. The model is designed to process scientific documents and generate summaries that consider both document content and citation context.
___

## Data
This project uses three publicly available datasets: CL-SciSumm [link](https://aclanthology.org/2020.sdp-1.24/), 
SciSumNet [link](https://ojs.aaai.org/index.php/AAAI/article/view/4727), 
and CiteArXiv [link](https://dl.acm.org/doi/abs/10.1145/3701716.3735083). 
CL-SciSumm and SciSumNet are formatted following CiteArXiv format and provided in this repository.
You can download CiteArXiv from  [link](https://drive.google.com/drive/folders/1A8HCo4yTZuHn6Tz3fbna5L1F72Ej3vCt?usp=sharing).
These datasets are used to construct three experimental scenarios:
- (1) **High-quality Dataset**  Training:  CL-SciSumm training set, Testing: CL-SciSumm test set.
- (2) **Cross-Dataset** Training: SciSumNet, Testing: Full CL-SciSumm.
- (3) **Large Dataset** Training: CiteArXiv training set, Testing: CiteArXiv test set.


## Usage

### Data Preprocessing
To preprocess the data, run:
```bash
jupyter nbconvert --to notebook --execute notebook-process/processing_data.ipynb
```
### Training Process and Evaluation Process

```bash
jupyter nbconvert --to notebook --execute notebook-process/training_and_evaluation.ipynb
```
### Inference 
```bash
jupyter nbconvert --to notebook --execute notebook-process/inference.ipynb
```
