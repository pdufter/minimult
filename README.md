
Code for the Paper "Identifying Necessary Elements for BERT's Multilinguality"
==============


About
--------

These experiments aim at identifying necessary elements for BERT's multilinguality. 
The objective is to model this in a small, laboratory setting that allows
for fast experimentation. 

It allows to train BERT for two languages: English and Fake-English. 
The objective is to identify architectural properties of BERT as well as 
linguistic properties of the involved languages that are necessary in order for 
BERT to create multilingual representations. 

Among the things investigated are
* Number of Parameters
* Shifting special tokens
* Language specific position embeddings
* Not replacing masked tokens with random tokens
* Inverting the language order
* Avoiding parallel training corpus

Language model fit is evaluated with perplexity. Multilinguality with 
Word Alignment, Sentence Retrieval and Word Translation. See the paper
for more details. 

Data
--------

Unfortunately, due to copyright restrictions, the Easy-to-read Bible is currently not publicly available. Other data download links are in the paper.


Setup
--------

The code is mostly based on [huggingface transformers](https://github.com/huggingface/transformers) and their awesome pretraining scripts (thanks!). `setup.sh` preprocesses the data. `run.sh` contains all experiments in the small English-FakeEnglish setup.
The folder `real` contains code for experiments on Wikipedia and XNLI data. `real/run.sh` contains all experiments for the real data setup. 


References
--------
You can find the [paper](https://arxiv.org/abs/2005.00396) on arxiv. It will appear in the Proceedings of EMNLP 2020. 
```
@article{dufter2020identifying,
  title={Identifying Necessary Elements for BERT's Multilinguality},
  author={Dufter, Philipp and Sch{\"u}tze, Hinrich},
  journal={arXiv preprint arXiv:2005.00396},
  year={2020},
  comment={to appear in EMNLP 2020}
}
```

If you use the code, please consider citing the paper.

