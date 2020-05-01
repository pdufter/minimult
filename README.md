
# Identifying Necessary Elements for BERT's Multilingualitxy

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
for more details. The (forthcoming) link to the corresponding arxiv upload is here: [Paper](https://arxiv.org/pdf/tbd.pdf)
If you use the code, please consider citing the paper.


# TODO
* get rid of unnecessary folder hierarchies
* Update readme
* link arxiv upload
