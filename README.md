# Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks

This repository is the implementation of the paper [Sentence-Bert](https://arxiv.org/pdf/1908.10084.pdf) a modification of the pretrained  BERT network that use siamese and triplet network structures to derive semantically meaningful sentence embeddings that can be compared using cosine-similarity. 

This reduces the effort for finding the most similar pair from 65 hours with BERT / RoBERTa to about 5 seconds with SBERT, while maintaining the accuracy from BERT

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 

### Prerequisites

 
install the dependencies for this project by running the following commands in your terminal:

```
 pip install -r requirements.txt
```

run the model by running the following command in your terminal:

```
python src/sbert.py --train_file="./input/wolof.csv" \
                        --max_length=150 \
                        --epochs=10 \
                        --learning_rate=3e-8 \
                        --epsilone=1e-9 \
                        --train_batch_size=3 \
                        --model_name="bert-base-cased"
```

 
