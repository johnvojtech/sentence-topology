# sentence-topology
Sentence embeddings topology analysis: a project for NPFL087 Statistical Machine Translation at MFF UK.

## Installation

Just run the following command (using your virtual environment) in the root of
the project:

```bash
pip install -e .
```

## Scripts

- `sent-transfomer-embedding` - gets embeddings using `sentence-transformers`
  package

```bash
sent-transfomer-embedding -i ./data/COSTRA1.1.tsv -o ./embeddings.tsv
```

## Ideas

- [ ] visualize transformations for given seed **Leixin**
- [ ] training a simple classifier on the embeddings **David**
  - predicting transformation
  - predicting similar ids, dissimilar ids
  - using decision trees (or other explainable models) to explain why the embeddings are classified as such
- [ ] clustering of embeddings to see what the cluster will be
  - try different clustering algorithms
  - maybe hierarchical clustering
- [ ] visualize the distribution of distances of transformations to seeds
- [ ] maybe try different doc2vec hyperparameters
  - different architectures
  - smaller vector size? reasonable 100
