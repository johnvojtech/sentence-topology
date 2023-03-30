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

## Timeline

- Middle April: decide next direction based on the visualizations, classifications, clustering
- End of May: Starting to wrap up
- End of june: Finished for the purpose of MT subject

## Ideas

### Prioritized

- [ ] visualize transformations for given seed **Leixin**
- [ ] training a simple classifier on the embeddings **David**
  - [ ] predicting transformation
  - [ ] predicting similar ids, dissimilar ids
  - [ ] using decision trees (or other explainable models) to explain why the embeddings are classified as such
  - [ ] Confussion matrices on all embedding **David**
- [ ] generate random embeddings **Vojtěch**
- [ ] visualize the amount transformation using the more transformed and less transformed columns


### Backqueue



- [ ] have pipelines for evaluating all embeddings
  - clustering
  - visualization
  - classification
- [ ] clustering of embeddings to see what the cluster will be
  - try different clustering algorithms
  - maybe hierarchical clustering
- [ ] visualize the distribution of distances of transformations to seeds
- [ ] maybe try different doc2vec hyperparameters
  - different architectures
  - [x] smaller vector size? reasonable 100
- [ ] Ondrej: "Maybe by using only unsupervised embedding we are asking too much. Maybe we should generate the embeddings in a supervised fashion."
  - Using Jacked-Knife approach -- using 9/10 of the data to train, generate the embeddings for the 1/9 of the data
  - [x] Softmax loss
  - [ ] Contrastive loss
  - Confusion matrix **David**
- [ ] Ondrej: Is future-past relationship only a conincidence? Try also exchanging words, maybe PCA will find relationship as well...
- [ ] Run correlation between transformation ranking of the degree of the transformation and different distance metrics in the original embedding space.
- [ ] Check if a generalization transformation has swpaed "more" and "less" transformed.
