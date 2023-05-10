# sentence-topology

Sentence embeddings topology analysis: a project for NPFL087 Statistical Machine
Translation at MFF UK.

## Links

- [Notes in google docs](https://docs.google.com/document/d/1ywUvIOaBFazc-MaJnkXkC-_ILy4b_VzXz9301Ms0_Xw/edit)

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
# For unsupervised embeddings
sent-transfomer-embedding -i ./data/COSTRA1.1.tsv -o ./embeddings/{model}.tsv

# For supervised embeddings
sent-transfomer-embedding -i ./data/COSTRA1.1.tsv -o ./embeddings/{model}_{split_ind}.tsv --train_objective "transformation-prediction"
```

- `bow-embedding` - creates embeddings for BOW and TF-IDF models

```bash
# Pure BOW; filters out tokens based on their document frequency
bow-embedding -i ./data/COSTRA1.1.tsv -o ./embeddings/bow_limited.tsv --max_df 0.8 --min_df 0.001 --no-tfidf

# TF-IDF; does not limit tokens
bow-embedding -i ./data/COSTRA1.1.tsv -o ./embeddings/tfidf.tsv  --tfidf
```

## Timeline

- April: decide next direction based on the visualizations, classifications, clustering
- End of April: Starting to wrap up
- Middle of May: Finished for the purpose of MT subject


## Dones

- [x] generate random embeddings **Vojtěch**
- [x] visualize transformations for given seed **Leixin**
- [x] training a simple classifier on the embeddings **David**
  - [x] predicting transformation
  - [x] using decision trees (or other explainable models) to explain why the embeddings are classified as such
  - [x] Confussion matrices on all embedding **David**
- [x] training a simple classifier on the embeddings **David**
  - [x] subtracting seed before classifying
  - [x] TF-IDF embeddings

## Ideas

### Prioritized


- [ ] Hierarchical clustering - dendogram ? Maybe
  - [ ] data **Vojtěch**
  - [ ] visualization **Leixin**
- [ ] Accuracy by cancelling some easy-mixedup labels **Leixin**
- [ ] visualize the amount transformation using the more transformed and less transformed columns
- [ ] Write background
- [ ] Methodology steps
- [ ] Add the survey of Classification methods   **Leixin**

### Backqueue

- [ ] training a simple classifier on the embeddings
  - [ ] Permutate words (SBERT?)
- [ ] predicting similar ids, dissimilar ids
- [ ] have pipelines for evaluating all embeddings
  - clustering
  - visualization
  - [x] classification
- [ ] clustering of embeddings to see what the cluster will be
  - try different clustering algorithms
  - maybe hierarchical clustering
- [ ] visualize the distribution of distances of transformations to seeds
- [ ] maybe try different doc2vec hyperparameters
  - different architectures
  - [x] smaller vector size? reasonable 100
- [ ] Ondrej: "Maybe by using only unsupervised embedding we are asking too
  much. Maybe we should generate the embeddings in a supervised fashion."
  - Using Jackknife resampling -- $N \times$ training on $N - 1$ samples,
    aggregating predictions
  - [x] Softmax loss
  - [ ] Contrastive loss
  - [x] Confusion matrix
- [ ] Ondrej: Is future-past relationship only a conincidence? Try also exchanging words, maybe PCA will find relationship as well...
- [ ] Run correlation between transformation ranking of the degree of the transformation and different distance metrics in the original embedding space.
- [ ] Check if a generalization transformation has swpaed "more" and "less" transformed.
- [ ] Define what would be the idea result -- document in google doc
  - Good clustering of transformed embeddings - seed embeddings
  - High accuracy of classifiers predicting transformation from embeddings
- [ ] Look at the sentences which were classified correctly. Qualitative analysis, look at lexical forms.
