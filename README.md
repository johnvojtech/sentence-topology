# Topology of Czech sentences

This is a repo containg code to a paper we wrote for LREC COOLING 2024. The paper is called *"Unveiling semantic information in Sentence Embedding"* and is available in [ACL Anthology](https://aclanthology.org/2024.dmr-1.5/).

  
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

## Cite this

If you use our work, feel free to cite the mentioned paper:
```tex
@inproceedings{zhang-etal-2024-unveiling,
    title = "Unveiling Semantic Information in Sentence Embeddings",
    author = "Zhang, Leixin  and
      Burian, David  and
      John, Vojt{\v{e}}ch  and
      Bojar, Ond{\v{r}}ej",
    editor = "Bonial, Claire  and
      Bonn, Julia  and
      Hwang, Jena D.",
    booktitle = "Proceedings of the Fifth International Workshop on Designing Meaning Representations @ LREC-COLING 2024",
    month = may,
    year = "2024",
    address = "Torino, Italia",
    publisher = "ELRA and ICCL",
    url = "https://aclanthology.org/2024.dmr-1.5",
    pages = "39--47",
    abstract = "This study evaluates the extent to which semantic information is preserved within sentence embeddings generated from state-of-art sentence embedding models: SBERT and LaBSE. Specifically, we analyzed 13 semantic attributes in sentence embeddings. Our findings indicate that some semantic features (such as tense-related classes) can be decoded from the representation of sentence embeddings. Additionally, we discover the limitation of the current sentence embedding models: inferring meaning beyond the lexical level has proven to be difficult.",
}
```
