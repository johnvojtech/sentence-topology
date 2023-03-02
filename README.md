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

