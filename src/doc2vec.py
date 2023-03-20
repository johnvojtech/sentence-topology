import sys
import os
import pickle
import gensim
from gensim.models.doc2vec import Doc2Vec
from gensim.test.utils import get_tmpfile

def load_sentences(filename):
    with open(filename, "r") as r:
        sentences = [s.lower().strip().split("\t")[:4]for s in r]
        return sentences

def train_embeddings(filename, vector_size):
    model = Doc2Vec(corpus_file=filename, vector_size=vector_size, window=4, min_count=1)
    model.build_vocab(corpus_file=filename)
    model.train(corpus_file=filename, total_examples=model.corpus_count, total_words=model.corpus_total_words, epochs=model.epochs)
    fname = get_tmpfile("my_doc2vec_model")
    model.save(fname)
    return model

def get_embeddings(model, sentences, output_file):
    with open(output_file, "w") as w:
        for sentence in sentences:
            inferred = model.infer_vector(sentence[3].split())
            print("\t".join([str(x) for x in sentence[:3] + list(inferred)]), file=w)

if __name__ == "__main__":
    train_data = sys.argv[2]
    vector_size = int(sys.argv[3])
    model = train_embeddings(train_data, vector_size)
    sentences = load_sentences(sys.argv[1])
    get_embeddings(model, sentences, "doc2vec_vsize_" + str(vector_size) + ".tsv")
