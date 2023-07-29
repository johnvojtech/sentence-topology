#!python3

"""
How to use:
python3 clusters.py file_with_embeddings file_for_results
"""

import sys
import copy
from collections import defaultdict
import numpy as np
import sklearn.metrics as sm
from sklearn.decomposition import PCA

def load_embeddings(filename):
    """ Load the embeddings from <filename> """
    with open(filename, "r") as r:
        embeddings = [s.strip("\n").split("\t") for s in r]
        embeddings = [(e[:4], [float(n) for n in e[4:]]) for e in embeddings]
        return embeddings

def subtract_seeds(embeddings):
    """
    Subtract the seed embeddings from the embeddings 
    The <embeddings> is a list of tuples ([ID, seed, transformation, sentence], [the embeddings])
    """
    ebs = defaultdict(list)
    seeds = {}
    for e in embeddings:
        ebs[e[0][1]].append(e)
        if e[0][2]=="seed":
            seeds[e[0][1]] = e
    new_embeddings = []
    for k in ebs.keys():
        for e in ebs[k]:
            new_embeddings.append((e[0], [x - y for x, y in zip(e[1], seeds[e[0][1]][1])]))
    return new_embeddings

def transformations(embeddings, trans):
    """
    Get the embeddings corresponding to given transformation
    """
    features = []
    labels = []
    for embedding in embeddings:
        if embedding[0][2] in trans:
            features.append(embedding[1])
            labels.append(embedding[0][2])
    return np.array(features), labels

def pca_fit(inputs, n):
    """
    not used - the PCA
    """
    pca = PCA(n_components = n)
    pca.fit(np.transpose(inputs))
    pca_output = np.transpose(pca.components_)
    return pca_output

def evaluate(trans, pca_n=0, filename=None, embeddings=None):
    """
    Rub the Calinski-Harabasz index evaluation
    """
    if embeddings==None:
        embeddings = load_embeddings(filename)
    f, t = transformations(embeddings, trans)
    if pca_n > 0:
        f = pca_fit(f, pca_n)
    return sm.calinski_harabasz_score(f,t)

transforms = ['opposite meaning', 'simple sentence', 'formal sentence', 'ban', 'minimal change', 'possibility', 'generalization', 'nonstandard sentence', 'different meaning', 'seed', 'nonsense', 'past', 'future', 'paraphrase']
transforms.sort()

# embeddings without the seed embedding
embeddings = load_embeddings(sys.argv[1])
# embeddings with the seed embedding subtracted
embeddings_diff = subtract_seeds(embeddings)

#Evaluate the embeddings by CHI
vs = []
vs_diff = []
for i in range(len(transforms)):
    for j in range(i + 1,len(transforms)):
        t, t2 = transforms[i], transforms[j]
        val = evaluate([t, t2], embeddings=embeddings)
        v2 = evaluate([t, t2], embeddings=embeddings_diff)
        vs.append([val,t,t2])
        vs_diff.append([v2, t, t2])

#Print out the results

vs.sort(key=lambda x:x[0])
vs_diff.sort(key=lambda x:x[0])
with open(sys.argv[2] + "/" + sys.argv[1].split("/")[-1] + ".free", "w") as w:
    for item in vs:
        print(item[1] + "\t" + item[2] + "\t" + str(item[0]), file=w)

with open(sys.argv[2] + "/" + sys.argv[1].split("/")[-1] + ".seed_subtracted", "w") as w:
    for item in vs_diff:
        print(item[1] + "\t" + item[2] + "\t" + str(item[0]), file=w)

