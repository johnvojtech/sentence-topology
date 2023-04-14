#!python3
import sys
import numpy as np
import sklearn.metrics as sm
from sklearn.decomposition import PCA

def load_embeddings(filename):
    with open(filename, "r") as r:
        embeddings = [s.strip("\n").split("\t") for s in r]
        #print(set([x[2] for x in embeddings]))
        embeddings = [(e[:4], [float(n) for n in e[4:]]) for e in embeddings]
        return embeddings

def transformations(embeddings, trans):
    features = []
    labels = []
    for embedding in embeddings:
        if embedding[0][2] in trans:
            features.append(embedding[1])
            labels.append(embedding[0][2])
    return np.array(features), labels

def pca_fit(inputs, n):
    pca = PCA(n_components = n)
    pca.fit(np.transpose(inputs))
    pca_output = np.transpose(pca.components_)
    return pca_output

def evaluate(trans, pca_n=0,filename=None, embeddings=None):
    if embeddings==None:
        embeddings = load_embeddings(filename)
    f, t = transformations(embeddings, trans)
    if pca_n > 0:
        f = pca_fit(f, pca_n)
    #return sm.silhouette_score(f,t)
    #return sm.davies_bouldin_score(f,t)
    return sm.calinski_harabasz_score(f,t)

transforms = ['opposite meaning', 'simple sentence', 'formal sentence', 'ban', 'minimal change', 'possibility', 'generalization', 'nonstandard sentence', 'different meaning', 'seed', 'nonsense', 'past', 'future', 'paraphrase']
transforms.sort()

embeddings = load_embeddings(sys.argv[1])
"""
labels = [x[0][2] for x in embeddings]
features = np.array([x[1] for x in embeddings])
print(sm.davies_bouldin_score(features, labels))
"""
vs = []
for i in range(len(transforms)):
    for j in range(i + 1,len(transforms)):
        t, t2 = transforms[i], transforms[j]
        val = evaluate([t, t2], embeddings=embeddings)
        vs.append([val,t,t2])

vs.sort(key=lambda x:x[0])
for item in vs:
    print(item[1] + " x " + item[2] + " : " + str(item[0]))

#print(evaluate(transforms, embeddings=embeddings)
