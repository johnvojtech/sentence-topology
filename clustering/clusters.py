#!python3
import sys
import copy
from collections import defaultdict
import numpy as np
import sklearn.metrics as sm
from sklearn.decomposition import PCA

def load_embeddings(filename):
    with open(filename, "r") as r:
        embeddings = [s.strip("\n").split("\t") for s in r]
        #print(set([x[2] for x in embeddings]))
        embeddings = [(e[:4], [float(n) for n in e[4:]]) for e in embeddings]
        return embeddings

def subtract_seeds(embeddings):
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
embeddings_diff = subtract_seeds(embeddings)
#print(embeddings)
#print(embeddings_diff)

"""
chs = sm.calinski_harabasz_score(features, labels)

subdif = {}

wia = []
for i in range(len(transforms)):
    filtered = [copy.deepcopy(x) for x in embeddings]
    for x in filtered:
        if x[0][2] != transforms[i]:
            x[0][2] = "N"
    labels = [x[0][2] for x in filtered]
    features = np.array([x[1] for x in filtered])
    chsn = sm.calinski_harabasz_score(features, labels)
    subdif[transforms[i]] = chsn
    wia.append([chsn, transforms[i]])
wia.sort(key=lambda x:x[0])
for w in wia:
    print(w[1] + "\t" + str(w[0]) + "\t" + str(w[0] - chs))


labels = [x[0][2] for x in embeddings_diff]
features = np.array([x[1] for x in embeddings_diff])
chs = sm.calinski_harabasz_score(features, labels)
print("Subtract seed")
wia = []
for i in range(len(transforms)):
    filtered = [copy.deepcopy(x) for x in embeddings_diff]
    for x in filtered:
        if x[0][2] != transforms[i]:
            x[0][2] = "N"
    labels = [x[0][2] for x in filtered]
    features = np.array([x[1] for x in filtered])
    chsn = sm.calinski_harabasz_score(features, labels)
    wia.append([chsn, transforms[i]])
    subdif[transforms[i]] -= chsn
wia.sort(key=lambda x:x[0])
for w in wia:
    print(w[1] + "\t" + str(w[0]) + "\t" + str(chs - w[0]))

print("Diff:")
wls = [[k, subdif[k]] for k in subdif.keys()]
wls.sort(key=lambda x:x[1])
for item in wls:
     print(str(item[0]) + "\t" + str(item[1]))
print()


for i in range(len(transforms)):
    filtered = [x for x in embeddings if x[0][1] != transforms[i]]
    labels = [x[0][2] for x in filtered]
    features = np.array([x[1] for x in filtered])
    wia.append([sm.calinski_harabasz_score(features, labels), transforms[i]])
wia.sort(key=lambda x:x[1])
for w in wia:
    print(str(w[0]) + "\t" + w[1])
"""

vs = []
vs_diff = []
for i in range(len(transforms)):
    for j in range(i + 1,len(transforms)):
        t, t2 = transforms[i], transforms[j]
        val = evaluate([t, t2], embeddings=embeddings)
        v2 = evaluate([t, t2], embeddings=embeddings_diff)
        vs.append([val,t,t2])
        vs_diff.append([v2, t, t2])

vs.sort(key=lambda x:x[0])
vs_diff.sort(key=lambda x:x[0])
with open(sys.argv[1].split("/")[-1] + ".free", "w") as w:
    for item in vs:
        print(item[1] + "\t" + item[2] + "\t" + str(item[0]), file=w)
with open(sys.argv[1].split("/")[-1] + ".seed_subtracted", "w") as w:
    for item in vs_diff:
        print(item[1] + "\t" + item[2] + "\t" + str(item[0]), file=w)
