import matplotlib.pyplot as plt

from sentence_topology.classification.analysis import ClassifierAnalysisResults
from sentence_topology.visualization.predictions import draw_confusion_matrix
doc2vec_results = ClassifierAnalysisResults.load(
    "./results/best_classifier_analysis-doc2vec-cs-vecsize-512-train-1M-sents.tsv.pkl"
)

# Un-normalized confusion matrix
print(doc2vec_results.confusion_matrix)
print('done')