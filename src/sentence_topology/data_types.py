import collections

CostraSentence = collections.namedtuple(
    "CostraSentence",
    [
        "id",
        "seed_id",
        "trans",
        "text",
        "more_trans",
        "less_trans",
        "similar",
        "dissimilar",
    ],
)

CostraEmbedding = collections.namedtuple(
    "CostraEmbedding", ["id", "seed_id", "trans", "embedding"]
)
