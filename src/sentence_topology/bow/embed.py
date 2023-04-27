from typing import Any, Optional

from sklearn import pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sentence_topology.data_types import CostraEmbedding, CostraSentence


def get_embeddings(
    corpus: list[CostraSentence],
    *,
    tfidf: bool = True,
    vectorizer_kwargs: Optional[dict[str, Any]] = None,
    tfidf_kwargs: Optional[dict[str, Any]] = None,
) -> list[CostraEmbedding]:
    if vectorizer_kwargs is None:
        vectorizer_kwargs = {}

    if tfidf_kwargs is None:
        tfidf_kwargs = {}

    steps: list[tuple[str, Any]] = [
        ("vectorizer", CountVectorizer(**vectorizer_kwargs))
    ]
    if tfidf:
        steps.append(("tfidf", TfidfTransformer(**tfidf_kwargs)))
    vectorizer = pipeline.Pipeline(steps)

    vectorizer.fit((doc.text for doc in corpus))

    embeddings = []
    for doc in corpus:
        embeddings.append(
            CostraEmbedding(
                id=doc.id,
                seed_id=doc.seed_id,
                trans=doc.trans,
                embedding=vectorizer.transform([doc.text]).toarray()[0],
            )
        )

    return embeddings
