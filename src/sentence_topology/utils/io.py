from typing import Any, Iterable
import numpy as np

from ..data_types import CostraEmbedding, CostraSentence


def load_corpus(path: str) -> Iterable[CostraSentence]:
    """Loads .tsv file at `path`."""

    def _process_id_list(field: str) -> list[int]:
        ids = filter(lambda id: id != "", field.split(","))
        ids = map(int, ids)
        return list(ids)

    with open(path, mode="r", encoding="utf-8") as corpus:
        for line in corpus:
            line = line.rstrip("\n")
            fields: list[Any] = list(line.split("\t"))
            for i in range(2):
                fields[i] = int(fields[i])
            for i in range(4, 8):
                fields[i] = _process_id_list(fields[i])
            yield CostraSentence(*fields)


def save_embeddings(embeddings: list[CostraEmbedding], path: str) -> None:
    """Saves `embeddings` to a .tsv file at `path`."""

    def _arr_to_str(array: np.ndarray) -> str:
        return "\t".join(map(str, array))

    with open(path, mode="w", encoding="utf-8") as out_file:
        for sent_embed in embeddings:
            fields = [
                str(sent_embed.id),
                str(sent_embed.seed_id),
                str(sent_embed.trans),
                _arr_to_str(sent_embed.embedding),
            ]
            print("\t".join(fields), file=out_file)


def load_embedding(path: str) -> Iterable[CostraEmbedding]:
    """Loads embeddings from a .tsv file at `path`."""

    with open(path, mode="r", encoding="utf-8") as embeddings:
        for line in embeddings:
            line = line.rstrip("\n")
            fields = line.split("\t")
            yield CostraEmbedding(
                int(fields[0]),
                int(fields[1]),
                fields[2],
                np.array([float(dim) for dim in fields[3:]], dtype=np.float32),
            )
