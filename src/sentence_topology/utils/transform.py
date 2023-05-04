import logging
import os
from glob import glob
from random import Random
from typing import Callable, Iterable, Optional

import numpy as np
from tqdm.auto import tqdm

from sentence_topology.data_types import CostraEmbedding
from sentence_topology.utils.io import load_embedding

logger = logging.getLogger(__name__)


class EmbeddingContextualizationError(Exception):
    pass


CONTEXT_MODES = {
    "diff": lambda seed, sent: sent - seed,
    "concat": lambda seed, sent: np.concatenate([seed, sent]),
}


def contextualize_embeddings(
    embeddings: list[CostraEmbedding],
    *,
    mode: str = "diff",
) -> list[CostraEmbedding]:
    id_to_embed = {}

    for embed in embeddings:
        if embed.trans == "seed":
            id_to_embed[embed.seed_id] = embed.embedding

    transform = CONTEXT_MODES[mode]
    transformed = []
    for embed in embeddings:
        if embed.seed_id not in id_to_embed:
            raise EmbeddingContextualizationError()

        new_embedding = transform(id_to_embed[embed.seed_id], embed.embedding)
        transformed.append(
            CostraEmbedding(
                id=embed.id,
                seed_id=embed.seed_id,
                trans=embed.trans,
                embedding=new_embedding,
            )
        )

    return transformed


def equalize_transformations(
    embeddings: list[CostraEmbedding],
    *,
    include_seed: bool = False,
    random_state: Optional[int] = None,
    shuffle_after: bool = True,
) -> list[CostraEmbedding]:
    by_trans = {}
    for embed in embeddings:
        if embed.trans not in by_trans:
            by_trans[embed.trans] = []

        by_trans[embed.trans].append(embed)

    if not include_seed and "seed" in by_trans:
        del by_trans["seed"]

    rand_gen = Random(random_state)
    samples_num = min(len(embeds) for embeds in by_trans.values())
    new_embeddings = []
    for embeds in by_trans.values():
        for sampled_embed in rand_gen.sample(embeds, k=samples_num):
            new_embeddings.append(sampled_embed)

    if shuffle_after:
        rand_gen.shuffle(new_embeddings)

    return new_embeddings


class EmbeddingsLoader:
    def __init__(
        self,
        embeddings_dir: str,
        *,
        context_mode: Optional[str],
        equalize_trans: bool,
    ) -> None:
        self._embeddings_dir = embeddings_dir
        self._context_mode = context_mode
        self._equalize_trans = equalize_trans

    def list_all(
        self, *, tqdm_enable: bool = False, tqdm_desc: str = "Processing embeddings"
    ) -> Iterable[str]:
        total_embeddings = len(glob(os.path.join(self._embeddings_dir, "*.tsv")))
        with tqdm(
            os.scandir(self._embeddings_dir),
            desc=tqdm_desc,
            total=total_embeddings,
            disable=not tqdm_enable,
        ) as progress_bar:
            for entry in progress_bar:
                progress_bar.set_postfix(embedding=entry.name)
                if entry.name.endswith(".tsv"):
                    try:
                        self.load(entry.name)
                        yield entry.name
                    except EmbeddingContextualizationError:
                        logger.info(
                            "Unable to load and process %s. Skipping.", entry.name
                        )

    def load(self, name: str) -> list[CostraEmbedding]:
        embedding = list(load_embedding(os.path.join(self._embeddings_dir, name)))
        if self._context_mode is not None:
            embedding = contextualize_embeddings(embedding, mode=self._context_mode)

        if self._equalize_trans:
            embedding = equalize_transformations(embedding)

        return embedding
