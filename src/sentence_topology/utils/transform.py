import numpy as np

from sentence_topology.data_types import CostraEmbedding

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
            id_to_embed[embed.id] = embed.embedding

    transformed = []
    for embed in embeddings:
        new_embedding = CONTEXT_MODES[mode](id_to_embed[embed.seed_id], embed.embedding)
        transformed.append(
            CostraEmbedding(
                id=embed.id,
                seed_id=embed.seed_id,
                trans=embed.trans,
                embedding=new_embedding,
            )
        )

    return transformed
