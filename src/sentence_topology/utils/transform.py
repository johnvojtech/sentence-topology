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
) -> tuple[list[CostraEmbedding], int]:
    id_to_embed = {}

    for embed in embeddings:
        if embed.trans == "seed":
            id_to_embed[embed.seed_id] = embed.embedding

    skipped_count = 0
    transform = CONTEXT_MODES[mode]
    transformed = []
    for embed in embeddings:
        if embed.seed_id not in id_to_embed:
            skipped_count += 1
            continue
        new_embedding = transform(id_to_embed[embed.seed_id], embed.embedding)
        transformed.append(
            CostraEmbedding(
                id=embed.id,
                seed_id=embed.seed_id,
                trans=embed.trans,
                embedding=new_embedding,
            )
        )

    return transformed, skipped_count
