from sentence_topology.data_types import CostraEmbedding, CostraSentence
from sentence_transformers import SentenceTransformer


def get_embeddings(
    corpus: list[CostraSentence],
    sentence_model: str,
    *,
    verbose: bool = False,
) -> list[CostraEmbedding]:
    model = SentenceTransformer(sentence_model)

    all_texts = [sent.text for sent in corpus]
    all_embeds = model.encode(all_texts, show_progress_bar=verbose)

    all_id_embeds = []
    for sent, embed in zip(corpus, all_embeds):
        all_id_embeds.append(CostraEmbedding(sent.id, sent.seed_id, embed))

    return all_id_embeds
