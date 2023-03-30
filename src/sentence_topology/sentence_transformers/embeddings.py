import os
from typing import Iterable, Optional

from sentence_transformers import SentenceTransformer
from sklearn import model_selection
from torch.utils.data import DataLoader

from sentence_topology.data_types import CostraEmbedding, CostraSentence

from .train import (create_transformation_prediction_data,
                    train_with_transformation_prediction)


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
        all_id_embeds.append(CostraEmbedding(sent.id, sent.seed_id, sent.trans, embed))

    return all_id_embeds


def get_embeddings_trans_prediction(
    corpus: list[CostraSentence],
    sentence_model: str,
    *,
    splits: int = 5,
    epochs: int = 4,
    verbose: bool = False,
    log_dir: Optional[str] = None,
) -> Iterable[list[CostraEmbedding]]:
    kfold = model_selection.StratifiedGroupKFold(n_splits=splits)
    data = create_transformation_prediction_data(corpus)
    for split_ind, (train_inds, test_inds) in enumerate(
        kfold.split(data.sent_pairs, y=data.trans, groups=data.seed_ids)
    ):
        train = DataLoader(
            [data.sent_pairs[i] for i in train_inds],
            batch_size=8,
            shuffle=True,
        )
        test = DataLoader([data.sent_pairs[i] for i in test_inds], batch_size=8)

        model = SentenceTransformer(sentence_model)

        split_log_dir = None
        if log_dir is not None:
            split_log_dir = os.path.join(log_dir, str(split_ind))

        train_with_transformation_prediction(
            model=model,
            train_data=train,
            test_data=test,
            epochs=epochs,
            num_labels=len(data.label_encoder.classes_),
            log_dir=split_log_dir,
            verbose=verbose,
        )

        test_sents = [data.sents[i] for i in test_inds]
        test_embeds = model.encode(
            [sent.text for sent in test_sents],
            show_progress_bar=verbose,
        )

        yield [
            CostraEmbedding(sent.id, sent.seed_id, sent.trans, embed)
            for sent, embed in zip(test_sents, test_embeds)
        ]
