import math
import os
from dataclasses import dataclass
from typing import Iterable, Optional, cast

import numpy as np
import torch
from sentence_transformers import InputExample, SentenceTransformer, losses
from sentence_transformers.evaluation import (SentenceEvaluator,
                                              SequentialEvaluator)
from sentence_transformers.util import batch_to_device
from sklearn import preprocessing
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from sentence_topology.data_types import CostraSentence


class TBLabelAccuracyEvaluator(SentenceEvaluator):
    def __init__(
        self,
        dataloader: DataLoader,
        *,
        softmax_model: torch.nn.Module,
        name: str,
        log_dir: str,
        steps_per_epoch: int,
    ):
        self.dataloader = dataloader
        self._name = name
        self._writer = SummaryWriter(log_dir)
        self._softmax_model = softmax_model
        self._steps_per_epoch = steps_per_epoch

    def __call__(
        self, model, output_path: str = None, epoch: int = -1, steps: int = -1
    ) -> float:
        total = 0
        correct = 0

        for prediction, label_ids in predict_with_classifier(
            self.dataloader, self._softmax_model, model
        ):
            total += prediction.size(0)
            correct += prediction.eq(label_ids).sum().item()
        accuracy = correct / total

        name = self._name
        if epoch == -1:
            epoch = 0
            steps = 0
            name = f"eval_{self._name}"
        elif steps == -1:
            steps = self._steps_per_epoch
        self._writer.add_scalar(name, accuracy, epoch * self._steps_per_epoch + steps)

        return accuracy


def predict_with_classifier(
    dataloader: DataLoader,
    classifier: torch.nn.Module,
    model: SentenceTransformer,
) -> Iterable[tuple[torch.Tensor, torch.Tensor]]:
    """Predicts transformation using a sentence_transformers softmax classifier."""
    model.eval()

    dataloader.collate_fn = model.smart_batching_collate
    for _, batch in enumerate(dataloader):
        features, label_ids = batch
        for idx in range(len(features)):
            features[idx] = batch_to_device(features[idx], model.device)
        label_ids = label_ids.to(model.device)
        with torch.no_grad():
            _, prediction = classifier(features, labels=None)

        yield torch.argmax(prediction, dim=1), label_ids


@dataclass
class TransformationPredictionData:
    sent_pairs: list[InputExample]
    seed_ids: np.ndarray
    trans: np.ndarray
    label_encoder: preprocessing.LabelEncoder
    sents: list[CostraSentence]


def create_transformation_prediction_data(
    corpus: list[CostraSentence],
) -> TransformationPredictionData:
    """Generates pairs of sentences with used seed id and transformation used."""
    seed_sents = {sent.seed_id: sent for sent in corpus if sent.trans == "seed"}
    label_encoder = preprocessing.LabelEncoder()
    all_transformations = list(set((sent.trans for sent in corpus)))
    label_encoder.fit(all_transformations)

    input_examples = []
    seeds = []
    trans = []
    sents = []
    for sent in corpus:
        seed_sent = seed_sents[sent.seed_id]
        transform = label_encoder.transform([sent.trans])[0]
        sent_pair = InputExample(
            texts=[sent.text, seed_sent.text],
            label=transform,
        )
        input_examples.append(sent_pair)
        seeds.append(sent.seed_id)
        trans.append(transform)
        sents.append(sent)

    return TransformationPredictionData(
        sent_pairs=input_examples,
        seed_ids=np.array(seeds),
        trans=np.array(trans),
        label_encoder=label_encoder,
        sents=sents,
    )


def train_with_transformation_prediction(
    *,
    model: SentenceTransformer,
    train_data: DataLoader,
    test_data: DataLoader,
    num_labels: int,
    epochs: int,
    log_dir: Optional[str] = None,
    verbose: bool = False,
) -> torch.nn.Module:
    train_loss = losses.SoftmaxLoss(
        model,
        sentence_embedding_dimension=cast(
            int, model.get_sentence_embedding_dimension()
        ),
        num_labels=num_labels,
    )

    epoch_steps = len(train_data)
    evaluator = None
    if log_dir is not None:
        train_evaluator = TBLabelAccuracyEvaluator(
            train_data,
            steps_per_epoch=epoch_steps,
            softmax_model=train_loss,
            name="accuracy",
            log_dir=os.path.join(log_dir, "train"),
        )
        test_evaluator = TBLabelAccuracyEvaluator(
            test_data,
            steps_per_epoch=epoch_steps,
            softmax_model=train_loss,
            name="accuracy",
            log_dir=os.path.join(log_dir, "test"),
        )
        evaluator = SequentialEvaluator([train_evaluator, test_evaluator])

    model.fit(
        [(train_data, train_loss)],
        epochs=epochs,
        evaluator=evaluator,
        evaluation_steps=math.floor(epoch_steps / 2),
        show_progress_bar=verbose,
    )

    return train_loss
