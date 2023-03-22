# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.3.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---
# %%
import math
import os
import random
from datetime import datetime
from typing import cast

import torch
from sentence_transformers import InputExample, SentenceTransformer, losses
from sentence_transformers.evaluation import (SentenceEvaluator,
                                              SequentialEvaluator)
from sentence_transformers.util import batch_to_device
from sklearn import preprocessing
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

import sentence_topology as st
from sentence_topology.data_types import CostraEmbedding

# %%

corpus = st.utils.load_corpus("../data/COSTRA1.1.tsv")
corpus = list(corpus)
# %%

all_seed_ids = set(map(lambda sent: sent.seed_id, corpus))
# %%
print(len(all_seed_ids))
# %%
print(all_seed_ids)
# %%
NUM_SPLITS = 5
# TODO: Use sklearn.model_selection.KFold for generating splits?
# %%
shuffled_seed_ids = list(all_seed_ids)
random.shuffle(shuffled_seed_ids)
# %%
split_len = math.floor(len(all_seed_ids) / NUM_SPLITS)
surplus = len(all_seed_ids) % NUM_SPLITS

seed_ids_splits = []
for split_ind in range(NUM_SPLITS):
    begin_ind = split_ind * split_len
    end_ind = (split_ind + 1) * split_len
    end_ind = end_ind if surplus == 0 else end_ind + 1

    surplus = max(0, surplus - 1)
    seed_ids_splits.append(shuffled_seed_ids[begin_ind:end_ind])

# %%
for split in seed_ids_splits:
    print(split, end=" ")
    print(len(split))
    print()
# %%
seed_sents = {sent.seed_id: sent for sent in corpus if sent.trans == "seed"}
label_encoder = preprocessing.LabelEncoder()
all_transformations = list(set((sent.trans for sent in corpus)))
label_encoder.fit(all_transformations)

print(label_encoder.classes_)

print(label_encoder.transform(["seed"]))

# %%
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
        model.eval()
        total = 0
        correct = 0

        self.dataloader.collate_fn = model.smart_batching_collate
        for _, batch in enumerate(self.dataloader):
            features, label_ids = batch
            for idx in range(len(features)):
                features[idx] = batch_to_device(features[idx], model.device)
            label_ids = label_ids.to(model.device)
            with torch.no_grad():
                _, prediction = self._softmax_model(features, labels=None)

            total += prediction.size(0)
            correct += torch.argmax(prediction, dim=1).eq(label_ids).sum().item()
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


# %%

log_dir = os.path.join("log", f"{datetime.now().strftime('%Y-%m-%d_%H%M%S')}")

for split_ind in range(NUM_SPLITS):
    test_seed_ids = seed_ids_splits[split_ind]

    test_sents = []
    train_data = []
    test_data = []
    for sent in corpus:
        if sent.seed_id in test_seed_ids:
            test_sents.append(sent)

        seed_sent = seed_sents[sent.seed_id]
        sent_pair = InputExample(
            texts=[sent.text, seed_sent.text],
            label=label_encoder.transform([sent.trans])[0],
        )

        if sent.seed_id in test_seed_ids:
            test_data.append(sent_pair)
        else:
            train_data.append(sent_pair)

    train_data = DataLoader(train_data, shuffle=True, batch_size=6)
    test_data = DataLoader(test_data, batch_size=6)

    epoch_steps = len(train_data)

    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    train_loss = losses.SoftmaxLoss(
        model,
        sentence_embedding_dimension=cast(
            int, model.get_sentence_embedding_dimension()
        ),
        num_labels=len(all_transformations),
    )

    train_evaluator = TBLabelAccuracyEvaluator(
        train_data,
        steps_per_epoch=epoch_steps,
        softmax_model=train_loss,
        name=f"accuracy_{split_ind}",
        log_dir=os.path.join(log_dir, "train"),
    )
    test_evaluator = TBLabelAccuracyEvaluator(
        test_data,
        steps_per_epoch=epoch_steps,
        softmax_model=train_loss,
        name=f"accuracy_{split_ind}",
        log_dir=os.path.join(log_dir, "test"),
    )
    evaluator = SequentialEvaluator([train_evaluator, test_evaluator])
    model.fit(
        [(train_data, train_loss)],
        epochs=5,
        evaluator=evaluator,
        evaluation_steps=math.floor(epoch_steps / 2),
    )

    all_texts = [sent.text for sent in test_sents]
    all_embeds = model.encode(all_texts, show_progress_bar=True)

    embeds = []
    for sent, embed in zip(test_sents, all_embeds):
        embeds.append(CostraEmbedding(sent.id, sent.seed_id, sent.trans, embed))
    st.utils.save_embeddings(
        embeds,
        f"../embeddings/paraphrase-multilingual-MiniLM-L12-v2_supervised_{split_ind}",
    )

# %%
import matplotlib.pyplot as plt

# %%
