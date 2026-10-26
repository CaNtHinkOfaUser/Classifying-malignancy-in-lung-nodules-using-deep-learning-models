# Written by Claude for Ishaan

"""Class-weighted cross-entropy, for Model/no_folds/train_no_fold.py --loss weighted.

898 of the 1,885 nodules are rated 3, so plain cross-entropy is rewarded for predicting 3.
Weighting each rating by how rare it is in the training nodules pushes the model towards
ratings 1, 2, 4 and 5. Takes a class index (hard) or 5 probabilities (soft), like
nn.CrossEntropyLoss.

Each model is weighted by how often each rating occurs in its OWN training targets:
    Model A (hard): the median ratings      -> rating 3 is 47.7% of the training nodules
    Model B (soft): the readers' shares     -> rating 3 is 38.9% of the readers' ratings
The median over-represents rating 3, so weights from the median ratings would push Model B
away from 3 far more than its own targets justify.
"""

import torch
import torch.nn as nn


def class_weights(counts, power):
    """(1 / frequency of each rating) ** power, scaled to average 1.

    counts: how often each rating 1-5 occurs in the training targets (5 numbers).
    power 0.5 = sqrt (mild), 1 = full inverse.
    """
    counts = counts.float()
    w = (counts.sum() / counts) ** power
    return w / w.mean()


def rating_counts(labels, train_hard, train_soft):
    """How often each rating occurs in the training targets of a `labels` model.

    hard: each nodule counts once, for its median rating (class indices 0-4).
    soft: each nodule spreads its 1 over the ratings its readers gave, so a nodule
          rated 2, 3, 3 adds 1/3 to rating 2 and 2/3 to rating 3.
    """
    if labels == "hard":
        return torch.bincount(train_hard, minlength=5)
    return train_soft.sum(dim=0)


def make_loss(name, labels, train_hard, train_soft, power=0.5):
    """The loss for --loss NAME, and the class weights it uses (None for plain cross-entropy)."""
    if name == "weighted":
        weights = class_weights(rating_counts(labels, train_hard, train_soft), power)
        return nn.CrossEntropyLoss(weight=weights), weights
    return nn.CrossEntropyLoss(), None
