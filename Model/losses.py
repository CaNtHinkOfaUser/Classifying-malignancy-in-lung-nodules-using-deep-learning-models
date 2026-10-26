# Written by Claude for Ishaan

"""Class-weighted cross-entropy, for Model/no_folds/train_no_fold.py --loss weighted.

898 of the 1,885 nodules are rated 3, so plain cross-entropy is rewarded for predicting 3.
Weighting each rating by how rare it is in the training nodules pushes the model towards
ratings 1, 2, 4 and 5. Takes a class index (hard) or 5 probabilities (soft), like
nn.CrossEntropyLoss, so Models A and B are weighted the same way.
"""

import torch
import torch.nn as nn


def class_weights(hard, power):
    """(1 / frequency of each rating in the TRAINING nodules) ** power, scaled to average 1.

    hard: class indices 0-4 of the training nodules. power 0.5 = sqrt (mild), 1 = full inverse.
    """
    counts = torch.bincount(hard, minlength=5).float()
    w = (counts.sum() / counts) ** power
    return w / w.mean()


def make_loss(name, train_hard, power=0.5):
    """The loss for --loss NAME, and the class weights it uses (None for plain cross-entropy)."""
    if name == "weighted": # from the training nodules' median ratings, for hard AND soft runs
        weights = class_weights(train_hard, power)
        return nn.CrossEntropyLoss(weight=weights), weights
    return nn.CrossEntropyLoss(), None
