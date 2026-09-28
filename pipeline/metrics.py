# Ishaan

"""Every number in the results tables comes from these functions.

Baselines, the hard CNN and the soft CNN are all scored by the same code, so
a difference between them can't come from two slightly different formulas.

Inputs, always per NODULE:
    probs  (N, 5) float   predicted probability of rating 1..5 (rows sum to 1)
    hard   (N,)   int     median reader rating, 1..5
    soft   (N, 5) float   share of readers who gave each rating
"""

import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, cohen_kappa_score,
                             f1_score, roc_auc_score)

EPS = 1e-7


def suspicious(hard):
    """Binary ground truth: median rating 4-5 = suspicious, 1-3 = not."""
    return (np.asarray(hard) >= 4).astype(int)


def p_suspicious(probs): 
    """Binary score from the 5-class model: P(4) + P(5). Metric for soft label model on how likely the nodule is greater than or equal to malignancy rating 4"""
    return probs[:, 3] + probs[:, 4]


def ece(probs, hard, n_bins=10):
    """Expected calibration error, top-label version.

    Bin nodules by the model's confidence (its top probability). In each bin
    compare average confidence with how often the top class was right. ECE is
    the size-weighted average gap: 0 = says 70% and is right 70% of the time.
    """
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) + 1) == hard
    edges = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return total


def five_class(probs, hard, soft):
    """Table 1, the 5-class rows."""
    probs = np.clip(probs, EPS, 1)
    pred = probs.argmax(axis=1) + 1
    return {
        "accuracy": accuracy_score(hard, pred),
        "balanced_accuracy": balanced_accuracy_score(hard, pred),
        "macro_f1": f1_score(hard, pred, average="macro", labels=[1, 2, 3, 4, 5], zero_division=0),
        # ordinal agreement: calling a 5 a 4 is punished far less than calling it a 1
        "quadratic_kappa": cohen_kappa_score(hard, pred, weights="quadratic", labels=[1, 2, 3, 4, 5]),
        "mae_rating": np.abs(pred - hard).mean(),
        # how well the predicted distribution matches the median...
        "nll_vs_median": -np.log(probs[np.arange(len(hard)), hard - 1]).mean(),
        # ...and the whole panel of readers: THE soft-label metric
        "ce_vs_readers": -(soft * np.log(probs)).sum(axis=1).mean(),
        "ece": ece(probs, hard),
    }


def youden_threshold(score, y):
    """Threshold maximising sensitivity + specificity - 1. Pick it on VAL, apply it to TEST."""
    cands = np.unique(score)
    best_t, best_j = 0.5, -1
    for t in cands:
        pred = score >= t
        sens = pred[y == 1].mean()
        spec = (~pred[y == 0]).mean()
        if sens + spec - 1 > best_j:
            best_t, best_j = t, sens + spec - 1
    return float(best_t)


def binary(score, y, threshold):
    """Table 1, the suspicious-vs-not rows. AUC needs no threshold; the rest do."""
    pred = (score >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    sens = tp / max(tp + fn, 1)
    ppv = tp / max(tp + fp, 1)
    return {
        "auc": roc_auc_score(y, score),
        "threshold": threshold,
        "sensitivity": sens,                    # = recall
        "specificity": tn / max(tn + fp, 1),
        "precision": ppv,                       # = PPV
        "npv": tn / max(tn + fn, 1),
        "f1": 2 * ppv * sens / max(ppv + sens, EPS),
        "binary_accuracy": (tp + tn) / len(y),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def confidence_by_spread(probs, spread):
    """Figure 1: mean top probability for each level of reader disagreement."""
    conf = probs.max(axis=1)
    return {int(s): (float(conf[spread == s].mean()), int((spread == s).sum()))
            for s in np.unique(spread)}



def bootstrap_diff(metric, a_runs, b_runs, hard, soft, n=2000, seed=0):
    """95% CI for metric(model A) - metric(model B) on the same test nodules.

    a_runs, b_runs: (seeds, N, 5) -- the test predictions of every seed of each model.
    metric: function(probs, hard, soft) -> float, e.g.
            lambda p, h, s: five_class(p, h, s)["ce_vs_readers"]
    Each resample draws N nodules with replacement, scores every seed of both
    models on THOSE nodules, averages over seeds, and takes A - B. Paired, so
    how hard the drawn nodules happen to be cancels out. If the interval
    contains 0, you can't claim a difference.
    """
    rng = np.random.default_rng(seed)
    N = len(hard)
    diffs = []
    for _ in range(n):
        i = rng.integers(0, N, N)
        a = np.mean([metric(p[i], hard[i], soft[i]) for p in a_runs])
        b = np.mean([metric(p[i], hard[i], soft[i]) for p in b_runs])
        diffs.append(a - b)
    return float(np.mean(diffs)), tuple(np.percentile(diffs, [2.5, 97.5]).round(4))
